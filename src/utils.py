import numpy as np
import pandas as pd

from typing import List, Dict
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, precision_score
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV

from src.preprocess import multilabel_onehot_encode
from src.plots import plot_feature_importance, plot_roc_auc_curve

from loguru import logger


def built_target(data: pd.DataFrame) -> pd.DataFrame:
    """
    """
    dataf = data.copy()
    # Target indica se a oferta foi bem sucedida
    # Para ofertas informacionais, são bem sucedidas as transações após visualizações
    # Para ofertas de bogo e desconto, são bem sucedidas transações após satisfazer requisito da oferta
    dataf["target"] = np.where(
        ((dataf["offer_type"] == "informational") & (dataf["event"] == "4-transaction")) |
        ((dataf["event"] == "4-transaction") & (dataf["transaction_reward"].notna())),
        1, 0
    )
    return dataf


def rolling_agg(series, values, agg_func, window_size):
    """Aplica agregação rolling baseada em dias, com shift para evitar vazamento."""
    result = []
    for i in range(len(series)):
        t0 = series.iloc[i]
        mask = (series < t0) & (series >= t0 - window_size)
        vals = values[mask]
        result.append(agg_func(vals) if len(vals) > 0 else np.nan)
        
    return pd.Series(result, index=series.index)


def calculate_days_between_same_events(data: pd.DataFrame, event_windows: dict) -> pd.DataFrame:
    dataf = data.copy()

    # Garante ordenação
    dataf = dataf.sort_values(by=["account_id", "event", "time_since_test_start"]).reset_index()

    result_rows = []

    for (account, event), group in dataf.groupby(["account_id", "event"]):
        window_size = event_windows.get(event)
        if window_size is None:
            continue  # pula eventos que não estão no dicionário

        group = group.copy()
        times = group["time_since_test_start"].values
        indices = group["index"].values
        avg_diffs = []

        for i in range(len(times)):
            t0 = times[i]

            # eventos anteriores dentro da janela de 15 dias
            mask = (times < t0) & (times >= t0 - window_size)
            prev_times = times[mask]

            # calcula diffs consecutivos
            if len(prev_times) >= 2:
                deltas = np.diff(prev_times)
                avg_diffs.append(np.mean(deltas))
            else:
                avg_diffs.append(np.nan)

        col_ = event.split("-")[1]
        # salva os resultados diretamente
        temp_result = pd.DataFrame({
            "index": indices,
            f"avg_days_between_event_{col_}_{window_size}d": avg_diffs
        })
        result_rows.append(temp_result)

    # junta os resultados
    final = pd.concat(result_rows).set_index("index").sort_index()

    # junta ao dataframe original
    dataf = dataf.set_index("index")
    dataf = pd.concat([dataf, final], axis=1).reset_index(drop=True)

    return dataf


def calculate_days_between_receiving_viewing(data: pd.DataFrame, agg_columns, window_size: int = 30) -> pd.DataFrame:
    dataf = data.copy()
    dataf = dataf.sort_values(by=agg_columns + ["time_since_test_start"]).reset_index(drop=False)  # preserva índice original

    results = []

    for group_keys, g in dataf.groupby(agg_columns):
        g = g.copy()

        # Separando eventos
        received = g[g["event"] == "1-offer received"].copy()
        viewed = g[g["event"] == "2-offer viewed"].copy()

        received = received.rename(columns={"time_since_test_start": "received_time"})
        viewed = viewed.rename(columns={"time_since_test_start": "viewed_time"})

        received = received.sort_values("received_time")
        viewed = viewed.sort_values("viewed_time")

        # Faz merge por offer_id (mesmo grupo)
        pairs = pd.merge_asof(
            received,
            viewed,
            left_on="received_time",
            right_on="viewed_time",
            by=["offer_id"],
            direction="forward",
            suffixes=("_received", "_viewed")
        )
        # Calcula delta
        pairs["delta_received_view"] = pairs["viewed_time"] - pairs["received_time"]
        pairs = pairs[pairs["delta_received_view"] >= 0].copy()

        # Rolling média para cada ponto de tempo no grupo
        rolling_avg = []
        for i in range(len(g)):
            t0 = g.iloc[i]["time_since_test_start"]
            mask = (pairs["received_time"] < t0) & (pairs["received_time"] >= t0 - window_size)
            avg_delta = pairs.loc[mask, "delta_received_view"].mean() if not pairs.loc[mask].empty else np.nan
            rolling_avg.append(avg_delta)

        g["avg_days_received_view_30d"] = rolling_avg
        results.append(g)

    final_df = pd.concat(results, axis=0)
    final_df = final_df.sort_values("index").set_index("index").sort_index()  # restaura índice original

    return final_df.reset_index(drop=True)


def build_customer_features(data: pd.DataFrame, agg_columns: List[str], window_size: int = 7) -> pd.DataFrame:
    """
    """
    dataf = data.copy()
    dataf = dataf.sort_values(by=agg_columns + ["time_since_test_start"])

    grouped = dataf.groupby(agg_columns)

    dataf[f"avg_ticket_{window_size}d"] = grouped.apply(lambda g: rolling_agg(
        g["time_since_test_start"], g["amount"], np.mean, window_size
    )).reset_index(level=0, drop=True).T

    dataf[f"total_amount_{window_size}d"] = grouped.apply(lambda g: rolling_agg(
        g["time_since_test_start"], g["amount"], np.sum, window_size
    )).reset_index(level=0, drop=True).T

    # One-hot encoding feature
    dataf = pd.get_dummies(dataf, columns=["gender"], dummy_na=True) 
    dataf = dataf.sort_values(["account_id", "time_since_test_start", "event"]).reset_index(drop=True)
    
    return dataf


def build_offer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    """
    # Features da oferta
    # One-hot encoding features
    # Obs: Remover um label caso seja usado modelo de regressão
    offer_feats = pd.get_dummies(data, columns=["offer_type"]) 
    offer_feats = pd.get_dummies(offer_feats, columns=["event"])
    channels_encoded = multilabel_onehot_encode(data=offer_feats, column="channels")

    offer_feats = pd.concat(
        [offer_feats.drop(columns=["channels"]), channels_encoded]
        , axis=1
    )
    return offer_feats.sort_values(["account_id", "time_since_test_start"]).reset_index(drop=True)


def build_engagement_features(data: pd.DataFrame, agg_columns: List[str], window_size: int = 15) -> pd.DataFrame:
    dataf = data.copy()
    dataf = dataf.sort_values(by=agg_columns + ["time_since_test_start"])

    features = {
        f"pct_type_bogo_{window_size}d": "offer_type_bogo",
        f"pct_type_discount_{window_size}d": "offer_type_discount",
        f"pct_type_informational_{window_size}d": "offer_type_informational",
        f"pct_viewed_offers_{window_size}d": "event_2-offer viewed",
        f"pct_completed_offers_{window_size}d": "event_3-offer completed",
        f"pct_channel_email_{window_size}d": "channels_email",
        f"pct_channel_mobile_{window_size}d": "channels_mobile",
        f"pct_channel_social_{window_size}d": "channels_social",
        f"pct_channel_web_{window_size}d": "channels_web",
    }
    grouped = dataf.groupby(agg_columns)

    for new_col, base_col in features.items():
        dataf[new_col] = grouped.apply(lambda g: rolling_agg(
            g["time_since_test_start"], g[base_col], np.mean, window_size
        )).reset_index(level=0, drop=True).T

    return dataf


def fill_with_mean(data: pd.DataFrame, features: List[str]):
    dataf = data.copy() 
    dataf[features] = dataf.groupby("account_id")[features].transform(lambda x: x.fillna(x.mean()))
    
    return dataf


def aggregate_modeling_dataset(data: pd.DataFrame, agg_dict: Dict[str, str], feats_to_fill: List[str]):
    model_feats = data.copy()

    # Ajusta nome das colunas
    model_feats.columns = [col.replace(" ", "_") for col in model_feats.columns]
    
    for col in feats_to_fill:
        model_feats[col + "_is_missing"] = model_feats[col].isnull().astype(int)

    # Agrega features
    model_feats = model_feats.groupby(["account_id", "offer_id"]).agg(agg_dict).reset_index()    
    model_feats = fill_with_mean(data=model_feats, features=feats_to_fill)

    return model_feats


def uppercut_features(
        df: pd.DataFrame, features, lower_q: float = 0.01, upper_q: float = 0.99
    ) -> pd.DataFrame:
    """
    """
    df_copy = df.copy()

    for feature in features:
        lower = df_copy[feature].quantile(lower_q)
        upper = df_copy[feature].quantile(upper_q)
        df_copy[feature] = df_copy[feature].clip(lower, upper)

    return df_copy


def get_baseline_logloss(y: pd.Series):
    """
    """
    p0 = y.value_counts(dropna=False, normalize=True).get(0)  # negative class proportion
    p1 = y.value_counts(dropna=False, normalize=True).get(1)  # positive class proportion

    baseline_log_loss = -(p1 * np.log(p1) + p0 * np.log(p0))
    logger.info(f"Baseline LogLoss={baseline_log_loss:.4f}")


def calculate_precision_recall_at_k(y_true, y_pred_proba, k=100):

    df = pd.DataFrame({"y_true": y_true, "y_score": y_pred_proba})
    df_sorted = df.sort_values("y_score", ascending=False).head(k)

    precision_at_k = df_sorted["y_true"].sum() / k
    recall_at_k = df_sorted["y_true"].sum() / df["y_true"].sum()

    logger.info(f"Precision@{k} = {precision_at_k:.4f}")
    logger.info(f"Recall@{k} = {recall_at_k:.4f}")


def precision_recall_at_threshold(y_true, y_proba, threshold=0.5):
    """
    """
    y_pred = (y_proba >= threshold).astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    logger.info(f"Precision@thr[{threshold:.2f}] = {precision:.4f}")
    logger.info(f"Recall@thr[{threshold:.2f}] = {recall:.4f}")
    
    return precision, recall


def feature_selection(
        data: pd.DataFrame, 
        features: List[str], 
        target: str
    ) -> pd.DataFrame:
    """
    """
    X, y = data[features], data[target]

    train_end = int(len(X) * 0.6) # 60% para treino
    calib_end = int(len(X) * 0.8) # 20% para calibração e 20% para teste

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end] # Conjunto para treinar
    X_calib, y_calib = X.iloc[train_end:calib_end], y.iloc[train_end:calib_end] # Conjunto para calibrar
    X_valid, y_valid = X.iloc[calib_end:], y.iloc[calib_end:] # Conjunto para testar

    params_ = {
        "max_depth": 3,
        "num_leaves": 8,
        "n_estimators": 100,
        "learning_rate": 0.05,
    }

    lgbm = LGBMClassifier(**params_, random_state=42, verbosity=-1)
    lgbm.fit(X_train, y_train)

    # Calibrando o modelo
    model_calib = CalibratedClassifierCV(lgbm, method="isotonic", cv="prefit")
    model_calib.fit(X_calib, y_calib)

    y_train_pred = model_calib.predict_proba(X_train)[:, 1]
    y_test_pred = model_calib.predict_proba(X_valid)[:, 1]

    auc_train = roc_auc_score(y_train, y_train_pred)
    auc_test = roc_auc_score(y_valid, y_test_pred)

    logger.info(f"Train AUC: {auc_train:.4f}")
    logger.info(f"Valid AUC: {auc_test:.4f}")

    # Calculando a importância por permutação
    perm_importance = permutation_importance(
        lgbm, X_valid, y_valid, n_repeats=30, random_state=42, scoring="roc_auc"
    )
    # Ordenando as variáveis por importância média
    sorted_importances = sorted(
        zip(X_train.columns, perm_importance.importances_mean), 
        key=lambda x: x[1], 
        reverse=True
    )

    feats_neutral = {}
    feats_positive = {}
    feats_negative = {}

    for feature, importance in sorted_importances:    
        if importance > 0:
            feats_positive[feature] = importance
        elif importance == 0:
            feats_neutral[feature] = importance
        else:
            feats_negative[feature] = importance

    logger.success("POSITIVE IMPORTANCE:")
    for feat, impact in feats_positive.items():
        logger.success(f"{impact:.6f}: {feat}")

    logger.debug("NO IMPORTANCE:")
    for feat, impact in feats_neutral.items():
        logger.debug(f"{impact:.6f}: {feat}")
    
    logger.warning("NEGATIVE IMPORTANCE:")
    for feat, impact in feats_negative.items():
        logger.warning(f"{impact:.6f}: {feat}")

    plot_feature_importance(X=X_train, importance_values=perm_importance)
    
    return list(feats_positive.keys())


def get_roc_auc_curve(y, y_pred):
    fpr, tpr, thresholds = roc_curve(y, y_pred)

    gmeans = np.sqrt(tpr * (1 - fpr))
    best_idx = np.argmax(gmeans)

    # Finding optimal threshold
    optimal_threshold = thresholds[best_idx]
    best_auc = gmeans[best_idx]

    binary = [1 if pred > optimal_threshold else 0 for pred in y_pred]

    # Complementary metrics
    precision = precision_score(y, binary)
    recall = recall_score(y, binary)

    logger.info(" Resulting metrics based on the optimal auc threshold:")
    logger.info(f"Threshold={optimal_threshold:.4f}")
    logger.info(f"AUC={best_auc:.4f}")
    logger.info(f"Precision={precision:.4f}")
    logger.info(f"Recall={recall:.4f}")

    plot_roc_auc_curve(fpr, tpr, best_idx)
    logger.info(f"Percentual de oportunidades aproveitadas: {sum(y_pred >= optimal_threshold) / len(y_pred):.2%}")
    return optimal_threshold


def evaluate_potential(data: pd.DataFrame, threshold: float):
    total_clientes = data["account_id"].nunique()
    clientes_com_oferta = data[data['pred'] > threshold]["account_id"].nunique()
    total_ofertas = data["offer_id"].nunique()

    combinacoes_possiveis = total_clientes * total_ofertas
    combinacoes_testadas = data.drop_duplicates(["account_id", "offer_id"]).shape[0]
    combinacoes_aprovadas = data[data["pred"] > threshold].drop_duplicates(["account_id", "offer_id"]).shape[0]

    pct_testado = combinacoes_testadas / combinacoes_possiveis
    pct_aprovacao = combinacoes_aprovadas / combinacoes_testadas
    pct_clientes_enviados = clientes_com_oferta / total_clientes

    logger.info(f"% de combinações cliente-oferta testadas: {pct_testado:.2%}")
    logger.info(f"% de combinações cliente-oferta aprovadas: {pct_aprovacao:.2%}")
    logger.info(f"% clientes com pelo menos uma oferta sugerida: {pct_clientes_enviados:.2%}")
    logger.info(f"Potencial capturado das combinações cliente-oferta: {combinacoes_aprovadas/combinacoes_possiveis:.2%}")
