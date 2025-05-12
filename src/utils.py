import numpy as np
import pandas as pd

from typing import List
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
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


def calculate_days_between_same_events(data: pd.DataFrame) -> pd.DataFrame:
    """
    """
    dataf = data.copy()
    dataf["delta_days_between_event"] = np.nan

    # Calcula a média de dias entre eventos do mesmo tipo defasando o evento
    for (account, event), group in dataf.groupby(["account_id", "event"]):
        # Calcula as diferenças entre eventos consecutivos
        diffs = group["time_since_test_start"].diff()

        # Calcula a média cumulativa defasada para não usar eventos futuros 
        expanding_mean = diffs.expanding().mean().shift(1)
        dataf.loc[group.index, "delta_days_between_event"] = expanding_mean

    # # Calcula a média de dias entre eventos do mesmo tipo
    avg_days_between_event = (
        dataf[["account_id", "event", "delta_days_between_event"]]
        .dropna().groupby(["account_id", "event"])
        .mean().reset_index()
    )
    # Pivota os resultados para uma linha por `account_id``
    avg_days_pivot = avg_days_between_event.pivot(
        index="account_id", columns="event", values="delta_days_between_event"
    )
    # Renomeia colunas e ajusta índice
    avg_days_between_event = (
        avg_days_pivot[["1-offer received", "2-offer viewed", "3-offer completed"]]
        .rename(columns={
            "1-offer received": "avg_days_offers_received",
            "2-offer viewed": "avg_days_offers_viewed",
            "3-offer completed": "avg_days_offers_completed"
        }).reset_index()
        .rename_axis(None, axis=1)
        .reset_index(drop=True)
    )
    # Une resultado com dados originais
    dataf = dataf.merge(
        avg_days_between_event, on=["account_id"], how="left"
    )
    return dataf.drop(columns="delta_days_between_event")


def calculate_days_between_receiving_viewing(data: pd.DataFrame) -> pd.DataFrame:
    """
    """
    dataf = data.copy()
    # Consultando eventos de interesse
    received = dataf[dataf["event"] == "1-offer received"]
    viewed = dataf[dataf["event"] == "2-offer viewed"]

    received = received.rename(columns={"time_since_test_start": "received_time"})
    viewed = viewed.rename(columns={"time_since_test_start": "viewed_time"})

    # Ordenada pela coluna de tempo para o `merge_asof`
    received = received.sort_values("received_time")
    viewed = viewed.sort_values("viewed_time")

    # Mergeia alinhando a visualização imediatamente após cada recebimento
    # Estratificado por pessoa e por oferta
    pairs = pd.merge_asof(
        received,
        viewed,
        left_on="received_time",
        right_on="viewed_time",
        by=["account_id", "offer_id"],
        direction="forward",
        suffixes=("_received", "_viewed")
    )
    # Diferença de dias entre receber e vizualizar
    pairs["delta_received_view"] = pairs["viewed_time"] - pairs["received_time"]

    # Removendo casos de inconsistência
    pairs = pairs[pairs["delta_received_view"] >= 0]

    avg_days_between_received_view = (
        pairs.groupby("account_id")["delta_received_view"].mean()
        .reset_index(name="avg_days_between_received_view")
    )
    dataf = dataf.merge(avg_days_between_received_view, on="account_id", how="left")
    data = data.sort_values(["account_id", "time_since_test_start", "event"]).reset_index(drop=True)

    return dataf


def build_customer_features(data: pd.DataFrame, agg_columns: List[str]) -> pd.DataFrame:
    """
    """
    dataf = data.copy()

    # Features que precisam ser defasadas
    dataf["avg_ticket"] = dataf.groupby(agg_columns)["amount"].expanding().mean().shift(1).reset_index(level=0, drop=True)  
    dataf["total_amount"] = dataf.groupby(agg_columns)["amount"].expanding().sum().shift(1).reset_index(level=0, drop=True)   

    dataf["days_to_first_interaction"] = dataf.groupby(agg_columns)["time_since_test_start"].transform(lambda x: x.expanding().min().shift(1))
    dataf["day_of_last_interaction"] = dataf.groupby(agg_columns)["time_since_test_start"].transform(lambda x: x.expanding().max().shift(1))

    # Features do cliente
    customer_feats = dataf.groupby(agg_columns).agg(
        age=("age", "first"), 
        gender=("gender", "first"), 
        credit_card_limit=("credit_card_limit", "first"),
        registered_on=("registered_on", "first"),
        unique_offers=("offer_id", "nunique"),                               # ofertas únicas
        avg_ticket=("avg_ticket", "last"),                                   # ticket médio
        total_amount=("total_amount", "last"),                               # gasto total
        days_to_first_interaction=("days_to_first_interaction", "first"),    # dias para o primeiro evento
        day_of_last_interaction=("day_of_last_interaction", "last"),         # dia do último evento
        avg_days_offers_received=("avg_days_offers_received", "first"),      # média de dias entre ofertas recebidas
        avg_days_offers_viewed=("avg_days_offers_viewed", "first"),          # média de dias entre ofertas vizualizadas
        avg_days_offers_completed=("avg_days_offers_completed", "first")     # média de dias entre ofertas completadas
    ).reset_index()

    # One-hot encoding feature
    customer_feats = pd.get_dummies(customer_feats, columns=["gender"], dummy_na=True) 
    
    return customer_feats


def build_offer_features(data: pd.DataFrame, agg_columns: List[str]) -> pd.DataFrame:
    """
    """
    # Features da oferta
    offer_feats = data.groupby(agg_columns).agg(
        offer_type=("offer_type", "first"),
        avg_days_between_received_view=("avg_days_between_received_view", "first"),
        event=("event", "unique"),
        channels=("channels", "first"),
        n_canais=("channels", lambda x: x.explode().nunique()),
    ).reset_index()

    # One-hot encoding features
    # Obs: Remover um label caso seja usado modelo de regressão
    offer_feats = pd.get_dummies(offer_feats, columns=["offer_type"]) 
    event_encoded = multilabel_onehot_encode(data=offer_feats, column="event")
    channels_encoded = multilabel_onehot_encode(data=offer_feats, column="channels")

    return pd.concat(
        [offer_feats.drop(columns=["channels", "event"]), event_encoded, channels_encoded]
        , axis=1
    )


def build_engagement_features(data: pd.DataFrame, agg_columns: List[str]) -> pd.DataFrame:
    """
    """
    dataf = data.copy()

    # Percentual de cada tipo de oferta até o momento atual
    dataf["pct_type_bogo"] = dataf.groupby(agg_columns)["offer_type_bogo"].transform(lambda x: x.expanding().mean().shift(1))
    dataf["pct_type_discount"] = dataf.groupby(agg_columns)["offer_type_discount"].transform(lambda x: x.expanding().mean().shift(1))
    dataf["pct_type_informational"] = dataf.groupby(agg_columns)["offer_type_informational"].transform(lambda x: x.expanding().mean().shift(1))

    # Engajamento com eventos
    dataf["pct_viewed_offers"] = dataf.groupby(agg_columns)["event_2-offer viewed"].transform(lambda x: x.expanding().mean().shift(1))
    dataf["pct_completed_offers"] = dataf.groupby(agg_columns)["event_3-offer completed"].transform(lambda x: x.expanding().mean().shift(1))

    # Canais utilizados
    dataf["pct_channel_email"] = dataf.groupby(agg_columns)["channels_email"].transform(lambda x: x.expanding().mean().shift(1))
    dataf["pct_channel_mobile"] = dataf.groupby(agg_columns)["channels_mobile"].transform(lambda x: x.expanding().mean().shift(1))
    dataf["pct_channel_social"] = dataf.groupby(agg_columns)["channels_social"].transform(lambda x: x.expanding().mean().shift(1))
    dataf["pct_channel_web"] = dataf.groupby(agg_columns)["channels_web"].transform(lambda x: x.expanding().mean().shift(1))

    # Features de engajamento
    engagement_feats = dataf.groupby(agg_columns).agg(
        pct_type_bogo=("pct_type_bogo", "last"),                     # % de ofertas tipo bogo
        pct_type_discount=("pct_type_discount", "last"),             # % de ofertas tipo discount
        pct_type_informational=("pct_type_informational", "last"),   # % de ofertas tipo informational
        pct_viewed_offers=("pct_viewed_offers", "last"),             # % de ofertas vizualizadas
        pct_completed_offers=("pct_completed_offers", "last"),       # % de ofertas completadas
        pct_channel_email=("pct_channel_email", "last"),             # % de ofertas recebidas via email
        pct_channel_mobile=("pct_channel_mobile", "last"),           # % de ofertas recebidas via mobile
        pct_channel_social=("pct_channel_social", "last"),           # % de ofertas recebidas via social media
        pct_channel_web=("pct_channel_web", "last"),                 # % de ofertas recebidas via web
    ).reset_index()
    
    return engagement_feats


def unify_modeling_dataset(
        offer_feats: pd.DataFrame, 
        engagement_feats: pd.DataFrame, 
        customer_feats: pd.DataFrame,
        profile_offer_target: pd.DataFrame,
    ) -> pd.DataFrame:
    """
    """
    return (
        offer_feats.merge(customer_feats, on="account_id")
        .merge(engagement_feats, on="account_id")
        .merge(profile_offer_target, on=["account_id", "offer_id"])
    )


def get_baseline_logloss(y: pd.Series):
    """
    """
    p0 = y.value_counts(dropna=False, normalize=True).get(0)  # negative class proportion
    p1 = y.value_counts(dropna=False, normalize=True).get(1)  # positive class proportion

    baseline_log_loss = -(p1 * np.log(p1) + p0 * np.log(p0))
    logger.info(f"Baseline LogLoss={baseline_log_loss:.4f}")


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

    params_ = {"n_estimators": 60}

    lgbm = LGBMClassifier(**params_, random_state=42)
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

    return fpr, tpr, best_idx, optimal_threshold

    # binary = [1 if pred > optimal_threshold else 0 for pred in y_pred]


def get_roc_auc_curve(y, y_pred):
    fpr, tpr, thresholds = roc_curve(y, y_pred)

    gmeans = np.sqrt(tpr * (1 - fpr))
    best_idx = np.argmax(gmeans)

    # Finding optimal threshold
    optimal_threshold = thresholds[best_idx]
    best_auc = gmeans[best_idx]

    binary = [1 if pred > optimal_threshold else 0 for pred in y_pred]

    # Complementary metrics
    # accuracy = accuracy_score(y, binary)
    # precision = precision_score(y, binary)
    # recall = recall_score(y, binary)
    # f_score = f1_score(y, binary)
    # logloss = log_loss(y, y_pred)

    logger.info(" Resulting metrics based on the optimal threshold:")
    logger.info(f"Threshold={optimal_threshold:.4f}")
    logger.info(f"AUC={best_auc:.4f}")
    # logger.info(f"Accuracy={accuracy:.4f}")
    # logger.info(f"LogLoss={logloss:.4f}")
    # logger.info(f"Precision={precision:.4f}")
    # logger.info(f"Recall={recall:.4f}")
    # logger.info(f"F-Score={f_score:.4f}")

    plot_roc_auc_curve(fpr, tpr, best_idx)
    return optimal_threshold