import numpy as np
import pandas as pd

from typing import List
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV

from src.preprocess import multilabel_onehot_encode
from src.plots import plot_feature_importance
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
    # Calcula a média de dias entre eventos do mesmo tipo
    dataf["delta_days_between_event"] = dataf.groupby(["account_id", "event"])["time_since_test_start"].diff().fillna(0)
    avg_days_between_event = (
        dataf.groupby(["account_id", "event"])["delta_days_between_event"].mean()
        .reset_index(name="avg_days_between_events")
    )
    # Pivota os resultados para uma linha por `account_id``
    avg_days_pivot = avg_days_between_event.pivot(
        index="account_id", columns="event", values="avg_days_between_events"
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
    ).drop(columns="delta_days_between_event")

    return dataf


def calculate_days_between_receiving_viewing(data: pd.DataFrame) -> pd.DataFrame:
    """
    """
    dataf = data.copy()
    # Consultando eventos de interesse
    received = dataf[dataf["event"] == "1-offer received"]
    viewed = dataf[dataf["event"] == "2-offer viewed"]

    # Alinhando eventos
    pairs = pd.merge(
        received[["account_id", "offer_id", "time_since_test_start"]],
        viewed[["account_id", "offer_id", "time_since_test_start"]],
        on=["account_id", "offer_id"],
        how="left",
        suffixes=("_received", "_viewed")
    )
    # Diferença de dias entre receber e vizualizar
    pairs["delta_received_view"] = pairs["time_since_test_start_viewed"] - pairs["time_since_test_start_received"]

    avg_days_between_received_view = (
        pairs.groupby("account_id")["delta_received_view"].mean()
        .reset_index(name="avg_days_between_received_view")
    )
    dataf = dataf.merge(avg_days_between_received_view, on="account_id", how="left")

    return dataf


def build_customer_features(data: pd.DataFrame, agg_columns: List[str]) -> pd.DataFrame:
    """
    """
    # Features do cliente
    customer_feats = data.groupby(agg_columns).agg(
        age=("age", "first"), 
        gender=("gender", "first"), 
        credit_card_limit=("credit_card_limit", "first"),
        registered_on=("registered_on", "first"),
        n_offers=("offer_id", "count"),                                    # número de ofertas
        unique_offers=("offer_id", "nunique"),                             # ofertas únicas
        avg_ticket=("amount", "mean"),                                     # ticket médio
        total_amount=("amount", "sum"),                                    # gasto total
        days_to_first_interaction=("time_since_test_start", "min"),        # dias para primeiro evento
        day_of_last_interaction=("time_since_test_start", "max"),          # dia do último evento
        avg_days_offers_received=("avg_days_offers_received", "first"),    # média de dias entre ofertas recebidas
        avg_days_offers_viewed=("avg_days_offers_viewed", "first"),        # média de dias entre ofertas vizualizadas
        avg_days_offers_completed=("avg_days_offers_completed", "first")   # média de dias entre ofertas completadas
    ).reset_index()
    
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
    offer_feats = pd.get_dummies(offer_feats, columns=["offer_type"]) # Obs: Remover um label caso seja usado modelo de regressão
    event_encoded = multilabel_onehot_encode(data=offer_feats, column="event")
    channels_encoded = multilabel_onehot_encode(data=offer_feats, column="channels")

    return pd.concat(
        [offer_feats.drop(columns=["channels", "event"]), event_encoded, channels_encoded]
        , axis=1
    )


def build_engagement_features(data: pd.DataFrame, agg_columns: List[str]) -> pd.DataFrame:
    """
    """
    # Features de engajamento
    engagement_feats = data.groupby(agg_columns).agg(
        pct_type_bogo=("offer_type_bogo", "mean"),                     # % de ofertas tipo bogo
        pct_type_discount=("offer_type_discount", "mean"),             # % de ofertas tipo discount
        pct_type_informational=("offer_type_informational", "mean"),   # % de ofertas tipo informational
        pct_viewed_offers=("event_2-offer viewed", "mean"),            # % de ofertas vizualizadas
        pct_completed_offers=("event_3-offer completed", "mean"),      # % de ofertas completadas
        pct_channel_email=("channels_email", "mean"),                  # % de ofertas recebidas via email
        pct_channel_mobile=("channels_mobile", "mean"),                # % de ofertas recebidas via mobile
        pct_channel_social=("channels_social", "mean"),                # % de ofertas recebidas via social media
        pct_channel_web=("channels_web", "mean"),                      # % de ofertas recebidas via web
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