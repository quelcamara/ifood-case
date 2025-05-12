import numpy as np
import pandas as pd
from typing import List, Tuple
from lightgbm import LGBMClassifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

from loguru import logger


def fit(X: pd.DataFrame, y: pd.Series, params: dict) -> LGBMClassifier:
    """
    """
    # Treinando o modelo
    lgbm = LGBMClassifier(**params)
    lgbm.fit(X, y)
    
    return lgbm


def calibrate(model: LGBMClassifier, X: pd.DataFrame, y: pd.Series) -> CalibratedClassifierCV:
    """
    """
    # Calibrando o modelo
    lgbm_calib = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    lgbm_calib.fit(X, y)

    return lgbm_calib


def validate_model(
        data: pd.DataFrame, 
        features: List[str], 
        target: str,
        params: dict,
        n_folds: int=5
    ) -> Tuple[list, list, list]:
    """
    """
    X, y = data[features], data[target]
    tscv = TimeSeriesSplit(n_splits=n_folds)

    brier_scores = []                   # para avaliar a calibração
    auc_scores, logloss_scores = [], [] # para avaliar o modelo

    for train_index, valid_index in tscv.split(X):

        X_train, X_temp = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_temp = y.iloc[train_index], y.iloc[valid_index]

        # Separando um conjunto para testar a calibração
        split_point = int(len(X_temp) * 0.7)
        X_calib, X_valid = X_temp.iloc[:split_point], X_temp.iloc[split_point:]
        y_calib, y_valid = y_temp.iloc[:split_point], y_temp.iloc[split_point:]

        lgbm = fit(X=X_train, y=y_train, params=params)
        lgbm_calib = calibrate(model=lgbm, X=X_calib, y=y_calib)

        preds = lgbm_calib.predict_proba(X_valid)[:, 1]

        logloss = log_loss(y_valid, preds)
        auc = roc_auc_score(y_valid, preds)
        brier = brier_score_loss(y_valid, preds)

        auc_scores.append(auc)
        brier_scores.append(brier)
        logloss_scores.append(logloss)
    
    return auc_scores, brier_scores, logloss_scores
    

def train_model(
        dtrain: pd.DataFrame, 
        dtest: pd.DataFrame, 
        features: List[str], 
        target: str, 
        params: dict
    ) -> Tuple[CalibratedClassifierCV, pd.DataFrame, pd.DataFrame]:
    """
    """
    logger.info(f"Running model training...")
    
    auc_scores, brier_scores, _ = validate_model(data=dtrain, features=features, target=target, params=params)
    logger.info(f"Validation Performance: AUC: {np.mean(auc_scores).round(4):.4f} | Brier: {np.mean(brier_scores).round(4):.4f}")

    X_test, y_test = dtest[features], dtest[target]
    X_train, X_calib, y_train, y_calib = train_test_split(
        dtrain[features], dtrain[target], 
        test_size=0.2, shuffle=False
    )

    lgbm = fit(X=X_train, y=y_train, params=params)
    lgbm_calib = calibrate(model=lgbm, X=X_calib, y=y_calib)

    preds = lgbm_calib.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, preds)
    brier = brier_score_loss(y_test, preds)

    logger.success(f"Test Model Performance: AUC {np.mean(auc).round(4)} | Brier: {brier:.4f}")

    return lgbm_calib, X_train, y_train, y_test, preds
