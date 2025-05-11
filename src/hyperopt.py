import optuna
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.utils import get_baseline_logloss

from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.calibration import CalibratedClassifierCV

from loguru import logger


class RollingWindowOptimizer:
    def __init__(self, data, features, target, n_folds, trials):
        self.data = data
        self.features = features
        self.target = target
        self.n_folds = n_folds
        self.trials = trials

        self.i = 1 # para log apenas

    def run(self):
        logger.info(" Optimizing...")

        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.trials)
        logger.info(" Done...")

        self.study = study
        self.best_result = study.best_value
        self.best_params = study.best_params

        # Validação com melhores parâmetros
        X, y = self.data[self.features], self.data[self.target]
        
        # 60% para treino, 20% para calibração e 20% para validação
        # Mantendo `shuffle=False` pois dados já estão ordenados pela data de registro do cliente
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False)
        X_calib, X_valid, y_calib, y_valid = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

        logger.info(" Evaluating best hyperparameters on the test set...")

        model = LGBMClassifier(**self.best_params)
        model.fit(X_train, y_train)

        # Calibrando o modelo
        model_calib = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        model_calib.fit(X_calib, y_calib)

        preds = model_calib.predict_proba(X_valid)[:, 1]

        logloss = log_loss(y_valid, preds)
        auc = roc_auc_score(y_valid, preds)
        brier = brier_score_loss(y_valid, preds)

        get_baseline_logloss(y=y_valid)
        logger.success(f"LogLoss: {logloss:.4f} | AUC: {auc:.4f} | Brier: {brier:.4f}")

    def train(self, params):
        X, y = self.data[self.features], self.data[self.target]

        tscv = TimeSeriesSplit(n_splits=self.n_folds)

        brier_scores = []                   # para avaliar a calibração
        auc_scores, logloss_scores = [] ,[] # para avaliar o modelo

        for fold, (train_index, valid_index) in enumerate(tscv.split(X)):
            # logger.info(f"=== Fold {fold + 1}/{self.n_folds} ===")

            X_train, X_temp = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_temp = y.iloc[train_index], y.iloc[valid_index]

            # Separando um conjunto para testar a calibração
            split_point = int(len(X_temp) * 0.7)
            X_calib, X_valid = X_temp.iloc[:split_point], X_temp.iloc[split_point:]
            y_calib, y_valid = y_temp.iloc[:split_point], y_temp.iloc[split_point:]

            model = LGBMClassifier(**params)
            model.fit(X_train, y_train)

            # Calibrando o modelo
            model_calib = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
            model_calib.fit(X_calib, y_calib)

            preds = model_calib.predict_proba(X_valid)[:, 1]

            logloss = log_loss(y_valid, preds)
            auc = roc_auc_score(y_valid, preds)
            brier = brier_score_loss(y_valid, preds)

            auc_scores.append(auc)
            brier_scores.append(brier)
            logloss_scores.append(logloss)

            # logger.success(f"LogLoss: {logloss:.4f} | AUC: {auc:.4f} | Brier: {brier:.4f}")

            # logger.info(f" Train from {self.data.loc[train_index, 'registered_on'].min().strftime('%Y-%m-%d')} to {self.data.loc[train_index, 'registered_on'].max().strftime('%Y-%m-%d')}")
            # logger.info(f" Valid from {self.data.iloc[valid_index].reset_index(drop=True).loc[:split_point, 'registered_on'].min().strftime('%Y-%m-%d')} to {self.data.iloc[valid_index].reset_index(drop=True).loc[:split_point, 'registered_on'].max().strftime('%Y-%m-%d')}")
            # logger.info(f" Test from {self.data.iloc[valid_index].reset_index(drop=True).loc[split_point:, 'registered_on'].min().strftime('%Y-%m-%d')} to {self.data.iloc[valid_index].reset_index(drop=True).loc[split_point:, 'registered_on'].max().strftime('%Y-%m-%d')}")

        mean_auc = np.mean(auc_scores)
        mean_brier = np.mean(brier_scores)
        mean_logloss = np.mean(logloss_scores)

        # get_baseline_logloss(y=y_valid)
        logger.success(
            f" TRIAL {self.i}: Logloss={mean_logloss:.5f} (σ={np.var(logloss_scores):.5f}) | AUC={mean_auc:.5f} (σ={np.var(auc_scores):.5f}) | Brier={mean_brier:.5f} (σ={np.var(brier_scores):.5f})"
        )
        self.i += 1
        return mean_logloss

    def objective(self, trial):
        self.fixed_params = {
            "verbosity": -1,
            "random_state": 42,
            "metric": "log_loss",
            "objective": "binary",
            "boosting_type": "gbdt",
        }
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "num_leaves": trial.suggest_int("num_leaves", 2, 16),  # 2^max_depth
            "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=10),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1, step=0.01),
            "pos_bagging_fraction": trial.suggest_float("pos_bagging_fraction", 0.5, 1, log=True),
            "neg_bagging_fraction": trial.suggest_float("neg_bagging_fraction", 0.5, 1, log=True),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        }
        # Supress Optuna warnings
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        return self.train(params={**self.fixed_params, **params})
