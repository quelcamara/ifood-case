import shap
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Any
from loguru import logger
from sklearn.calibration import calibration_curve

from src.utils import precision_recall_at_threshold


def plot_distribution(cohort_series: pd.Series):
    plt.figure(figsize=(25, 5))
    plt.style.use("seaborn-v0_8-darkgrid")

    counts = cohort_series.value_counts().sort_index()

    x = np.arange(len(counts))
    bar_width = 0.999
    plt.bar(
        x, counts.values, width=bar_width,
        color="darkred", edgecolor="white"
    )

    plt.xticks(
        x, labels=counts.index,
        rotation=45, fontsize=13,
        fontweight="bold", ha="center"
    )
    plt.yticks(fontsize=13)
    plt.margins(x=0.005)
    plt.tight_layout()
    plt.show()


def plot_corr(df, method):
    plt.figure(figsize=(25, 10))
    plt.style.use("seaborn-v0_8-darkgrid")

    corr = df.corr(method=method)

    # generates a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # custom colormap palette
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # draws the heatmap with the mask
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        annot_kws={"size": 9, "color": "black"},
        annot=True,
        fmt=".2f",
        square=False,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    # Configurações dos rótulos
    plt.xticks(np.arange(len(corr.columns)), labels=corr.columns, fontsize=11, rotation=30, ha="right")
    plt.yticks(np.arange(len(corr.columns)), labels=corr.columns, fontsize=11)

    plt.title(f"Correlação ({method})", size=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(X: pd.DataFrame, importance_values: Any):
    """
    """
    plt.figure(figsize=(25, 8))
    plt.style.use("seaborn-v0_8-darkgrid")

    # Plotando o gráfico com os valores ordenados
    sorted_idx = importance_values.importances_mean.argsort()
    plt.barh(
        X.columns[sorted_idx], 
        importance_values.importances_mean[sorted_idx],
        color="darkred"
    )
    plt.title("Feature importance", size=18, fontweight="bold")
    plt.xlabel("Redução na métrica", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_roc_auc_curve(fpr, tpr, best_idx):
    plt.figure(figsize=(25, 6))
    plt.style.use("ggplot")

    plt.plot(fpr, tpr, lw=3, color="#212F3D", label="ROC Curve")
    plt.plot([0, 1], [0, 1], color="#85929E", lw=1.5, linestyle="--")
    plt.scatter(fpr[best_idx], tpr[best_idx], color="#922B21", label="Ponto Ótimo")

    plt.axvline(x=fpr[best_idx], color="#922B21", linestyle="--", lw=2)
    plt.axhline(y=tpr[best_idx], color="#922B21", linestyle="--", lw=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel("FPR", size=16, fontweight="bold")
    plt.ylabel("TPR", size=16, fontweight="bold")
    plt.title("Curva ROC AUC", size=18, fontweight="bold")

    plt.legend(prop={"size": 14})
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_summary_shap(shap_values, X, cmap):
    shap.summary_plot(
        shap_values,
        features=X,
        plot_size=(25, 6),
        cmap=cmap,
        max_display=X.shape[1]
    )


def plot_scatter_shap(explanaition, shap_values, X, cmap, scatter_n_top_feats=3):
    importance = np.abs(shap_values).mean(axis=0)
    top_index = np.argsort(importance)[-scatter_n_top_feats:][::-1]

    rows = int(np.ceil(scatter_n_top_feats / 3))
    height = 5 * rows

    _, axes = plt.subplots(nrows=rows, ncols=3, figsize=(25, height))
    axes = axes.flatten()

    for i, idx in enumerate(top_index):
        shap.plots.scatter(explanaition[:, idx], color=shap_values[:, idx], dot_size=8, alpha=0.7, ax=axes[i], cmap=cmap, show=False)


def plot_scatter_permutations_shap(explanaition, shap_values, cmap, scatter_n_top_feats=3):
    importance = np.abs(shap_values).mean(axis=0)
    top_index = np.argsort(importance)[-scatter_n_top_feats:][::-1]

    feature_combinations = list(itertools.permutations(top_index, 2))

    rows = int(np.ceil(len(feature_combinations) / 3))
    height = 5 * rows

    _, axes = plt.subplots(nrows=rows, ncols=3, figsize=(25, height))
    axes = axes.flatten()

    for i, (feat1, feat2) in enumerate(feature_combinations):
        shap_values_feat1 = explanaition[:, feat1]
        shap_values_feat2 = explanaition[:, feat2]

        shap.plots.scatter(shap_values_feat1, shap_values_feat2, dot_size=8, alpha=0.7, ax=axes[i], cmap=cmap, show=False)


def visualize_feature_importance(model, X, scatter_n_top_feats: int = 3):
    explainer = shap.Explainer(model.estimator)
    shap_values = explainer.shap_values(X)
    explanaition = explainer(X)

    plt.style.use("ggplot")
    cmap = plt.get_cmap("RdYlBu_r")

    plot_summary_shap(shap_values=shap_values, X=X, cmap=cmap)
    plot_scatter_shap(explanaition, shap_values, X, cmap, scatter_n_top_feats)
    plot_scatter_permutations_shap(explanaition, shap_values, cmap, scatter_n_top_feats)

    plt.show()


def plot_calibration_curve(y, preds, threshold):
    fig, ax = plt.subplots(figsize=(25, 6))
    plt.style.use("ggplot")

    # Curva de calibração
    fraction_of_positives, mean_predicted_value = calibration_curve(y, preds, n_bins=15, strategy="uniform")
    ax.plot(mean_predicted_value, fraction_of_positives, lw=3, color="#212F3D", marker="o", ms=10, label="XGBoost")
    ax.plot([0, 1], [0, 1], color="#85929E", linestyle="--", lw=1.5, label="perfectly calibrated")
    ax.vlines(threshold, 0, 1, color="#922B21", linestyle="--", lw=2, label="threshold")

    # Marcação de subestimativa e superestimativa
    ax.text(threshold + 0.01, 0.95, "Underestimating (conservative)", color="#1F618D", fontsize=14, weight="bold")
    ax.text(threshold + 0.01, 0, "Overestimating (overconfident)", color="#A93226", fontsize=14, weight="bold")

    ax.annotate("", xy=(threshold - 0.01, 1), xytext=(threshold - 0.01, 0.9), arrowprops=dict(arrowstyle="->", color="#1F618D", lw=2))
    ax.annotate("", xy=(threshold - 0.01, 0), xytext=(threshold - 0.01, 0.10), arrowprops=dict(arrowstyle="->", color="#A93226", lw=2))

    # Adicionar o histograma no fundo com eixo secundário
    ax_hist = ax.twinx()
    ax_hist.hist(preds, bins=15, range=(0, 1), color="#1F618D", alpha=0.15, edgecolor="black")
    ax_hist.set_ylabel("Frequency", fontsize=14)
    ax_hist.set_yticks([])

    # Títulos e rótulos
    ax.set_title("Calibration Plot", size=18, weight="bold")
    ax.set_xlabel("Mean predicted value", size=16, fontweight="bold")
    ax.set_ylabel("Fraction of positives", size=16, fontweight="bold")

    ax.legend(prop={"size": 14})
    ax.yticks = plt.yticks(fontsize=14)
    ax.xticks = plt.xticks(fontsize=14)

    plt.tight_layout()
    plt.show()


def lift_curve_with_threshold(y_true, y_pred_proba, n_bins=10, target_fraction=0.2):
    fig, ax = plt.subplots(figsize=(25, 6))
    plt.style.use("ggplot")

    df = pd.DataFrame({"y_true": y_true, "y_score": y_pred_proba})
    df = df.sort_values("y_score", ascending=False).reset_index(drop=True)

    top_n = int(len(df) * target_fraction)
    threshold = df.loc[top_n - 1, "y_score"]

    df["bucket"] = pd.qcut(df.index, n_bins, labels=False)
    
    lift_table = df.groupby("bucket").agg({"y_true": ["count", "sum"]}).reset_index()
    lift_table.columns = ["bucket", "total", "positives"]

    lift_table["cumulative_positives"] = lift_table["positives"].cumsum()
    lift_table["cumulative_total"] = lift_table["total"].cumsum()
    lift_table["cumulative_gain"] = lift_table["cumulative_positives"] / df["y_true"].sum()
    lift_table["baseline_gain"] = lift_table["cumulative_total"] / len(df)

    lift_table = pd.concat([
        pd.DataFrame({
            "cumulative_gain": [0],
            "baseline_gain": [0]
        }),
        lift_table[["cumulative_gain", "baseline_gain"]]
    ], ignore_index=True)

    plt.plot(lift_table["baseline_gain"], lift_table["baseline_gain"], label="Aleatório", linestyle="--", color="darkgrey")
    plt.plot(lift_table["baseline_gain"], lift_table["cumulative_gain"], label="Modelo", linewidth=2, color="darkred")

    plt.axvline(x=target_fraction, color="black", linestyle=":", label=f"Top {int(target_fraction*100)}%")
    plt.title("Lift Curve (Ganho Acumulado)", size=16, weight="bold")
    plt.xlabel("% da base abordada", size=14, fontweight="bold")
    plt.ylabel("% de conversões capturadas", size=14, fontweight="bold")

    plt.legend(prop={"size": 12})
    plt.show()

    # Lift @ top 20%
    positives_top = df.loc[:top_n, "y_true"].sum()
    conversion_top = positives_top / top_n
    conversion_overall = df["y_true"].mean()
    lift_topn = conversion_top / conversion_overall

    logger.info(f"Threshold para top {int(target_fraction*100)}% oportunidades: {threshold:.4f}")
    logger.info(f"Taxa de conversão no top {target_fraction:.0%}: {conversion_top:.4f}")
    logger.info(f"Taxa de conversão geral: {conversion_overall:.4f}")
    logger.info(f"Lift no top {target_fraction:.0%}: {lift_topn:.2f}x")

    precision_recall_at_threshold(y_true, y_pred_proba, threshold)
    return threshold


def plot_real_conversion_per_decil_score(data):
    plt.figure(figsize=(25, 6))
    plt.style.use("ggplot")

    df = data.copy()

    df = df.sort_values("pred", ascending=False).reset_index(drop=True)
    df["decil"] = pd.qcut(df.index, q=10, labels=[f"{i+1}" for i in range(10)])

    summary = df.groupby("decil").agg(
        total=("target", "count"),
        conversoes=("target", "sum")
    ).reset_index()
    summary["taxa_conversao"] = summary["conversoes"] / summary["total"]

    plt.bar(summary["decil"], summary["taxa_conversao"], color="darkred")

    plt.title("Taxa de Conversão Real por Decil do Score", fontsize=18, weight="bold")
    plt.xlabel("Decil", size=16, fontweight="bold")
    plt.ylabel("Taxa de Conversão Real", size=16, fontweight="bold")
    
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    
    plt.tight_layout()
    plt.show()