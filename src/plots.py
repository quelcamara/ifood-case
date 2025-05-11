import numpy as np
import pandas as pd
import seaborn as sns

from typing import Any
import matplotlib.pyplot as plt


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
    plt.style.use("ggplot")

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
        annot_kws={"size": 5, "color": "black"},
        annot=True,
        fmt=".2f",
        square=False,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    # Configurações dos rótulos
    plt.xticks(np.arange(len(corr.columns)), labels=corr.columns, fontsize=16, rotation=30, ha="right")
    plt.yticks(np.arange(len(corr.columns)), labels=corr.columns, fontsize=16)

    plt.title(f"Correlação ({method})", size=18, fontweight="bold")
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