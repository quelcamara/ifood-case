import numpy as np
import pandas as pd
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

