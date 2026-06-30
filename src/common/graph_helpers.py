from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MultiLabelBinarizer

def make_binary_label_matrix(raw_data: List[Dict[str, Any]], taxonomy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw annotation records into a binary multi-label matrix.
    """
    cat_labels = taxonomy_df["cat_label"].unique().tolist()
    
    # Extract all labels per record as a list of lists
    labels_per_record = [
        [need["label"] for need in record.get("needs", []) if "label" in need]
        for record in raw_data
    ]

    ids = [record.get('id', '') for record in raw_data]
    
    # Encode using MultiLabelBinarizer (1 if class exists, 0 if not)
    mlb = MultiLabelBinarizer(classes=cat_labels)
    matrix = mlb.fit_transform(labels_per_record)
    
    return pd.DataFrame(matrix, columns=mlb.classes_, index=ids) # type: ignore


def compute_irl_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Imbalance Ratio (IR) metrics from a binary multi-label matrix.
    """
    freq = df.sum(axis=0).sort_values(ascending=False)
    max_freq = freq.max()
    
    irlbl = max_freq / freq
    irlbl_df = pd.DataFrame({
        "label": freq.index,
        "frequency": freq.values,
        "IRLbl": irlbl.values,
    }).reset_index(drop=True)
    
    summary_df = pd.DataFrame({
        "Metric": ["MeanIR", "MaxIR", "CVIR"],
        "Value": [irlbl.mean(), irlbl.max(), irlbl.std() / irlbl.mean()],
    })
    summary_df = summary_df.style.format({"Value": "{:.2f}"})
    
    return irlbl_df, summary_df # type: ignore


def plot_label_distribution(df: pd.DataFrame, 
                            log_scale: bool = True, 
                            figsize: Tuple[int, int] = (13, 13)) -> plt.Axes: # type: ignore
    """
    Plot category frequency distribution (optionally with formatted labels).
    """
    freq = df.sum(axis=0).sort_values(ascending=False)
    
    # Prepare plotting DataFrame
    plot_df = freq.reset_index(name="frequency").rename(columns={"index": "cat_label"})
    plot_df["display_label"] = plot_df["cat_label"]
    
    # Plot
    plt.figure(figsize=figsize, dpi=80)
    ax = sns.barplot(data=plot_df, y="display_label", x="frequency", hue="display_label", palette="viridis", legend=False)
    
    if log_scale:
        plt.xscale("log")
    
    # Add frequency labels on all bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3) # type: ignore
    
    title = f"Count of Notes by Category (n={len(df)})"
    if log_scale:
        title += " (log scale)"
    ax.set_title(title)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Category")
    plt.tight_layout()
    
    return ax

def analyze_category_distribution_from_dict(raw_data: List[Dict[str, Any]], 
                                            taxonomy_df: pd.DataFrame, 
                                            log_scale: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, plt.Axes]: # type: ignore
    """
    Complete analysis pipeline: build matrix from raw dicts, compute metrics, and plot.
    """
    df_labels = make_binary_label_matrix(raw_data, taxonomy_df)
    return analyze_category_distribution_from_df(df_labels, taxonomy_df, log_scale)


def analyze_category_distribution_from_df(df: pd.DataFrame, 
                                          taxonomy_df: Optional[pd.DataFrame] = None, 
                                          log_scale: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, plt.Axes]: # type: ignore
    """
    Complete analysis pipeline: take existing label matrix, compute metrics, and plot.
    """
    irlbl_df, summary_df = compute_irl_metrics(df)
    ax = plot_label_distribution(df, log_scale)
    return irlbl_df, summary_df, ax


def plot_cooccurrence_heatmap(data, taxonomy_df, figsize: Tuple[int, int] = (13, 13)) -> plt.Axes: # type: ignore
    """
    Plot a normalized co-occurrence heatmap for multi-label categories.
    """
    # 1. If it's a list of dicts, convert it to a DataFrame
    if isinstance(data, list):
        df_cats = make_binary_label_matrix(data, taxonomy_df)
    else:
        df_cats = data[taxonomy_df['cat_label']].astype(int)


    # 2. Compute co-occurrence matrix (multiply by transpose)
    co_matrix = df_cats.T.dot(df_cats)

    # 3. Normalize row-wise to get P(column | row)
    # Extract diagonal (class frequencies) as a Series with matching index
    class_frequencies = pd.Series(np.diag(co_matrix), index=co_matrix.index)
    # Divide each row by its diagonal value to get proportions
    normalized_matrix = co_matrix.div(class_frequencies, axis=0)

    # 4. Force diagonal to 0 so it doesn't blow out the scale
    normalized_matrix_arr = normalized_matrix.to_numpy(copy=True)
    np.fill_diagonal(normalized_matrix_arr, 0)
    normalized_matrix = pd.DataFrame(
        normalized_matrix_arr,
        index=co_matrix.index,
        columns=co_matrix.columns,
    )

    # 5. Plot the heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        normalized_matrix,
        cmap='Blues',
        annot=False,
        linewidths=0.5,
        linecolor="lightgrey",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Proportion of Co-occurrence P(Column | Row)"},
    )

    ax.set_title("Label Co-occurrence", fontsize=18, pad=20)
    ax.set_xlabel("Categories", fontsize=14)
    ax.set_ylabel("Categories", fontsize=14)

    plt.tight_layout()
    plt.show()
    return ax