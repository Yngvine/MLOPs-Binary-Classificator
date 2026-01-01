"""
Data handling utilities for rice grain classification.

Provides visualization and data manipulation functions for binary classification
with support for creating imbalanced datasets.
"""

from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

# Project paths
DATA_DIR = Path(__file__).parent.parent / "data"
UNPACKED_DIR = DATA_DIR / "unpacked"
UNBALANCED_DIR = DATA_DIR / "umbalanced"

# Feature columns (excluding id and Class)
FEATURE_COLUMNS = [
    "Area",
    "MajorAxisLength",
    "MinorAxisLength",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Perimeter",
    "Roundness",
    "AspectRation",
]

CLASS_LABELS = {0: "Gonen", 1: "Jasmine"}


def load_data(filepath: Path | str | None = None) -> pd.DataFrame:
    """
    Load the rice classification dataset.

    Parameters
    ----------
    filepath : Path | str | None, optional
        Path to CSV file. Defaults to unpacked/riceClassification.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with rice grain measurements.
    """
    if filepath is None:
        filepath = UNPACKED_DIR / "riceClassification.csv"
    return pd.read_csv(filepath)


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_violin_per_feature(
    df: pd.DataFrame,
    features: list[str] | None = None,
    figsize: tuple[int, int] = (16, 12),
    palette: str = "inferno",
    save_path: Path | str | None = None,
) -> Figure:
    """
    Create combined violin and box plots for each feature, split by class.

    Displays the distribution density (violin) with overlaid quartiles and outliers (boxplot).
    Each feature gets its own subplot with independent y-axis scale.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and 'Class' column.
    features : list[str] | None, optional
        List of feature columns to plot. Defaults to all features.
    figsize : tuple[int, int], optional
        Figure size as (width, height). Defaults to (16, 12).
    palette : str, optional
        Seaborn color palette name. Defaults to "inferno".
    save_path : Path | str | None, optional
        Optional path to save the figure.

    Returns
    -------
    Figure
        Matplotlib Figure object.
    """
    if features is None:
        features = FEATURE_COLUMNS

    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]
        # Violin plot (distribution)
        sns.violinplot(
            data=df,
            x="Class",
            y=feature,
            hue="Class",
            palette=palette,
            ax=ax,
            legend=False,
            inner=None,  # Remove default inner to make room for boxplot
            alpha=0.7,
        )
        
        # Box plot (outliers and quartiles)
        sns.boxplot(
            data=df,
            x="Class",
            y=feature,
            width=0.15,
            boxprops={"facecolor": "white", "edgecolor": "black", "alpha": 0.9},
            medianprops={"color": "black"},
            whiskerprops={"color": "black"},
            capprops={"color": "black"},
            flierprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 3},
            ax=ax,
            zorder=2,
        )
        
        ax.set_xlabel("")
        ax.set_ylabel(feature)
        ax.set_xticklabels([CLASS_LABELS.get(cast(int, i), str(i)) for i in sorted(df["Class"].unique())])
        ax.set_title(f"{feature} Distribution by Class")

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: list[str] | None = None,
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "inferno",
    save_path: Path | str | None = None,
) -> Figure:
    """
    Create a clustered correlation heatmap for feature relationships.

    Uses hierarchical clustering to group correlated features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features.
    features : list[str] | None, optional
        List of feature columns. Defaults to all features.
    figsize : tuple[int, int], optional
        Figure size. Defaults to (10, 8).
    cmap : str, optional
        Colormap for heatmap. Defaults to "inferno".
    save_path : Path | str | None, optional
        Optional path to save the figure.

    Returns
    -------
    Figure
        Matplotlib Figure object.
    """
    if features is None:
        features = FEATURE_COLUMNS

    corr_matrix = df[features].corr()

    # Use clustermap to group correlated features
    g = sns.clustermap(
        corr_matrix,
        cmap=cmap,
        center=0,
        figsize=figsize,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        dendrogram_ratio=0.15,
    )

    g.fig.suptitle("Clustered Feature Correlation Matrix", y=1.02, fontsize=14)

    if save_path:
        g.savefig(save_path, dpi=150, bbox_inches="tight")

    return g.fig


def plot_class_distribution(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (8, 5),
    palette: str = "inferno",
    save_path: Path | str | None = None,
) -> Figure:
    """
    Plot class distribution as a bar chart with counts and percentages.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Class' column.
    figsize : tuple[int, int], optional
        Figure size. Defaults to (8, 5).
    palette : str, optional
        Seaborn color palette. Defaults to "inferno".
    save_path : Path | str | None, optional
        Optional path to save the figure.

    Returns
    -------
    Figure
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    class_counts = df["Class"].value_counts().sort_index()
    total = len(df)

    bars = sns.barplot(
        x=[CLASS_LABELS.get(cast(int, i), str(i)) for i in class_counts.index],
        y=class_counts.values,
        palette=palette,
        ax=ax,
    )

    # Add count and percentage labels
    for idx, (count, bar) in enumerate(zip(class_counts.values, bars.patches)):
        rect = cast(Rectangle, bar)
        percentage = count / total * 100
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + total * 0.01,
            f"{count:,}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    ax.set_ylim(0, max(class_counts.values) * 1.15)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_pairplot_sample(
    df: pd.DataFrame,
    features: list[str] | None = None,
    sample_size: int = 1000,
    palette: str = "inferno",
    save_path: Path | str | None = None,
) -> sns.PairGrid:
    """
    Create a pairplot for feature relationships colored by class.

    Uses a sample to avoid slow rendering with large datasets.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and 'Class' column.
    features : list[str] | None, optional
        List of features to include. Defaults to first 5 features.
    sample_size : int, optional
        Number of samples per class. Defaults to 1000.
    palette : str, optional
        Seaborn color palette. Defaults to "inferno".
    save_path : Path | str | None, optional
        Optional path to save the figure.

    Returns
    -------
    sns.PairGrid
        Seaborn PairGrid object.
    """
    if features is None:
        features = FEATURE_COLUMNS[:5]  # Limit for readability

    # Stratified sample
    sampled = df.groupby("Class", group_keys=False).apply(
        lambda x: x.sample(n=min(sample_size, len(x)), random_state=42),
    )

    plot_df = sampled[features + ["Class"]].copy()
    plot_df["Class"] = plot_df["Class"].map(CLASS_LABELS)

    g = sns.pairplot(
        plot_df,
        hue="Class",
        palette=palette,
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "s": 20},
    )

    if save_path:
        g.savefig(save_path, dpi=150, bbox_inches="tight")

    return g


def plot_boxplot_comparison(
    df: pd.DataFrame,
    features: list[str] | None = None,
    figsize: tuple[int, int] = (14, 10),
    palette: str = "inferno",
    save_path: Path | str | None = None,
) -> Figure:
    """
    Create side-by-side boxplots for outlier visualization.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and 'Class' column.
    features : list[str] | None, optional
        List of features to plot. Defaults to all features.
    figsize : tuple[int, int], optional
        Figure size. Defaults to (14, 10).
    palette : str, optional
        Seaborn color palette. Defaults to "inferno".
    save_path : Path | str | None, optional
        Optional path to save the figure.

    Returns
    -------
    Figure
        Matplotlib Figure object.
    """
    if features is None:
        features = FEATURE_COLUMNS

    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]
        sns.boxplot(
            data=df,
            x="Class",
            y=feature,
            hue="Class",
            palette=palette,
            ax=ax,
            legend=False,
        )
        ax.set_xlabel("")
        ax.set_xticklabels([CLASS_LABELS.get(cast(int, i), str(i)) for i in sorted(df["Class"].unique())])
        ax.set_title(feature)

    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Boxplot Comparison by Class", y=1.02, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# Data Manipulation Functions
# =============================================================================


def create_imbalanced_dataset(
    df: pd.DataFrame | None = None,
    source_path: Path | str | None = None,
    target_class: int = 1,
    keep_percentage: float = 10.0,
    output_filename: str = "riceClassification_imbalanced.csv",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create an imbalanced dataset by downsampling one class.

    Stratified sampling ensures the subset is representative of the original
    class distribution in feature space.

    Parameters
    ----------
    df : pd.DataFrame | None, optional
        Source DataFrame. If None, loads from source_path.
    source_path : Path | str | None, optional
        Path to source CSV. Defaults to unpacked/riceClassification.csv.
    target_class : int, optional
        The class to downsample (0 or 1). Defaults to 1.
    keep_percentage : float, optional
        Percentage of target class to keep (0-100). Defaults to 10.0.
    output_filename : str, optional
        Name of output file in unbalanced folder. Defaults to "riceClassification_imbalanced.csv".
    random_state : int, optional
        Random seed for reproducibility. Defaults to 42.

    Returns
    -------
    pd.DataFrame
        Imbalanced DataFrame.

    Examples
    --------
    >>> # Keep only 10% of class 1, save to unbalanced folder
    >>> imbalanced_df = create_imbalanced_dataset(target_class=1, keep_percentage=10)
    """
    if df is None:
        if source_path is None:
            source_path = UNPACKED_DIR / "riceClassification.csv"
        df = pd.read_csv(source_path)

    if keep_percentage <= 0 or keep_percentage > 100:
        raise ValueError("keep_percentage must be between 0 and 100")

    # Split by class
    target_df = df[df["Class"] == target_class]
    other_df = df[df["Class"] != target_class]

    # Calculate sample size
    n_samples = int(len(target_df) * (keep_percentage / 100))

    # Stratified-like sampling (random sample for simple case)
    sampled_target = target_df.sample(n=n_samples, random_state=random_state)

    # Combine
    imbalanced_df = pd.concat([other_df, sampled_target], ignore_index=True)
    imbalanced_df = imbalanced_df.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )

    # Save to unbalanced folder
    UNBALANCED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = UNBALANCED_DIR / output_filename
    imbalanced_df.to_csv(output_path, index=False)

    # Print summary
    original_counts = df["Class"].value_counts().sort_index()
    new_counts = imbalanced_df["Class"].value_counts().sort_index()

    print(f"Created imbalanced dataset: {output_path}")
    print(f"\nOriginal distribution:")
    for cls, count in original_counts.items():
        print(f"  Class {cls} ({CLASS_LABELS.get(cast(int, cls), str(cls))}): {count:,}")
    print(f"\nNew distribution:")
    for cls, count in new_counts.items():
        pct = count / len(imbalanced_df) * 100
        print(f"  Class {cls} ({CLASS_LABELS.get(cast(int, cls), str(cls))}): {count:,} ({pct:.1f}%)")
    print(f"\nImbalance ratio: {max(new_counts) / min(new_counts):.2f}:1")

    return imbalanced_df


def get_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics per class for all features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and 'Class' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with statistics grouped by class.
    """
    stats = df.groupby("Class")[FEATURE_COLUMNS].agg(
        ["mean", "std", "min", "max", "median"]
    )
    return stats


def normalize_features(
    df: pd.DataFrame, method: Literal["minmax", "zscore"] = "zscore"
) -> pd.DataFrame:
    """
    Normalize feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features.
    method : Literal["minmax", "zscore"], optional
        'minmax' for [0,1] scaling or 'zscore' for standardization. Defaults to "zscore".

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized features (id and Class unchanged).
    """
    result = df.copy()

    for col in FEATURE_COLUMNS:
        if method == "minmax":
            min_val = result[col].min()
            max_val = result[col].max()
            result[col] = (result[col] - min_val) / (max_val - min_val)
        elif method == "zscore":
            mean_val = result[col].mean()
            std_val = result[col].std()
            result[col] = (result[col] - mean_val) / std_val

    return result


if __name__ == "__main__":
    # Example usage
    df = load_data()
    print(f"Loaded {len(df)} samples")
    print(f"\nClass distribution:")
    print(df["Class"].value_counts())

    # Create visualizations
    # plot_violin_per_feature(df, save_path="violins.png")
    # plot_correlation_heatmap(df, save_path="correlation.png")
    # plot_class_distribution(df, save_path="class_dist.png")

    # Create imbalanced dataset (10% of class 1)
    # imbalanced = create_imbalanced_dataset(target_class=1, keep_percentage=10)
