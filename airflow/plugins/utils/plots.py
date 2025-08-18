import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def plot_correlation_with_target(X, y, target_col="num", save_path=None):
    """
    Plots the correlation of each variable in the dataframe with the target column.

    Args:
    - X (pd.DataFrame): DataFrame containing the features (not including the target column).
    - y (pd.DataFrame | pd.Series): DataFrame or Series containing the target column. Must have the same number of observations as X.
    - target_col (str, optional): Name of the target column. Defaults to "num".
    - save_path (str, optional): Path to save the generated plot. If not specified, the plot won't be saved.

    Returns:
    - fig: The Matplotlib figure object.
    """

    # Check alignment between X and y
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y are not aligned")
    
    df = pd.concat([X, y], axis=1)

    # Compute correlation with respect to the target column
    correlations = df.corr()[target_col].drop(target_col).sort_values()

    # Build color palette
    colors = sns.diverging_palette(10, 130, as_cmap=True)
    color_mapped = correlations.map(colors)

    # Configure seaborn style
    sns.set_style(
        "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
    )

    # Create horizontal bar plot
    fig = plt.figure(figsize=(12, 8))
    bars = plt.barh(correlations.index, correlations.values, color=color_mapped)

    # Titles and axis labels
    plt.title(f"Correlation with {target_col.title()}", fontsize=18)
    plt.xlabel("Correlation Coefficient", fontsize=16)
    plt.ylabel("Variable", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="x")

    plt.tight_layout()

    # Save plot if requested
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    # Prevent Matplotlib from displaying automatically
    plt.close(fig)

    return fig


def plot_information_gain_with_target(X, y, target_col="num", save_path=None):
    """
    Plots the information gain (mutual information) of each variable with respect to the target column.

    Args:
    - X (pd.DataFrame): DataFrame containing the features.
    - y (pd.DataFrame | pd.Series): Target column with the same number of rows as X.
    - target_col (str, optional): Name of the target column. Defaults to "num".
    - save_path (str, optional): Path to save the generated plot. If not specified, the plot won't be saved.

    Returns:
    - fig: The Matplotlib figure object.
    """

    # Check alignment between X and y
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y are not aligned")

    # Compute mutual information scores with the target
    importances = pd.Series(mutual_info_classif(X, y.to_numpy().ravel()), X.columns).sort_values()

    # Build color palette
    colors = sns.diverging_palette(10, 130, as_cmap=True)
    color_mapped = importances.map(colors)

    # Configure seaborn style
    sns.set_style(
        "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
    )

    # Create horizontal bar plot
    fig = plt.figure(figsize=(12, 8))
    bars = plt.barh(importances.index, importances, color=color_mapped)

    # Titles and axis labels
    plt.title(f"Information Gain with {target_col.title()}", fontsize=18)
    plt.xlabel("Information Gain", fontsize=16)
    plt.ylabel("Variable", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="x")

    plt.tight_layout()

    # Save plot if requested
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    # Prevent Matplotlib from displaying automatically
    plt.close(fig)

    return fig


def plot_class_balance(y, target_col="num", save_path=None, normalize=True):
    """
    Plots the class balance of the target column.

    Args:
    - y (pd.DataFrame | pd.Series): Target variable. If DataFrame, must contain `target_col`.
    - target_col (str): Target column name if `y` is a DataFrame.
    - save_path (str, optional): Path to save the generated plot (PNG). If not specified, the plot won't be saved.
    - normalize (bool): If True, displays percentages along with counts.

    Returns:
    - fig: The Matplotlib figure object.
    """
    # Get target as Series
    if isinstance(y, pd.DataFrame):
        if target_col not in y.columns:
            raise ValueError(f"Target column '{target_col}' not found in y DataFrame")
        y_series = y[target_col]
    else:
        y_series = pd.Series(y, name=target_col)

    # Compute counts and (optionally) proportions
    counts = y_series.value_counts(dropna=False).sort_index()
    if normalize:
        props = (counts / counts.sum()) * 100.0

    # Build color palette
    colors = sns.diverging_palette(10, 130, as_cmap=True)
    unique_classes = counts.index.tolist()
    color_list = [colors(i / max(len(unique_classes)-1, 1)) for i in range(len(unique_classes))]

    sns.set_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})

    # Create bar plot
    fig = plt.figure(figsize=(10, 6))
    bars = plt.bar(unique_classes, counts.values, color=color_list)

    # Titles and axis labels
    plt.title(f"Class Balance: {target_col}", fontsize=18)
    plt.xlabel("Class", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y")

    # Annotate bars with values
    for i, b in enumerate(bars):
        cnt = counts.values[i]
        if normalize:
            pct = props.values[i]
            label = f"{cnt} ({pct:.1f}%)"
        else:
            label = f"{cnt}"
        plt.text(b.get_x() + b.get_width()/2, b.get_height(),
                 label, ha='center', va='bottom', fontsize=12)

    plt.tight_layout()

    # Save plot if requested
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    plt.close(fig)
    return fig

def plot_feature_importance(
    model,
    feature_names,
    X=None,
    y=None,
    *,
    kind: str = "auto",      # "auto" | "model" | "coef" | "permutation"
    n_top: int = 30,
    normalize: bool = False,
    save_path: str | None = None,
    random_state: int = 42,
):
    """
    Draw a barplot with feature importance.

    Parameters
    ----------
    model : trained sklearn estimator
    feature_names : list[str] | pd.Index
        Feature names in the same order expected by the model.
    X, y : optional
        Required if kind="permutation" or "auto" with no model attributes available.
    kind : {"auto","model","coef","permutation"}
        - "auto": tries model -> coef -> permutation (if X,y are given)
        - "model": uses attribute `feature_importances_`
        - "coef": uses `abs(coef_)` (flattens if multiclass)
        - "permutation": uses permutation importance (requires X,y)
    n_top : int
        Number of top features to display.
    normalize : bool
        If True, normalizes importance values so that they sum to 1.
    save_path : str | None
        Path to save the PNG.
    random_state : int
        Random seed for permutation importance.

    Returns
    -------
    fig : matplotlib.figure.Figure
    importance_df : pd.DataFrame with columns ["feature","importance"]
    """
    # Ensure the model is fitted
    try:
        check_is_fitted(model)
    except Exception:
        raise ValueError("The model must be fitted before computing feature importance.")

    feature_names = list(feature_names)

    # --- resolve kind ---
    importance = None
    chosen = kind

    if kind == "auto":
        if hasattr(model, "feature_importances_"):
            chosen = "model"
        elif hasattr(model, "coef_"):
            chosen = "coef"
        else:
            chosen = "permutation"

    if chosen == "model":
        if not hasattr(model, "feature_importances_"):
            raise ValueError("The model has no attribute feature_importances_.")
        importance = np.asarray(model.feature_importances_, dtype=float)

    elif chosen == "coef":
        if not hasattr(model, "coef_"):
            raise ValueError("The model has no attribute coef_.")
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 2:  # multi-class or binary with shape (1, n_features)
            coef = np.abs(coef).mean(axis=0)
        else:
            coef = np.abs(coef)
        importance = coef

    elif chosen == "permutation":
        if X is None or y is None:
            raise ValueError("Permutation importance requires both X and y.")
        # Ensure DataFrame/ndarray consistency
        if isinstance(X, pd.DataFrame):
            X_used = X.values
            if feature_names is None:
                feature_names = list(X.columns)
        else:
            X_used = np.asarray(X)

        res = permutation_importance(
            model, X_used, y, n_repeats=10, random_state=random_state, n_jobs=-1
        )
        importance = res.importances_mean

    else:
        raise ValueError(f"kind '{kind}' not recognized.")

    # --- build importance dataframe ---
    if len(importance) != len(feature_names):
        raise ValueError("Importance length does not match feature_names length.")

    imp = pd.DataFrame({"feature": feature_names, "importance": importance})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)

    if normalize and imp["importance"].sum() > 0:
        imp["importance"] = imp["importance"] / imp["importance"].sum()

    imp_top = imp.head(n_top).iloc[::-1]  # invert for ascending barh

    # --- style and plot ---
    sns.set_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    colors = [cmap(i / max(len(imp_top) - 1, 1)) for i in range(len(imp_top))]

    fig = plt.figure(figsize=(10, max(6, int(0.4 * len(imp_top)))))
    plt.barh(imp_top["feature"], imp_top["importance"], color=colors)
    ttl = "Feature Importance"
    if chosen == "permutation":
        ttl += " (Permutation)"
    elif chosen == "coef":
        ttl += " (|coef|)"
    plt.title(ttl, fontsize=16)
    plt.xlabel("Importance" + (" (normalized)" if normalize else ""), fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png", dpi=300)

    plt.close(fig)
    return fig, imp

def plot_calibration_curve(
    y_true,
    y_score,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",   # "uniform" or "quantile"
    show_hist: bool = True,
    title: str = "Calibration Curve",
    save_path: str | None = None,
):
    """
    Draw the calibration curve (reliability diagram) and return calibration metrics.

    Parameters
    ----------
    y_true : array-like of {0,1}
        Ground-truth labels.
    y_score : array-like of [0,1]
        Predicted probabilities.
    n_bins : int
        Number of bins for the curve.
    strategy : {"uniform","quantile"}
        Binning scheme (uniform in prob or quantile-based).
    show_hist : bool
        If True, adds histogram showing probability support.
    title : str
        Plot title.
    save_path : str | None
        Path to save PNG.

    Returns
    -------
    fig : matplotlib.figure.Figure
    metrics : dict with {"brier", "ece", "mce"}
    data : pd.DataFrame with columns ["bin_pred", "bin_true", "bin_count"]
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    # empirical curve (sklearn computes bins and returns mean_pred, frac_pos)
    frac_pos, mean_pred = calibration_curve(
        y_true, y_score, n_bins=n_bins, strategy=strategy
    )

    # rebuild counts per bin for ECE/MCE
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:  # quantile
        edges = np.unique(np.quantile(y_score, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_idx = np.digitize(y_score, edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)

    df = pd.DataFrame({"y": y_true, "p": y_score, "bin": bin_idx})
    by = df.groupby("bin", as_index=False).agg(
        bin_true=("y", "mean"), bin_pred=("p", "mean"), bin_count=("y", "size")
    )
    # Align with sklearn curve points (some bins may be empty)
    curve_df = by[by["bin_count"] > 0].copy()
    # Sort by predicted prob mean
    curve_df = curve_df.sort_values("bin_pred")

    # Metrics
    brier = brier_score_loss(y_true, y_score)
    weights = curve_df["bin_count"] / curve_df["bin_count"].sum()
    ece = np.sum(weights * np.abs(curve_df["bin_true"] - curve_df["bin_pred"]))
    mce = np.max(np.abs(curve_df["bin_true"] - curve_df["bin_pred"])) if not curve_df.empty else np.nan

    # Plot
    fig = plt.figure(figsize=(7, 6))
    ax = plt.gca()

    # Ideal line
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Perfectly calibrated")

    # Empirical curve
    ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, label="Empirical")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted probability", fontsize=12)
    ax.set_ylabel("Observed frequency", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # Histogram (support)
    if show_hist:
        ax2 = ax.twinx()
        ax2.hist(y_score, bins=edges, alpha=0.2, edgecolor="k")
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_ylim(0, max(5, int(df.shape[0] * 0.35)))  # reasonable scaling

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    metrics = {"brier": brier, "ece": float(ece), "mce": float(mce)}
    data = curve_df[["bin_pred", "bin_true", "bin_count"]].rename(
        columns={"bin_pred": "pred_mean", "bin_true": "obs_rate"}
    )
    return fig, metrics, data