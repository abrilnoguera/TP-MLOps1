import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from scipy.stats import ks_2samp

def performance_report(y_true, y_score, threshold: float = 0.5, n_bins: int = 10):
    """
    Generate performance report (deciles + summary) given true labels and predicted probabilities.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 = good, 1 = bad).
    y_score : array-like
        Predicted probabilities (float between 0 and 1).
    threshold : float, default=0.5
        Classification threshold for metrics.
    n_bins : int, default=10
        Number of bins (deciles by default).

    Returns
    -------
    out_deciles : pd.DataFrame
        Performance metrics calculated per decile.
    summary : pd.DataFrame
        Global classification metrics at the given threshold.
    """
    y_true = pd.Series(y_true).astype(int)
    y_score = pd.Series(y_score).astype(float)

    # Drop missing
    mask = ~(y_true.isna() | y_score.isna())
    y_true, y_score = y_true[mask], y_score[mask]

    # ================== Decile Table ==================
    edges = np.unique(np.quantile(y_score, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = 0.0, 1.0
    if len(edges) < 3:  # safeguard
        edges = np.linspace(0, 1, n_bins + 1)

    labels = list(range(1, len(edges)))
    decile = pd.cut(y_score, bins=edges, labels=labels, include_lowest=True, right=False).astype("Int64")

    df = pd.DataFrame({"y_true": y_true, "y_score": y_score, "decile": decile})

    agg = (
        df.groupby("decile", dropna=False)
        .agg(
            minprob=("y_score", "min"),
            maxprob=("y_score", "max"),
            avgprob=("y_score", "mean"),
            n=("y_true", "count"),
            n_bad=("y_true", "sum"),
        )
        .reset_index()
    )
    agg["n_good"] = agg["n"] - agg["n_bad"]
    agg["bad_rate"] = agg["n_bad"] / agg["n"]

    # Totals
    total_n = len(y_true)
    total_bad = int(y_true.sum())
    total_good = total_n - total_bad
    total_badrate = total_bad / total_n

    # Order by decile high → low
    agg = agg.sort_values("decile", ascending=False).reset_index(drop=True)

    # Cumulative
    agg["cum_n_top"] = agg["n"].cumsum()
    agg["cum_bad_top"] = agg["n_bad"].cumsum()
    agg["cum_good_top"] = agg["n_good"].cumsum()

    agg["share_cum_bad"] = agg["cum_bad_top"] / total_bad
    agg["share_cum_good"] = agg["cum_good_top"] / total_good
    agg["ks_diff"] = (agg["share_cum_bad"] - agg["share_cum_good"]).abs()
    ks_deciles = agg["ks_diff"].max()

    agg["precision"] = agg["cum_bad_top"] / agg["cum_n_top"]
    agg["recall"] = agg["cum_bad_top"] / total_bad
    agg["lift"] = agg["precision"] / total_badrate

    # Prob-based metrics
    auc_prob = roc_auc_score(y_true, y_score) if y_true.nunique() == 2 else np.nan
    gini_prob = (auc_prob - 0.5) * 2 if pd.notna(auc_prob) else np.nan
    avgpr_prob = average_precision_score(y_true, y_score) if y_true.nunique() == 2 else np.nan

    out_deciles = agg.round(
        {
            "minprob": 3, "maxprob": 3, "avgprob": 3,
            "bad_rate": 3, "precision": 3, "recall": 3, "lift": 3,
            "share_cum_bad": 3, "share_cum_good": 3, "ks_diff": 3
        }
    )

    # ================== Global Summary ==================
    y_pred = (y_score >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if y_true.nunique() == 2:
        auc = roc_auc_score(y_true, y_score)
        ks = ks_2samp(y_score[y_true == 0], y_score[y_true == 1]).statistic
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    else:
        auc, ks = np.nan, np.nan
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

    summary = pd.DataFrame([{
        "total_n": total_n,
        "total_bad": total_bad,
        "bad_rate": total_bad / total_n * 100,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "auc": auc, "ks": ks,
        "auc_prob": auc_prob, "gini_prob": gini_prob, "avgpr_prob": avgpr_prob,
        "ks_deciles": ks_deciles,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }]).round(3)

    return out_deciles, summary