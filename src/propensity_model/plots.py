"""Publication-quality plots for dual-horizon (H=1, H=3) propensity baselines."""
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9.5,
    }
)


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return str(path)


def generate_all_plots(
    run_dir: Path,
    bootstrap_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    slice_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    lc_df: pd.DataFrame,
    fi_df: pd.DataFrame,
    final_df: pd.DataFrame,
    winners: dict[str, str],
    exp_order: list[str],
) -> list[str]:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    # 1. Ablation ladder H=1 vs H=3 with 95% CI
    created.append(_plot_ablation_ladder(bootstrap_df, exp_order, fig_dir / "01_ablation_ladder_h1_vs_h3.png"))

    # 2. Paired ablation deltas forest plot
    if len(paired_df) > 0:
        created.append(_plot_paired_deltas(paired_df, fig_dir / "02_paired_ablation_deltas.png"))

    # 3. PR & ROC curves for key experiments on pooled validation
    created.append(_plot_pr_roc(run_dir, exp_order, winners, fig_dir / "03_pr_roc_curves.png"))

    # 4. Budget sensitivity (0.5%, 1%, 5%)
    created.append(_plot_budget_sensitivity(monthly_df, exp_order, fig_dir / "04_budget_lift_recall.png"))

    # 5. Calibration reliability curves & Top-1% confusion matrices
    created.append(_plot_calibration_and_cm(cal_df, monthly_df, final_df, winners, fig_dir / "05_calibration_and_confusion.png"))

    # 6. Slice-based performance & Top-1% composition
    created.append(_plot_slices(slice_df, winners, fig_dir / "06_slices_coverage_and_prior.png"))

    # 7. Final Holdout: H1 calibration on 2023-01 & H3 monthly vs quarterly on 2022-11
    if len(final_df) > 0:
        created.append(_plot_final_holdouts(final_df, fig_dir / "07_final_holdout_h1_h3.png"))

    # 8. Learning curves & Top-20 feature importances
    created.append(_plot_learning_and_importance(lc_df, fi_df, winners, fig_dir / "08_learning_curves_and_importance.png"))

    return created


def _plot_ablation_ladder(df: pd.DataFrame, exp_order: list[str], path: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    metrics = [
        ("lift_1pct", "Lift@1% (95% CI)"),
        ("recall_1pct", "Recall@1% (95% CI)"),
        ("average_precision", "Average Precision (95% CI)"),
        ("roc_auc", "ROC-AUC (95% CI)"),
    ]
    colors = {1: "#1f77b4", 3: "#d95f02"}
    x = np.arange(len(exp_order))
    width = 0.36

    for ax, (col, title) in zip(axes.ravel(), metrics):
        for offset, h in ((-width / 2, 1), (width / 2, 3)):
            sub = df[(df["horizon"] == h) & (df["scope"] == "validation_pooled")].set_index("experiment_id")
            vals, lo_err, hi_err = [], [], []
            for eid in exp_order:
                if eid in sub.index and pd.notna(sub.loc[eid, col]):
                    v = float(sub.loc[eid, col])
                    lo = float(sub.loc[eid, f"{col}_ci_low"])
                    hi = float(sub.loc[eid, f"{col}_ci_high"])
                    vals.append(v)
                    lo_err.append(max(0.0, v - lo))
                    hi_err.append(max(0.0, hi - v))
                else:
                    vals.append(np.nan)
                    lo_err.append(0.0)
                    hi_err.append(0.0)
            ax.bar(
                x + offset,
                vals,
                width=width,
                color=colors[h],
                alpha=0.88,
                label=f"H={h}",
                yerr=[lo_err, hi_err],
                capsize=2.5,
                error_kw={"lw": 1.0},
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(exp_order, rotation=32, ha="right", fontsize=8)
        if col == "lift_1pct":
            ax.axhline(1.0, ls="--", color="gray", lw=1)
        ax.legend(frameon=False, loc="best")

    fig.suptitle("Ablation Ladder on Rolling Validation: H=1 (09-11/2022) vs H=3 (08-10/2022)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def _plot_paired_deltas(paired_df: pd.DataFrame, path: Path) -> str:
    sub_val = paired_df[paired_df["scope"] == "validation_pooled"].copy()
    if len(sub_val) == 0:
        sub_val = paired_df.copy()
    sub_val["label"] = sub_val["model_b"] + " - " + sub_val["model_a"]
    pairs = list(dict.fromkeys(sub_val["label"].tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4.2, 0.55 * len(pairs) + 1.5)), sharey=True)
    y = np.arange(len(pairs))
    colors = {1: "#1f77b4", 3: "#d95f02"}

    for ax, (col, title) in zip(
        axes,
        [("delta_recall_1pct", "Paired Δ Recall@1% (95% CI)"), ("delta_average_precision", "Paired Δ Average Precision (95% CI)")],
    ):
        ax.axvline(0.0, color="black", lw=1, ls="--")
        for dy, h in ((-0.16, 1), (0.16, 3)):
            sub_h = sub_val[sub_val["horizon"] == h].set_index("label")
            for idx, lbl in enumerate(pairs):
                if lbl not in sub_h.index or pd.isna(sub_h.loc[lbl, col]):
                    continue
                v = float(sub_h.loc[lbl, col])
                lo = float(sub_h.loc[lbl, f"{col}_ci_low"])
                hi = float(sub_h.loc[lbl, f"{col}_ci_high"])
                ax.errorbar(
                    v,
                    idx + dy,
                    xerr=[[max(0.0, v - lo)], [max(0.0, hi - v)]],
                    fmt="o",
                    color=colors[h],
                    capsize=3,
                    label=f"H={h}" if idx == 0 else None,
                )
        ax.set_title(title, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(pairs, fontsize=8.5)
        ax.invert_yaxis()
        ax.legend(frameon=False, loc="best")

    fig.suptitle("Paired Client-Clustered Bootstrap Ablation Gains (B - A)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def _plot_pr_roc(run_dir: Path, exp_order: list[str], winners: dict[str, str], path: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    for row_idx, h in enumerate((1, 3)):
        pred_root = run_dir / "predictions" / f"h{h}_val"
        # Pick up to 5 representative models including B1, B2, first CB, winner
        chosen = []
        for eid in exp_order:
            if eid == "B0_prevalence":
                continue
            if eid in (winners.get(f"h{h}"), exp_order[1] if len(exp_order) > 1 else None, exp_order[2] if len(exp_order) > 2 else None):
                chosen.append(eid)
        for eid in exp_order:
            if eid not in chosen and eid != "B0_prevalence" and len(chosen) < 5:
                chosen.append(eid)

        ax_pr, ax_roc = axes[row_idx]
        prevalence_drawn = False
        for eid in chosen:
            exp_dir = pred_root / eid
            files = sorted(exp_dir.glob("fold_*.parquet"))
            if not files:
                continue
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            y = df["y_true"].to_numpy(dtype=np.int8)
            s = df["score"].to_numpy(dtype=np.float64)
            prec, rec, _ = precision_recall_curve(y, s)
            fpr, tpr, _ = roc_curve(y, s)
            lw = 2.2 if eid == winners.get(f"h{h}") else 1.3
            ax_pr.plot(rec, prec, lw=lw, label=eid)
            ax_roc.plot(fpr, tpr, lw=lw, label=eid)
            if not prevalence_drawn:
                prev = float(np.mean(y))
                ax_pr.axhline(prev, ls="--", color="gray", lw=1, label=f"Prevalence ({prev*100:.2f}%)")
                ax_roc.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
                prevalence_drawn = True

        ax_pr.set_title(f"H={h} Precision-Recall Curve (Validation Pooled)", fontweight="bold")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_xlim(0, 1)
        ax_pr.legend(frameon=False, fontsize=8)

        ax_roc.set_title(f"H={h} ROC Curve (Validation Pooled)", fontweight="bold")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    return _save(fig, path)


def _plot_budget_sensitivity(monthly_df: pd.DataFrame, exp_order: list[str], path: Path) -> str:
    val = monthly_df[monthly_df["stage"] == "validation"]
    agg = val.groupby(["horizon", "experiment_id"], as_index=False)[
        ["lift_0p5pct", "lift_1pct", "lift_5pct", "recall_0p5pct", "recall_1pct", "recall_5pct"]
    ].mean()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    budgets = ["0p5pct", "1pct", "5pct"]
    labels = ["Top 0.5%", "Top 1.0%", "Top 5.0%"]
    x = np.arange(len(budgets))

    for row_idx, h in enumerate((1, 3)):
        sub = agg[agg["horizon"] == h].set_index("experiment_id")
        ax_l, ax_r = axes[row_idx]
        for eid in exp_order:
            if eid not in sub.index:
                continue
            lifts = [sub.loc[eid, f"lift_{b}"] for b in budgets]
            recs = [sub.loc[eid, f"recall_{b}"] for b in budgets]
            ax_l.plot(x, lifts, marker="o", label=eid)
            ax_r.plot(x, recs, marker="o", label=eid)
        ax_l.set_title(f"H={h} Mean Lift@K by Contact Budget", fontweight="bold")
        ax_l.set_xticks(x)
        ax_l.set_xticklabels(labels)
        ax_l.set_ylabel("Lift@K")
        ax_l.axhline(1.0, ls="--", color="gray", lw=1)
        ax_r.set_title(f"H={h} Mean Recall@K by Contact Budget", fontweight="bold")
        ax_r.set_xticks(x)
        ax_r.set_xticklabels(labels)
        ax_r.set_ylabel("Recall@K")
        if row_idx == 0:
            ax_r.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=7.5, frameon=False)

    fig.tight_layout()
    return _save(fig, path)


def _plot_calibration_and_cm(
    cal_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    final_df: pd.DataFrame,
    winners: dict[str, str],
    path: Path,
) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    for row_idx, h in enumerate((1, 3)):
        w_id = winners.get(f"h{h}")
        ax_cal, ax_cm = axes[row_idx]
        sub_c = cal_df[(cal_df["horizon"] == h) & (cal_df["scope"] == "validation_pooled") & (cal_df["experiment_id"] == w_id)]
        if len(sub_c) > 0:
            mx = max(float(sub_c["mean_pred"].max()), float(sub_c["observed_rate"].max())) * 1.1
            ax_cal.plot([0, mx], [0, mx], ls="--", color="gray", lw=1, label="Perfect calibration")
            ax_cal.plot(sub_c["mean_pred"], sub_c["observed_rate"], marker="o", lw=2, label=f"Val pooled: {w_id}")
        if h == 1 and len(cal_df) > 0:
            for tag, style in (("H1_final_raw", ":"), ("H1_final_cal_intercept", "-."), ("H1_final_cal_platt", "-")):
                sc = cal_df[(cal_df["horizon"] == 1) & (cal_df["scope"] == "final_test") & (cal_df["experiment_id"] == tag)]
                if len(sc) > 0:
                    ax_cal.plot(sc["mean_pred"], sc["observed_rate"], marker="s", ls=style, lw=1.4, label=f"2023-01: {tag}")
        ax_cal.set_title(f"H={h} Reliability Curve (10-Quantile Bins)", fontweight="bold")
        ax_cal.set_xlabel("Mean Predicted Probability")
        ax_cal.set_ylabel("Observed Positive Rate")
        ax_cal.legend(frameon=False, fontsize=8)

        # Confusion matrix at Top-1% summed across validation folds for winner
        sub_m = monthly_df[(monthly_df["horizon"] == h) & (monthly_df["stage"] == "validation") & (monthly_df["experiment_id"] == w_id)]
        if len(sub_m) > 0:
            tp = int(sub_m["cm_primary_tp"].sum())
            fp = int(sub_m["cm_primary_fp"].sum())
            fn = int(sub_m["cm_primary_fn"].sum())
            tn = int(sub_m["cm_primary_tn"].sum())
            mat = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
            im = ax_cm.imshow(np.log1p(mat), cmap="Blues")
            ax_cm.set_xticks([0, 1])
            ax_cm.set_xticklabels(["Not Selected (99%)", "Top-1% Selected"])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_yticklabels(["Actual Negative", "Actual Positive"])
            for i in range(2):
                for j in range(2):
                    ax_cm.text(j, i, f"{mat[i, j]:,}", ha="center", va="center", fontweight="bold",
                               color="white" if i == 0 and j == 0 else "black")
            ax_cm.set_title(f"H={h} Top-1% Confusion Matrix ({w_id}, 3 Val Months)", fontweight="bold")
            ax_cm.grid(False)

    fig.tight_layout()
    return _save(fig, path)


def _plot_slices(slice_df: pd.DataFrame, winners: dict[str, str], path: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    for row_idx, h in enumerate((1, 3)):
        w_id = winners.get(f"h{h}")
        sub = slice_df[
            (slice_df["horizon"] == h)
            & (slice_df["scope"] == "validation_pooled")
            & (slice_df["experiment_id"] == w_id)
            & (slice_df["slice_dim"] == "coverage_slice")
        ].sort_values("prevalence", ascending=False)

        ax_l, ax_r = axes[row_idx]
        if len(sub) > 0:
            x = np.arange(len(sub))
            ax_l.bar(x - 0.18, sub["average_precision"].fillna(0.0), width=0.36, label="AP", color="#2b8cbe")
            ax_l.bar(x + 0.18, sub["roc_auc"].fillna(0.0), width=0.36, label="ROC-AUC", color="#74a9cf")
            ax_l.set_xticks(x)
            ax_l.set_xticklabels(sub["slice_value"], rotation=25, ha="right", fontsize=8.5)
            ax_l.set_title(f"H={h} Slice AP & ROC-AUC ({w_id})", fontweight="bold")
            ax_l.legend(frameon=False)

            ax_r.bar(x - 0.18, sub["population_share"] * 100, width=0.36, label="% Population", color="#bdbdbd")
            ax_r.bar(x + 0.18, sub["global_top1pct_share"] * 100, width=0.36, label="% of Top-1% Selected", color="#e6550d")
            ax_r.set_xticks(x)
            ax_r.set_xticklabels(sub["slice_value"], rotation=25, ha="right", fontsize=8.5)
            ax_r.set_ylabel("Share (%)")
            ax_r.set_title(f"H={h} Population vs Top-1% Share by Coverage Slice", fontweight="bold")
            ax_r.legend(frameon=False)

    fig.tight_layout()
    return _save(fig, path)


def _plot_final_holdouts(final_df: pd.DataFrame, path: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    h1 = final_df[final_df["horizon"] == 1]
    h3 = final_df[final_df["horizon"] == 3]

    if len(h1) > 0:
        x = np.arange(len(h1))
        axes[0].bar(x - 0.18, h1["log_loss"].fillna(0), width=0.36, label="Log Loss", color="#3182bd")
        ax2 = axes[0].twinx()
        ax2.plot(x, h1["calibration_ratio"].fillna(np.nan), color="#e6550d", marker="o", lw=2, label="Pred/Obs Ratio")
        ax2.axhline(1.0, ls="--", color="#e6550d", alpha=0.5, lw=1)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(h1["experiment_id"], rotation=25, ha="right", fontsize=8)
        axes[0].set_ylabel("Log Loss")
        ax2.set_ylabel("Calibration Ratio (sum p / sum y)")
        axes[0].set_title("H=1 Final Test (2023-01): Raw vs 2022-12 Calibrated", fontweight="bold")
        ax2.grid(False)

    if len(h3) > 0:
        x = np.arange(len(h3))
        axes[1].bar(x - 0.18, h3["lift_1pct"].fillna(0), width=0.36, label="Lift@1%", color="#31a354")
        ax3 = axes[1].twinx()
        ax3.plot(x, h3["average_precision"].fillna(np.nan), color="#756bb1", marker="s", lw=2, label="AP")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(h3["experiment_id"], rotation=25, ha="right", fontsize=8)
        axes[1].set_ylabel("Lift@1%")
        ax3.set_ylabel("Average Precision")
        axes[1].set_title("H=3 Final Test (2022-11): Monthly vs Quarterly Snapshots", fontweight="bold")
        ax3.grid(False)

    fig.tight_layout()
    return _save(fig, path)


def _plot_learning_and_importance(
    lc_df: pd.DataFrame, fi_df: pd.DataFrame, winners: dict[str, str], path: Path
) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    for row_idx, h in enumerate((1, 3)):
        w_id = winners.get(f"h{h}")
        ax_lc, ax_fi = axes[row_idx]
        sub_lc = lc_df[(lc_df["horizon"] == h) & (lc_df["experiment_id"] == w_id)]
        if len(sub_lc) > 0:
            agg_lc = sub_lc.groupby("ntree", as_index=False)[["train_log_loss", "eval_log_loss"]].mean()
            ax_lc.plot(agg_lc["ntree"], agg_lc["train_log_loss"], marker="o", label="Train Log Loss")
            ax_lc.plot(agg_lc["ntree"], agg_lc["eval_log_loss"], marker="s", label="Validation Log Loss")
            ax_lc.set_title(f"H={h} Learning Curve ({w_id})", fontweight="bold")
            ax_lc.set_xlabel("Trees (iterations)")
            ax_lc.set_ylabel("Log Loss")
            ax_lc.legend(frameon=False)

        sub_fi = fi_df[(fi_df["horizon"] == h) & (fi_df["experiment_id"] == w_id)]
        if len(sub_fi) > 0:
            top = (
                sub_fi.groupby("feature", as_index=False)["importance"]
                .mean()
                .sort_values("importance", ascending=False)
                .head(15)
            )
            ax_fi.barh(top["feature"][::-1], top["importance"][::-1], color="#2c7fb8")
            ax_fi.set_title(f"H={h} Top-15 Feature Importance ({w_id})", fontweight="bold")
            ax_fi.set_xlabel("Mean PredictionValuesChange")
            ax_fi.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    return _save(fig, path)
