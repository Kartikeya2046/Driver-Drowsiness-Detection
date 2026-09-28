"""Phase 5: figures for the final report, built from the result tables Phases 2b-4
already produced (no new experiments). Usage: python -m src.eval.make_report_figures
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_ORDER = ["xgboost", "cnn", "gru", "lstm", "avg_ensemble", "stacked_ensemble"]
MODEL_LABEL = {"xgboost": "XGBoost", "cnn": "1D-CNN", "gru": "GRU", "lstm": "LSTM",
               "avg_ensemble": "Avg ensemble", "stacked_ensemble": "Stacked ensemble"}


def fig_calibration_curve():
    df = pd.read_csv(RESULTS_DIR / "phase2b_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for model in MODEL_ORDER:
        d = df[df.model == model].sort_values("calib_s")
        axes[0].errorbar(d.calib_s, d.acc_mean, yerr=d.acc_std, marker="o", label=MODEL_LABEL[model])
        axes[1].errorbar(d.calib_s, d.auc_mean, yerr=d.auc_std, marker="o", label=MODEL_LABEL[model])
    for ax, title in zip(axes, ("Accuracy", "ROC-AUC")):
        ax.set_xlabel("calibration length (s)")
        ax.set_title(f"{title} vs. calibration length (deployable protocol)")
        ax.axhline(0.5 if title == "Accuracy" else 0.5, color="gray", lw=0.5, ls=":")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration_curve.png", dpi=150)
    plt.close(fig)


def fig_forecaster_baselines():
    df = pd.read_csv(RESULTS_DIR / "phase3_forecaster.csv", index_col=0)
    horizons = ["mae_10s", "mae_30s", "mae_60s"]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(horizons))
    for i, model in enumerate(df.index):
        ax.bar([xi + i * 0.2 for xi in x], df.loc[model, horizons], width=0.2, label=model)
    ax.set_xticks([xi + 0.3 for xi in x])
    ax.set_xticklabels(["10s", "30s", "60s"])
    ax.set_xlabel("forecast horizon")
    ax.set_ylabel("MAE (training-std units)")
    ax.set_title("Forecaster vs. persistence/mean/trend baselines")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "forecaster_baselines.png", dpi=150)
    plt.close(fig)


def fig_leadtime_curve():
    df = pd.read_csv(RESULTS_DIR / "phase3_curve.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for signal in ("reactive", "fc30", "step_now", "fused_rf", "trend"):
        d = df[(df.signal == signal) & (df.dur == 600)].sort_values("fa_per_h")
        axes[0].plot(d.fa_per_h, d.hit_rate, marker=".", label=signal, alpha=0.8)
        axes[1].plot(d.fa_per_h, d.median_lead, marker=".", label=signal, alpha=0.8)
    for ax in axes:
        ax.set_xlabel("false alarms / hour")
        ax.set_xlim(0, 10)
    axes[0].set_ylabel("hit rate")
    axes[0].set_title("Hit rate vs. false-alarm rate (10 min ramps)")
    axes[1].set_ylabel("median lead time (s)")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_title("Lead time vs. false-alarm rate (10 min ramps)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "leadtime_curve.png", dpi=150)
    plt.close(fig)


def fig_leadtime_vs_duration():
    df = pd.read_csv(RESULTS_DIR / "phase3_leadtime.csv")
    d = df[df.budget == 3.0]
    durs = [0, 120, 300, 600]
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, row in d.iterrows():
        ax.plot(durs, [row[f"lead_{dur}"] for dur in durs], marker="o", label=row.signal)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("ramp duration (s); 0 = hard splice (negative control)")
    ax.set_ylabel("median lead time (s)")
    ax.set_title("Lead time vs. transition speed, at ~3 false alarms/hour")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "leadtime_vs_duration.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_calibration_curve()
    fig_forecaster_baselines()
    fig_leadtime_curve()
    fig_leadtime_vs_duration()
    print(f"4 figures -> {FIG_DIR}")
