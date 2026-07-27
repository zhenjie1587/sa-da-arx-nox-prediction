"""Generate publication-ready PNG and vector PDF figures from result tables."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import DATA_FILE, FIGURES_DIR, TABLES_DIR


PALETTE = {
    "proposed": "#1F77B4",
    "clean": "#4C78A8",
    "robust": "#F28E2B",
    "baseline": "#9DA4AA",
    "source": "#6B7280",
    "danger": "#D62728",
    "green": "#2A9D8F",
}


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_data_overview() -> None:
    raw = pd.read_excel(DATA_FILE, sheet_name="original", header=None, skiprows=1)
    raw.columns = ["sample_id", *[f"F{i:02d}" for i in range(1, 28)], "NOx_A", "NOx_B"]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.8), sharex=True)
    axes[0].plot(raw["sample_id"], raw["F01"], lw=0.7, color=PALETTE["clean"])
    axes[0].set_ylabel("F01 load proxy")
    axes[1].plot(raw["sample_id"], raw["NOx_A"], lw=0.65, color="#D55E00")
    axes[1].set_ylabel(r"NOx A (mg/Nm$^3$)")
    axes[2].plot(raw["sample_id"], raw["NOx_B"], lw=0.65, color="#009E73")
    axes[2].set_ylabel(r"NOx B (mg/Nm$^3$)")
    axes[2].set_xlabel("Sequential one-minute sample")
    for axis in axes:
        axis.axvspan(11, 6500, color="#4C78A8", alpha=0.06)
        axis.axvspan(6501, 7001, color="#F28E2B", alpha=0.10)
        axis.axvspan(7002, 10000, color="#59A14F", alpha=0.06)
    axes[0].text(700, axes[0].get_ylim()[1], "Train", va="top", color="#315B7D")
    axes[0].text(6550, axes[0].get_ylim()[1], "Cal.", va="top", color="#9A5B00")
    axes[0].text(8350, axes[0].get_ylim()[1], "Test", va="top", color="#3D7A32")
    fig.tight_layout()
    save_figure(fig, "fig01_data_overview")


def figure_clean_comparison() -> None:
    data = pd.read_csv(TABLES_DIR / "clean_model_comparison.csv").sort_values(
        "rmse_mean", ascending=True
    )
    colors = []
    for _, row in data.iterrows():
        if row["model"] == "SA-DA-ARX (proposed)":
            colors.append(PALETTE["proposed"])
        elif row["model_origin"] == "source_workbook":
            colors.append(PALETTE["source"])
        elif row["model"] == "DA-ARX-Ridge":
            colors.append(PALETTE["clean"])
        else:
            colors.append(PALETTE["baseline"])
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bars = ax.barh(data["model"], data["rmse_mean"], color=colors, height=0.68)
    ax.invert_yaxis()
    ax.set_xlabel(r"Macro-average RMSE (mg/Nm$^3$)")
    for bar, value in zip(bars, data["rmse_mean"]):
        ax.text(value + 0.35, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center")
    ax.set_xlim(0, data["rmse_mean"].max() * 1.14)
    fig.tight_layout()
    save_figure(fig, "fig02_clean_model_comparison")


def figure_random_missingness() -> None:
    data = pd.read_csv(TABLES_DIR / "robustness_summary.csv")
    selected = [
        "SA-DA-ARX (proposed)",
        "DA-ARX-Ridge",
        "Robust-DA-ARX",
        "ExtraTrees-ARX",
        "RF-ARX",
    ]
    scenario_order = ["clean", "mcar10", "mcar20", "mcar30"]
    x_values = np.array([0, 10, 20, 30])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    color_map = {
        "SA-DA-ARX (proposed)": PALETTE["proposed"],
        "DA-ARX-Ridge": PALETTE["danger"],
        "Robust-DA-ARX": PALETTE["robust"],
        "ExtraTrees-ARX": "#7A5195",
        "RF-ARX": "#59A14F",
    }
    for model in selected:
        subset = (
            data[(data["model"] == model) & (data["scenario"].isin(scenario_order))]
            .set_index("scenario")
            .loc[scenario_order]
        )
        ax.errorbar(
            x_values,
            subset["rmse_mean"],
            yerr=subset["rmse_std"],
            marker="o",
            lw=1.7 if model == "SA-DA-ARX (proposed)" else 1.15,
            ms=4,
            capsize=2.5,
            color=color_map[model],
            label=model,
        )
    ax.set_xlabel("Randomly missing process-sensor values (%)")
    ax.set_ylabel(r"Macro-average RMSE (mg/Nm$^3$)")
    ax.set_xticks(x_values)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig03_random_missingness")


def figure_outage_stress_tests() -> None:
    data = pd.read_csv(TABLES_DIR / "robustness_summary.csv")
    models = [
        "DA-ARX-Ridge",
        "Robust-DA-ARX",
        "SA-DA-ARX (proposed)",
        "ExtraTrees-ARX",
    ]
    scenarios = ["block20", "critical_outage"]
    labels = ["20% one-hour blocks", "F26 complete outage"]
    x = np.arange(len(labels))
    width = 0.19
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = [PALETTE["danger"], PALETTE["robust"], PALETTE["proposed"], "#7A5195"]
    for idx, (model, color) in enumerate(zip(models, colors)):
        subset = (
            data[(data["model"] == model) & (data["scenario"].isin(scenarios))]
            .set_index("scenario")
            .loc[scenarios]
        )
        ax.bar(
            x + (idx - 1.5) * width,
            subset["rmse_mean"],
            width,
            yerr=subset["rmse_std"],
            capsize=2,
            label=model,
            color=color,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"Macro-average RMSE (mg/Nm$^3$)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, "fig04_outage_stress_tests")


def figure_predictions() -> None:
    data = pd.read_csv(TABLES_DIR / "predictions_clean.csv").iloc[:500]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    for target, axis in zip(("A", "B"), axes):
        axis.plot(
            data["sample_id"],
            data[f"NOx_{target}_actual"],
            color="#222222",
            lw=1.0,
            label="Measured",
        )
        axis.plot(
            data["sample_id"],
            data[f"SA_DA_ARX_proposed_NOx_{target}"],
            color=PALETTE["proposed"],
            lw=0.95,
            label="Proposed",
        )
        axis.plot(
            data["sample_id"],
            data[f"Source_LSTM_NOx_{target}"],
            color=PALETTE["source"],
            lw=0.75,
            alpha=0.85,
            label="Source LSTM",
        )
        axis.set_ylabel(rf"NOx {target} (mg/Nm$^3$)")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].set_xlabel("Sample ID")
    fig.tight_layout()
    save_figure(fig, "fig05_prediction_series")


def figure_conformal_intervals() -> None:
    data = pd.read_csv(TABLES_DIR / "predictions_proposed_scenarios.csv").iloc[:300]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    for target, axis in zip(("A", "B"), axes):
        x = data["sample_id"].to_numpy()
        lower = data[f"clean_NOx_{target}_lower95"].to_numpy()
        upper = data[f"clean_NOx_{target}_upper95"].to_numpy()
        axis.fill_between(
            x,
            lower,
            upper,
            color=PALETTE["proposed"],
            alpha=0.18,
            label="Blockwise 95% interval",
        )
        axis.plot(
            x,
            data[f"NOx_{target}_actual"],
            color="#222222",
            lw=0.9,
            label="Measured",
        )
        axis.plot(
            x,
            data[f"clean_NOx_{target}_prediction"],
            color=PALETTE["proposed"],
            lw=0.9,
            label="Prediction",
        )
        axis.set_ylabel(rf"NOx {target} (mg/Nm$^3$)")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].set_xlabel("Sample ID")
    fig.tight_layout()
    save_figure(fig, "fig06_conformal_intervals")


def figure_regime_performance() -> None:
    data = pd.read_csv(TABLES_DIR / "regime_summary.csv")
    data = data[
        (data["model"] == "SA-DA-ARX (proposed)")
        & (data["regime_type"] == "dynamic_regime")
        & (data["scenario"].isin(["clean", "mcar30"]))
    ]
    regime_order = ["stable", "ramp_up", "ramp_down"]
    x = np.arange(len(regime_order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for idx, (scenario, label, color) in enumerate(
        [
            ("clean", "Complete sensors", PALETTE["clean"]),
            ("mcar30", "30% random missing", PALETTE["robust"]),
        ]
    ):
        subset = data[data["scenario"] == scenario].set_index("regime").loc[regime_order]
        ax.bar(
            x + (idx - 0.5) * width,
            subset["rmse_mean"],
            width,
            yerr=subset["rmse_std"],
            capsize=2.5,
            label=label,
            color=color,
        )
    ax.set_xticks(x, ["Stable", "Ramp up", "Ramp down"])
    ax.set_ylabel(r"Macro-average RMSE (mg/Nm$^3$)")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig07_regime_performance")


def figure_feature_importance() -> None:
    data = pd.read_csv(TABLES_DIR / "feature_importance.csv").head(10)
    data = data.sort_values("rmse_increase", ascending=True)
    colors = [
        PALETTE["danger"] if feature == "F26" else PALETTE["clean"]
        for feature in data["feature"]
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.barh(data["feature"], data["rmse_increase"], color=colors)
    ax.set_xlabel(r"Increase in calibration RMSE (mg/Nm$^3$)")
    fig.tight_layout()
    save_figure(fig, "fig08_feature_importance")


def figure_selected_lags() -> None:
    data = pd.read_csv(TABLES_DIR / "selected_lags.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    colors = [
        PALETTE["proposed"] if lag > 0 else PALETTE["baseline"]
        for lag in data["selected_lag_min"]
    ]
    ax.bar(data["feature"], data["selected_lag_min"], color=colors, width=0.72)
    ax.set_ylabel("Selected delay (min)")
    ax.set_xlabel("Anonymized process measurement")
    ax.tick_params(axis="x", rotation=65)
    ax.set_yticks(range(0, 11, 2))
    fig.tight_layout()
    save_figure(fig, "fig09_selected_lags")


def figure_ablation_heatmap() -> None:
    data = pd.read_csv(TABLES_DIR / "ablation_study.csv")
    models = [
        "Static-Ridge",
        "AR-Ridge",
        "DA-ARX-Ridge",
        "Robust-DA-ARX",
        "SA-DA-ARX (proposed)",
    ]
    scenarios = ["clean", "mcar10", "mcar20", "mcar30", "block20", "critical_outage"]
    matrix = (
        data.pivot(index="model", columns="scenario", values="rmse_mean")
        .loc[models, scenarios]
        .rename(
            columns={
                "clean": "Clean",
                "mcar10": "MCAR 10%",
                "mcar20": "MCAR 20%",
                "mcar30": "MCAR 30%",
                "block20": "Block 20%",
                "critical_outage": "F26 outage",
            }
        )
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="Blues_r",
        linewidths=0.5,
        cbar_kws={"label": r"RMSE (mg/Nm$^3$)"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    save_figure(fig, "fig10_ablation_heatmap")


def figure_coverage() -> None:
    data = pd.read_csv(TABLES_DIR / "conformal_summary.csv")
    data = data[data["target"] == "Macro"].copy()
    order = ["clean", "mcar10", "mcar20", "mcar30", "block20", "critical_outage"]
    data = data.set_index("scenario").loc[order]
    labels = ["Clean", "MCAR 10%", "MCAR 20%", "MCAR 30%", "Block 20%", "F26 outage"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.bar(
        labels,
        100 * data["picp_mean"],
        yerr=100 * data["picp_std"],
        color=[
            PALETTE["clean"],
            PALETTE["proposed"],
            PALETTE["proposed"],
            PALETTE["proposed"],
            PALETTE["robust"],
            PALETTE["danger"],
        ],
        capsize=2.5,
    )
    ax.axhline(95, color="#222222", ls="--", lw=1.0, label="Nominal 95%")
    ax.set_ylim(85, 100)
    ax.set_ylabel("Empirical coverage (%)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False)
    for bar, value in zip(bars, 100 * data["picp_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.1f}", ha="center", fontsize=7)
    fig.tight_layout()
    save_figure(fig, "fig11_interval_coverage")


def main() -> None:
    configure_style()
    figure_data_overview()
    figure_clean_comparison()
    figure_random_missingness()
    figure_outage_stress_tests()
    figure_predictions()
    figure_conformal_intervals()
    figure_regime_performance()
    figure_feature_importance()
    figure_selected_lags()
    figure_ablation_heatmap()
    figure_coverage()
    print(f"Generated 11 figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
