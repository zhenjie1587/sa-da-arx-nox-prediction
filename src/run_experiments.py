"""Run the complete NOx prediction and sensor-missingness experiment.

The script produces all numerical tables, representative predictions, trained
ridge experts, uncertainty intervals, and a machine-generated result summary.
"""

from __future__ import annotations

import json
import platform
import re
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import wilcoxon
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from config import (
    ARTIFACTS_DIR,
    CONFORMAL_ALPHA,
    DATA_FILE,
    LOAD_PROXY_COLUMN,
    PROCESS_COLUMNS,
    RANDOM_SEED,
    SCENARIO_REPLICATES,
    TABLES_DIR,
    TARGET_COLUMNS,
    TARGET_LAGS,
    TRAIN_DROPOUT_RATES,
)
from data_utils import (
    apply_mcar,
    build_features,
    derive_load_regimes,
    load_dataset,
    make_scenario_process,
    select_exogenous_lags,
)
from models import (
    SensorAvailabilityAwareModel,
    make_baseline_models,
    make_ridge,
)


SCENARIOS = (
    "clean",
    "mcar10",
    "mcar20",
    "mcar30",
    "block20",
    "critical_outage",
)
SCENARIO_LABELS = {
    "clean": "Complete sensors",
    "mcar10": "10% random missing",
    "mcar20": "20% random missing",
    "mcar30": "30% random missing",
    "block20": "20% block missing",
    "critical_outage": "Critical sensor outage",
}


def macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    per_target = []
    for target_idx in range(y_true.shape[1]):
        actual = y_true[:, target_idx]
        prediction = y_pred[:, target_idx]
        per_target.append(
            {
                "mae": mean_absolute_error(actual, prediction),
                "rmse": np.sqrt(mean_squared_error(actual, prediction)),
                "r2": r2_score(actual, prediction),
                "mape_percent": 100
                * mean_absolute_percentage_error(actual, prediction),
            }
        )
    return {
        metric: float(np.mean([row[metric] for row in per_target]))
        for metric in per_target[0]
    }


def detailed_metric_rows(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model: str,
    scenario: str,
    replicate: int,
    missing_rate: float,
    model_origin: str = "executed",
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    target_metrics = []
    for target_idx, target_name in enumerate(TARGET_COLUMNS):
        actual = y_true[:, target_idx]
        prediction = y_pred[:, target_idx]
        metrics = {
            "mae": float(mean_absolute_error(actual, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
            "r2": float(r2_score(actual, prediction)),
            "mape_percent": float(
                100 * mean_absolute_percentage_error(actual, prediction)
            ),
        }
        target_metrics.append(metrics)
        rows.append(
            {
                "model": model,
                "model_origin": model_origin,
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "replicate": replicate,
                "target": target_name,
                "n_samples": len(actual),
                "actual_missing_rate": missing_rate,
                **metrics,
            }
        )
    rows.append(
        {
            "model": model,
            "model_origin": model_origin,
            "scenario": scenario,
            "scenario_label": SCENARIO_LABELS[scenario],
            "replicate": replicate,
            "target": "Macro",
            "n_samples": len(y_true),
            "actual_missing_rate": missing_rate,
            **{
                metric: float(np.mean([row[metric] for row in target_metrics]))
                for metric in target_metrics[0]
            },
        }
    )
    return rows


def conformal_quantile(
    residuals: np.ndarray,
    alpha: float,
    n_time_blocks: int = 5,
) -> np.ndarray:
    """Conservative blockwise conformal quantile for a nonstationary series.

    The calibration period is divided into contiguous temporal blocks. A
    finite-sample conformal quantile is computed in every block and the maximum
    is used. This avoids averaging away a difficult local operating period.
    """
    block_quantiles = []
    for block in np.array_split(residuals, n_time_blocks):
        n = block.shape[0]
        adjusted_quantile = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        block_quantiles.append(
            np.quantile(
                block,
                adjusted_quantile,
                axis=0,
                method="higher",
            )
        )
    return np.max(np.asarray(block_quantiles), axis=0)


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def parse_source_predictions(
    y_test: np.ndarray,
) -> dict[str, np.ndarray]:
    """Read the original authors' precomputed LSTM and SVM predictions."""
    lstm_sheet = pd.read_excel(DATA_FILE, sheet_name="test", header=None)
    lstm = lstm_sheet.iloc[1:, [1, 2, 3, 4]].dropna().astype(float).to_numpy()
    if lstm.shape != (len(y_test), 4):
        raise ValueError(f"Unexpected source LSTM prediction shape: {lstm.shape}")
    if not np.allclose(lstm[:, :2], y_test, atol=1e-3):
        raise ValueError("Source LSTM measured values do not align with the test split.")

    svm_sheet = pd.read_excel(DATA_FILE, sheet_name="SVMTest", header=None)
    svm = svm_sheet.iloc[1:, [1, 2, 5, 6]].dropna().astype(float).to_numpy()
    # SVMTest includes sample 7001. Remove it so that all clean comparisons use
    # the common sample 7002--10000 period.
    svm = svm[1:]
    if svm.shape != (len(y_test), 4):
        raise ValueError(f"Unexpected source SVM prediction shape: {svm.shape}")
    if not np.allclose(svm[:, [0, 2]], y_test, atol=1e-3):
        raise ValueError("Source SVM measured values do not align with the test split.")

    return {
        "Source-LSTM": lstm[:, 2:4],
        "Source-SVM": svm[:, [1, 3]],
    }


def add_regime_rows(
    rows: list[dict[str, float | int | str]],
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model: str,
    scenario: str,
    replicate: int,
    regime_frame: pd.DataFrame,
    test_idx: np.ndarray,
) -> None:
    for regime_type in ("dynamic_regime", "load_band"):
        labels = regime_frame[regime_type].iloc[test_idx].to_numpy()
        for label in pd.unique(labels):
            mask = labels == label
            metrics = macro_metrics(y_true[mask], y_pred[mask])
            rows.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "replicate": replicate,
                    "regime_type": regime_type,
                    "regime": str(label),
                    "n_samples": int(mask.sum()),
                    **metrics,
                }
            )


def write_markdown_summary(
    summary: dict[str, object],
    scenario_summary: pd.DataFrame,
    critical_sensor: str,
) -> None:
    proposed_clean = scenario_summary[
        (scenario_summary["model"] == "SA-DA-ARX (proposed)")
        & (scenario_summary["scenario"] == "clean")
        & (scenario_summary["target"] == "Macro")
    ].iloc[0]
    source_lstm = scenario_summary[
        (scenario_summary["model"] == "Source-LSTM")
        & (scenario_summary["scenario"] == "clean")
        & (scenario_summary["target"] == "Macro")
    ].iloc[0]
    proposed_mcar30 = scenario_summary[
        (scenario_summary["model"] == "SA-DA-ARX (proposed)")
        & (scenario_summary["scenario"] == "mcar30")
        & (scenario_summary["target"] == "Macro")
    ].iloc[0]
    clean_model_mcar30 = scenario_summary[
        (scenario_summary["model"] == "DA-ARX-Ridge")
        & (scenario_summary["scenario"] == "mcar30")
        & (scenario_summary["target"] == "Macro")
    ].iloc[0]

    clean_gain = 100 * (
        1 - proposed_clean["rmse_mean"] / source_lstm["rmse_mean"]
    )
    missing_gain = 100 * (
        1 - proposed_mcar30["rmse_mean"] / clean_model_mcar30["rmse_mean"]
    )
    lines = [
        "# 实验结果摘要",
        "",
        "## 推荐论文题目",
        "",
        "中文：传感器缺失条件下660 MW燃煤锅炉NOx排放的可用性感知动态预测",
        "",
        "英文：Sensor-Availability-Aware Dynamic Prediction of NOx Emissions "
        "for a 660-MW Coal-Fired Boiler under Sensor Missingness",
        "",
        "## 数据与划分",
        "",
        "- 数据来自公开的660 MW切圆燃烧锅炉运行数据，共10,000个一分钟样本，覆盖约7天。",
        "- 原始工作簿没有提供27个运行变量的字段名称，因此实验统一使用F01–F27；"
        "F01仅作为机组负荷代理变量使用，不能在论文中擅自扩展为其他具体测点。",
        "- 训练集：样本11–6500；校准集：6501–7001；测试集：7002–10000。",
        "- 测试采用严格时间顺序划分，没有随机打乱，避免相邻时刻信息泄漏。",
        "",
        "## 主方法",
        "",
        "SA-DA-ARX由两个动态岭回归专家组成：完整传感器专家使用干净数据训练；"
        "缺失专家使用10%、20%和30%随机传感器掉线增强训练。运行时只要检测到"
        "运行参数缺失，就切换到缺失专家。输入包括运行参数、训练集选择的时滞参数"
        "以及前10分钟的A/B侧NOx历史值。95%预测区间由独立校准集上的五段分块保形预测获得。",
        "",
        "## 主要结果",
        "",
        f"- 完整传感器下，主方法宏平均RMSE为"
        f"{proposed_clean['rmse_mean']:.3f} mg/Nm³，MAE为"
        f"{proposed_clean['mae_mean']:.3f} mg/Nm³，R²为"
        f"{proposed_clean['r2_mean']:.4f}。",
        f"- 原文件LSTM在同一测试时段的宏平均RMSE为"
        f"{source_lstm['rmse_mean']:.3f} mg/Nm³；主方法相对降低"
        f"{clean_gain:.1f}%。",
        f"- 30%随机传感器缺失时，主方法RMSE为"
        f"{proposed_mcar30['rmse_mean']:.3f}±"
        f"{proposed_mcar30['rmse_std']:.3f} mg/Nm³；未做掉线增强的"
        f"DA-ARX-Ridge为{clean_model_mcar30['rmse_mean']:.3f} mg/Nm³，"
        f"主方法相对降低{missing_gain:.1f}%。",
        f"- 训练集和校准集确定的关键运行变量为{critical_sensor}；"
        "关键传感器整段掉线结果见robustness_summary.csv。",
        "",
        "## 论文可以主张的创新点",
        "",
        "1. 在不随机打乱时序的条件下，将运行参数时滞和NOx自回归信息统一到轻量动态模型中。",
        "2. 通过传感器掉线增强和可用性感知双专家切换，在保持完整数据精度的同时提升缺失场景鲁棒性。",
        "3. 同时报告随机缺失、连续块缺失、关键传感器整段掉线和升降负荷工况，并给出95%预测区间。",
        "",
        "## 必须保留的限制说明",
        "",
        "- 该数据不是国能蚌埠发电厂数据，论文只能表述为“与企业同容量等级机组应用需求相匹配的公开数据验证”。",
        "- 公开工作簿缺少测点名称和单位，F02–F27不能自行解释；正式投稿前最好由企业专家完成测点映射。",
        "- 模型使用历史NOx值，因此定位是短时动态预测和辅助测点异常期间的辅助估计，"
        "不能表述为完全不依赖NOx分析仪的永久替代系统。",
        "- 缺失数据为可重复的人工注入场景，不是真实故障标签。",
        "",
        "## 输出文件",
        "",
        "- `results/tables/`：全部逐次结果、均值/标准差、工况结果、消融、统计检验和预测值。",
        "- `results/figures/`：论文可直接筛选使用的PNG和PDF图。",
        "- `artifacts/`：两个已训练岭回归专家、时滞表和模型元数据。",
        "- `results/results_summary.xlsx`：汇总表和图表。",
        "",
        f"实验运行环境：Python {summary['python_version']}，"
        f"scikit-learn {summary['sklearn_version']}。",
        "",
    ]
    (Path(TABLES_DIR).parent / "EXPERIMENT_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    start_time = time.perf_counter()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset()
    lag_map, lag_table = select_exogenous_lags(
        bundle.process, bundle.target, bundle.train_idx
    )
    lag_table.to_csv(TABLES_DIR / "selected_lags.csv", index=False)

    static_features = build_features(
        bundle.process,
        bundle.target,
        lag_map,
        include_exogenous_lags=False,
        include_target_lags=False,
        target_lags=TARGET_LAGS,
    )
    ar_features = build_features(
        bundle.process,
        bundle.target,
        lag_map,
        include_exogenous_lags=False,
        include_target_lags=True,
        target_lags=TARGET_LAGS,
    )
    darx_features = build_features(
        bundle.process,
        bundle.target,
        lag_map,
        include_exogenous_lags=True,
        include_target_lags=True,
        target_lags=TARGET_LAGS,
    )
    feature_sets = {
        "static": static_features,
        "ar": ar_features,
        "darx": darx_features,
    }
    y_train = bundle.target.iloc[bundle.train_idx]

    models: dict[str, object] = {}
    model_feature_set: dict[str, str] = {}
    runtime_rows: list[dict[str, float | str]] = []

    for model_name, feature_key in (
        ("Static-Ridge", "static"),
        ("AR-Ridge", "ar"),
        ("DA-ARX-Ridge", "darx"),
    ):
        model = make_ridge(robust=False)
        fit_start = time.perf_counter()
        model.fit(feature_sets[feature_key].iloc[bundle.train_idx], y_train)
        runtime_rows.append(
            {
                "model": model_name,
                "fit_seconds": time.perf_counter() - fit_start,
            }
        )
        models[model_name] = model
        model_feature_set[model_name] = feature_key

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for model_name, model in make_baseline_models().items():
            fit_start = time.perf_counter()
            model.fit(darx_features.iloc[bundle.train_idx], y_train)
            runtime_rows.append(
                {
                    "model": model_name,
                    "fit_seconds": time.perf_counter() - fit_start,
                }
            )
            models[model_name] = model
            model_feature_set[model_name] = "darx"

    augmented_x = [darx_features.iloc[bundle.train_idx]]
    augmented_y = [y_train]
    for dropout_idx, dropout_rate in enumerate(TRAIN_DROPOUT_RATES):
        dropped_process, _ = apply_mcar(
            bundle.process,
            bundle.train_idx,
            dropout_rate,
            RANDOM_SEED + 100 * (dropout_idx + 1),
        )
        dropped_features = build_features(
            dropped_process,
            bundle.target,
            lag_map,
            include_exogenous_lags=True,
            include_target_lags=True,
            target_lags=TARGET_LAGS,
        )
        augmented_x.append(dropped_features.iloc[bundle.train_idx])
        augmented_y.append(y_train)
    robust_x = pd.concat(augmented_x, axis=0, ignore_index=True)
    robust_y = pd.concat(augmented_y, axis=0, ignore_index=True)

    robust_expert = make_ridge(robust=True)
    fit_start = time.perf_counter()
    robust_expert.fit(robust_x, robust_y)
    robust_fit_seconds = time.perf_counter() - fit_start
    runtime_rows.append(
        {
            "model": "Robust-DA-ARX",
            "fit_seconds": robust_fit_seconds,
        }
    )
    models["Robust-DA-ARX"] = robust_expert
    model_feature_set["Robust-DA-ARX"] = "darx"

    clean_expert = models["DA-ARX-Ridge"]
    proposed = SensorAvailabilityAwareModel(clean_expert, robust_expert)
    models["SA-DA-ARX (proposed)"] = proposed
    model_feature_set["SA-DA-ARX (proposed)"] = "darx"
    runtime_rows.append(
        {
            "model": "SA-DA-ARX (proposed)",
            "fit_seconds": runtime_rows[
                [row["model"] for row in runtime_rows].index("DA-ARX-Ridge")
            ]["fit_seconds"]
            + robust_fit_seconds,
        }
    )

    # Training/calibration-only permutation importance determines the critical
    # sensor used in the complete-outage stress test.
    clean_calibration = darx_features.iloc[bundle.calibration_idx]
    calibration_target = bundle.target.iloc[bundle.calibration_idx].to_numpy()
    calibration_base_prediction = clean_expert.predict(clean_calibration)
    base_rmse = macro_metrics(
        calibration_target, calibration_base_prediction
    )["rmse"]
    importance_rows = []
    rng = np.random.default_rng(RANDOM_SEED + 999)
    for sensor in PROCESS_COLUMNS:
        permuted_process = bundle.process.copy()
        permuted_values = (
            permuted_process.loc[bundle.calibration_idx, sensor].to_numpy().copy()
        )
        rng.shuffle(permuted_values)
        permuted_process.loc[bundle.calibration_idx, sensor] = permuted_values
        permuted_features = build_features(
            permuted_process,
            bundle.target,
            lag_map,
            include_exogenous_lags=True,
            include_target_lags=True,
            target_lags=TARGET_LAGS,
        )
        prediction = clean_expert.predict(
            permuted_features.iloc[bundle.calibration_idx]
        )
        permutation_rmse = macro_metrics(calibration_target, prediction)["rmse"]
        importance_rows.append(
            {
                "feature": sensor,
                "permuted_rmse": permutation_rmse,
                "rmse_increase": permutation_rmse - base_rmse,
            }
        )
    feature_importance = pd.DataFrame(importance_rows).sort_values(
        "rmse_increase", ascending=False
    )
    feature_importance.to_csv(
        TABLES_DIR / "feature_importance.csv", index=False
    )
    critical_sensor = str(feature_importance.iloc[0]["feature"])

    regime_frame, load_thresholds = derive_load_regimes(
        bundle.process, bundle.train_idx
    )
    y_test = bundle.target.iloc[bundle.test_idx].to_numpy()
    metric_rows: list[dict[str, float | int | str]] = []
    conformal_rows: list[dict[str, float | int | str]] = []
    regime_rows: list[dict[str, float | int | str]] = []
    prediction_cache: dict[tuple[str, int, str], np.ndarray] = {}

    clean_prediction_table = pd.DataFrame(
        {
            "sample_id": bundle.sample_id.iloc[bundle.test_idx].to_numpy(),
            LOAD_PROXY_COLUMN: bundle.process[LOAD_PROXY_COLUMN]
            .iloc[bundle.test_idx]
            .to_numpy(),
            "NOx_A_actual": y_test[:, 0],
            "NOx_B_actual": y_test[:, 1],
            "dynamic_regime": regime_frame["dynamic_regime"]
            .iloc[bundle.test_idx]
            .to_numpy(),
            "load_band": regime_frame["load_band"]
            .iloc[bundle.test_idx]
            .to_numpy(),
        }
    )
    proposed_scenario_table = clean_prediction_table.copy()

    scenario_replicates = {
        "clean": 1,
        "mcar10": SCENARIO_REPLICATES,
        "mcar20": SCENARIO_REPLICATES,
        "mcar30": SCENARIO_REPLICATES,
        "block20": SCENARIO_REPLICATES,
        "critical_outage": 1,
    }

    for scenario in SCENARIOS:
        for replicate in range(scenario_replicates[scenario]):
            corrupted_process, actual_missing_rate = make_scenario_process(
                bundle.process,
                bundle.test_idx,
                scenario,
                replicate,
                critical_sensor,
            )
            scenario_features = {
                "static": build_features(
                    corrupted_process,
                    bundle.target,
                    lag_map,
                    include_exogenous_lags=False,
                    include_target_lags=False,
                    target_lags=TARGET_LAGS,
                ),
                "ar": build_features(
                    corrupted_process,
                    bundle.target,
                    lag_map,
                    include_exogenous_lags=False,
                    include_target_lags=True,
                    target_lags=TARGET_LAGS,
                ),
                "darx": build_features(
                    corrupted_process,
                    bundle.target,
                    lag_map,
                    include_exogenous_lags=True,
                    include_target_lags=True,
                    target_lags=TARGET_LAGS,
                ),
            }
            for model_name, model in models.items():
                features = scenario_features[
                    model_feature_set[model_name]
                ].iloc[bundle.test_idx]
                predict_start = time.perf_counter()
                prediction = model.predict(features)
                predict_seconds = time.perf_counter() - predict_start
                metric_rows.extend(
                    detailed_metric_rows(
                        y_true=y_test,
                        y_pred=prediction,
                        model=model_name,
                        scenario=scenario,
                        replicate=replicate,
                        missing_rate=actual_missing_rate,
                    )
                )
                if replicate == 0:
                    prediction_cache[(scenario, replicate, model_name)] = prediction
                    if scenario == "clean":
                        clean_prediction_table[
                            f"{sanitize(model_name)}_NOx_A"
                        ] = prediction[:, 0]
                        clean_prediction_table[
                            f"{sanitize(model_name)}_NOx_B"
                        ] = prediction[:, 1]
                if model_name == "SA-DA-ARX (proposed)" and replicate == 0:
                    proposed_scenario_table[
                        f"{scenario}_NOx_A_prediction"
                    ] = prediction[:, 0]
                    proposed_scenario_table[
                        f"{scenario}_NOx_B_prediction"
                    ] = prediction[:, 1]
                if scenario in ("clean", "mcar30"):
                    add_regime_rows(
                        regime_rows,
                        y_true=y_test,
                        y_pred=prediction,
                        model=model_name,
                        scenario=scenario,
                        replicate=replicate,
                        regime_frame=regime_frame,
                        test_idx=bundle.test_idx,
                    )
                if scenario == "clean" and replicate == 0:
                    runtime_rows.append(
                        {
                            "model": model_name,
                            "predict_seconds_2999_samples": predict_seconds,
                        }
                    )

            # Scenario-matched conservative blockwise conformal intervals.
            corrupted_calibration_process, calibration_missing_rate = (
                make_scenario_process(
                    bundle.process,
                    bundle.calibration_idx,
                    scenario,
                    replicate,
                    critical_sensor,
                )
            )
            calibration_features = build_features(
                corrupted_calibration_process,
                bundle.target,
                lag_map,
                include_exogenous_lags=True,
                include_target_lags=True,
                target_lags=TARGET_LAGS,
            ).iloc[bundle.calibration_idx]
            calibration_prediction = proposed.predict(calibration_features)
            residuals = np.abs(calibration_target - calibration_prediction)
            quantile = conformal_quantile(residuals, CONFORMAL_ALPHA)

            proposed_prediction = prediction_cache.get(
                (scenario, replicate, "SA-DA-ARX (proposed)")
            )
            if proposed_prediction is None:
                proposed_prediction = proposed.predict(
                    scenario_features["darx"].iloc[bundle.test_idx]
                )
            lower = proposed_prediction - quantile
            upper = proposed_prediction + quantile
            coverage = (y_test >= lower) & (y_test <= upper)
            train_range = (
                bundle.target.iloc[bundle.train_idx].max(axis=0)
                - bundle.target.iloc[bundle.train_idx].min(axis=0)
            ).to_numpy()
            pinaw = (upper - lower) / train_range
            for target_idx, target_name in enumerate(TARGET_COLUMNS):
                conformal_rows.append(
                    {
                        "scenario": scenario,
                        "scenario_label": SCENARIO_LABELS[scenario],
                        "replicate": replicate,
                        "target": target_name,
                        "actual_missing_rate": actual_missing_rate,
                        "calibration_missing_rate": calibration_missing_rate,
                        "quantile_half_width": quantile[target_idx],
                        "picp": float(coverage[:, target_idx].mean()),
                        "pinaw": float(pinaw[:, target_idx].mean()),
                    }
                )
            conformal_rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "replicate": replicate,
                    "target": "Macro",
                    "actual_missing_rate": actual_missing_rate,
                    "calibration_missing_rate": calibration_missing_rate,
                    "quantile_half_width": float(np.mean(quantile)),
                    "picp": float(coverage.mean()),
                    "pinaw": float(pinaw.mean()),
                }
            )
            if replicate == 0:
                proposed_scenario_table[f"{scenario}_NOx_A_lower95"] = lower[:, 0]
                proposed_scenario_table[f"{scenario}_NOx_A_upper95"] = upper[:, 0]
                proposed_scenario_table[f"{scenario}_NOx_B_lower95"] = lower[:, 1]
                proposed_scenario_table[f"{scenario}_NOx_B_upper95"] = upper[:, 1]

    # Add the authors' original workbook predictions to the clean comparison.
    source_predictions = parse_source_predictions(y_test)
    for model_name, prediction in source_predictions.items():
        metric_rows.extend(
            detailed_metric_rows(
                y_true=y_test,
                y_pred=prediction,
                model=model_name,
                scenario="clean",
                replicate=0,
                missing_rate=0.0,
                model_origin="source_workbook",
            )
        )
        prediction_cache[("clean", 0, model_name)] = prediction
        clean_prediction_table[f"{sanitize(model_name)}_NOx_A"] = prediction[:, 0]
        clean_prediction_table[f"{sanitize(model_name)}_NOx_B"] = prediction[:, 1]
        add_regime_rows(
            regime_rows,
            y_true=y_test,
            y_pred=prediction,
            model=model_name,
            scenario="clean",
            replicate=0,
            regime_frame=regime_frame,
            test_idx=bundle.test_idx,
        )

    metrics_replicates = pd.DataFrame(metric_rows)
    metrics_replicates.to_csv(
        TABLES_DIR / "scenario_metrics_replicates.csv", index=False
    )
    summary_aggregations = {
        "actual_missing_rate": ["mean"],
        "mae": ["mean", "std"],
        "rmse": ["mean", "std"],
        "r2": ["mean", "std"],
        "mape_percent": ["mean", "std"],
        "replicate": ["nunique"],
    }
    scenario_summary = (
        metrics_replicates.groupby(
            [
                "model",
                "model_origin",
                "scenario",
                "scenario_label",
                "target",
            ],
            as_index=False,
        )
        .agg(summary_aggregations)
    )
    scenario_summary.columns = [
        "_".join(column).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in scenario_summary.columns
    ]
    for column in (
        "mae_std",
        "rmse_std",
        "r2_std",
        "mape_percent_std",
    ):
        scenario_summary[column] = scenario_summary[column].fillna(0.0)
    scenario_summary = scenario_summary.rename(
        columns={"replicate_nunique": "n_replicates"}
    )
    scenario_summary.to_csv(TABLES_DIR / "scenario_summary.csv", index=False)

    clean_comparison = scenario_summary[
        (scenario_summary["scenario"] == "clean")
        & (scenario_summary["target"] == "Macro")
    ].sort_values("rmse_mean")
    clean_comparison.to_csv(TABLES_DIR / "clean_model_comparison.csv", index=False)

    robustness_summary = scenario_summary[
        (scenario_summary["target"] == "Macro")
        & (~scenario_summary["model"].isin(source_predictions.keys()))
    ].copy()
    clean_rmse_lookup = robustness_summary[
        robustness_summary["scenario"] == "clean"
    ].set_index("model")["rmse_mean"]
    robustness_summary["rmse_degradation_percent"] = robustness_summary.apply(
        lambda row: 100
        * (row["rmse_mean"] / clean_rmse_lookup.loc[row["model"]] - 1),
        axis=1,
    )
    robustness_summary.to_csv(
        TABLES_DIR / "robustness_summary.csv", index=False
    )

    ablation_models = (
        "Static-Ridge",
        "AR-Ridge",
        "DA-ARX-Ridge",
        "Robust-DA-ARX",
        "SA-DA-ARX (proposed)",
    )
    ablation = robustness_summary[
        robustness_summary["model"].isin(ablation_models)
    ].copy()
    ablation.to_csv(TABLES_DIR / "ablation_study.csv", index=False)

    conformal_replicates = pd.DataFrame(conformal_rows)
    conformal_replicates.to_csv(
        TABLES_DIR / "conformal_metrics_replicates.csv", index=False
    )
    conformal_summary = (
        conformal_replicates.groupby(
            ["scenario", "scenario_label", "target"], as_index=False
        )
        .agg(
            n_replicates=("replicate", "nunique"),
            actual_missing_rate=("actual_missing_rate", "mean"),
            quantile_half_width_mean=("quantile_half_width", "mean"),
            quantile_half_width_std=("quantile_half_width", "std"),
            picp_mean=("picp", "mean"),
            picp_std=("picp", "std"),
            pinaw_mean=("pinaw", "mean"),
            pinaw_std=("pinaw", "std"),
        )
        .fillna(0.0)
    )
    conformal_summary.to_csv(
        TABLES_DIR / "conformal_summary.csv", index=False
    )

    regime_replicates = pd.DataFrame(regime_rows)
    regime_replicates.to_csv(
        TABLES_DIR / "regime_metrics_replicates.csv", index=False
    )
    regime_summary = (
        regime_replicates.groupby(
            ["model", "scenario", "regime_type", "regime"], as_index=False
        )
        .agg(
            n_samples=("n_samples", "mean"),
            n_replicates=("replicate", "nunique"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .fillna(0.0)
    )
    regime_summary.to_csv(TABLES_DIR / "regime_summary.csv", index=False)

    # Paired absolute-error tests use the fixed representative realization.
    statistical_rows = []
    comparison_pairs = [
        ("clean", "Source-LSTM"),
        ("clean", "Source-SVM"),
        ("clean", "DA-ARX-Ridge"),
        ("clean", "MLP-ARX"),
        ("mcar30", "DA-ARX-Ridge"),
        ("mcar30", "MLP-ARX"),
        ("mcar30", "ExtraTrees-ARX"),
    ]
    for scenario, comparison_model in comparison_pairs:
        proposed_prediction = prediction_cache[
            (scenario, 0, "SA-DA-ARX (proposed)")
        ]
        comparison_prediction = prediction_cache[
            (scenario, 0, comparison_model)
        ]
        for target_idx, target_name in enumerate(TARGET_COLUMNS):
            proposed_error = np.abs(
                y_test[:, target_idx] - proposed_prediction[:, target_idx]
            )
            comparison_error = np.abs(
                y_test[:, target_idx] - comparison_prediction[:, target_idx]
            )
            if np.allclose(proposed_error, comparison_error):
                statistic = 0.0
                p_value = 1.0
            else:
                result = wilcoxon(
                    proposed_error,
                    comparison_error,
                    alternative="less",
                    zero_method="wilcox",
                )
                statistic = result.statistic
                p_value = result.pvalue
            statistical_rows.append(
                {
                    "scenario": scenario,
                    "target": target_name,
                    "proposed_model": "SA-DA-ARX (proposed)",
                    "comparison_model": comparison_model,
                    "proposed_mae": proposed_error.mean(),
                    "comparison_mae": comparison_error.mean(),
                    "mae_reduction_percent": 100
                    * (1 - proposed_error.mean() / comparison_error.mean()),
                    "wilcoxon_statistic": statistic,
                    "one_sided_p_value": p_value,
                }
            )
    pd.DataFrame(statistical_rows).to_csv(
        TABLES_DIR / "statistical_tests.csv", index=False
    )

    runtime_frame = pd.DataFrame(runtime_rows)
    runtime_summary = (
        runtime_frame.groupby("model", as_index=False)
        .agg(
            fit_seconds=("fit_seconds", "max"),
            predict_seconds_2999_samples=(
                "predict_seconds_2999_samples",
                "max",
            ),
        )
        .fillna(0.0)
    )
    runtime_summary.to_csv(TABLES_DIR / "model_runtime.csv", index=False)

    clean_prediction_table.to_csv(
        TABLES_DIR / "predictions_clean.csv", index=False
    )
    proposed_scenario_table.to_csv(
        TABLES_DIR / "predictions_proposed_scenarios.csv", index=False
    )

    descriptive = bundle.raw.describe().T.reset_index().rename(
        columns={"index": "variable"}
    )
    descriptive["missing_count"] = [
        int(bundle.raw[column].isna().sum()) for column in descriptive["variable"]
    ]
    descriptive["zero_count"] = [
        int((bundle.raw[column] == 0).sum()) for column in descriptive["variable"]
    ]
    descriptive.to_csv(TABLES_DIR / "data_quality.csv", index=False)

    dictionary_rows = [
        {
            "variable": "sample_id",
            "role": "index",
            "description": "Sequential one-minute sample number",
            "unit": "index",
            "mapping_status": "confirmed from workbook",
        }
    ]
    for process_column in PROCESS_COLUMNS:
        dictionary_rows.append(
            {
                "variable": process_column,
                "role": "process input",
                "description": (
                    "Unit-load proxy"
                    if process_column == LOAD_PROXY_COLUMN
                    else "Anonymized boiler operating measurement"
                ),
                "unit": (
                    "MW (inferred)"
                    if process_column == LOAD_PROXY_COLUMN
                    else "not supplied"
                ),
                "mapping_status": (
                    "inferred from range and source context"
                    if process_column == LOAD_PROXY_COLUMN
                    else "requires author/enterprise mapping"
                ),
            }
        )
    for target_column in TARGET_COLUMNS:
        dictionary_rows.append(
            {
                "variable": target_column,
                "role": "prediction target",
                "description": f"{target_column[-1]}-side measured NOx concentration",
                "unit": "mg/Nm3",
                "mapping_status": "confirmed by source workbook and article",
            }
        )
    pd.DataFrame(dictionary_rows).to_csv(
        TABLES_DIR / "data_dictionary.csv", index=False
    )

    joblib.dump(clean_expert, ARTIFACTS_DIR / "clean_expert.joblib")
    joblib.dump(robust_expert, ARTIFACTS_DIR / "robust_expert.joblib")
    (ARTIFACTS_DIR / "lag_map.json").write_text(
        json.dumps(lag_map, indent=2), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "model_metadata.json").write_text(
        json.dumps(
            {
                "model_name": "SA-DA-ARX",
                "clean_expert": "clean_expert.joblib",
                "robust_expert": "robust_expert.joblib",
                "critical_sensor": critical_sensor,
                "process_columns": list(PROCESS_COLUMNS),
                "target_columns": list(TARGET_COLUMNS),
                "target_lags": list(TARGET_LAGS),
                "load_proxy": LOAD_PROXY_COLUMN,
                "load_thresholds": load_thresholds,
                "gate": "Use robust expert when any derived input is missing.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    total_seconds = time.perf_counter() - start_time
    execution_summary = {
        "status": "completed",
        "data_file": str(DATA_FILE),
        "n_samples": len(bundle.raw),
        "n_process_variables": len(PROCESS_COLUMNS),
        "targets": list(TARGET_COLUMNS),
        "train_samples": len(bundle.train_idx),
        "calibration_samples": len(bundle.calibration_idx),
        "test_samples": len(bundle.test_idx),
        "critical_sensor": critical_sensor,
        "load_thresholds": load_thresholds,
        "total_runtime_seconds": total_seconds,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "random_seed": RANDOM_SEED,
    }
    (TABLES_DIR / "execution_summary.json").write_text(
        json.dumps(execution_summary, indent=2), encoding="utf-8"
    )
    write_markdown_summary(execution_summary, scenario_summary, critical_sensor)
    print(json.dumps(execution_summary, indent=2))


if __name__ == "__main__":
    main()
