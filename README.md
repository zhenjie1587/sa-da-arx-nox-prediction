# Sensor-Availability-Aware Dynamic NOx Prediction

Implementation and experimental outputs for a sensor-availability-aware,
delay-aware autoregressive model with exogenous inputs (SA-DA-ARX) for
**contemporaneous NOx prediction (nowcasting)** in a 660-MW coal-fired boiler
under simulated process-sensor missingness.

At sample time `t`, the model predicts the current A- and B-side NOx
concentrations using current and selected delayed process measurements together
with measured NOx history from `t-1` to `t-10`. **The current target is excluded
from the inputs.** The data are sampled once per minute; this does not establish
a one-minute forecasting horizon or a lead time relative to the NOx analyser.

The repository accompanies the manuscript:

> Sensor-Availability-Aware Dynamic Prediction of NOx Emissions for a 660-MW Coal-Fired Boiler under Sensor Missingness

**Author:** Shiyu Guo  
**Affiliation:** CHN Energy Bengbu Power Generation Co., Ltd., Bengbu, Anhui, China  
**Correspondence:** 13196243595@163.com

## Main features

- Chronological training, calibration, and test partitions without shuffling.
- Process-delay selection using the training period only.
- A 62-feature dynamic representation: 27 current process measurements,
  15 selected delayed process measurements, and 20 measured NOx-history values.
- Separate clean and sensor-dropout-trained ridge experts, selected by the
  availability of the actual current and delayed process features.
- A robust regressor with 104 inputs after adding 42 missingness indicators;
  the measured NOx-history features remain observed in these experiments.
- Random entry loss, 60-minute contiguous block loss, and complete outage of
  the calibration-selected process channel F26.
- Scenario-matched five-block calibration, with separate output intervals and
  empirical coverage evaluation.
- Executable SVR-ARX and independently implemented compact/larger LSTM
  baselines, distinguished from archived Source-SVM and Source-LSTM predictions.
- Saved numerical outputs and a script reproducing the moving-block-bootstrap
  sensitivity analysis.

## Repository structure

```text
.
├── artifacts/                 Trained ridge experts, lag map, and model metadata
├── docs/
│   ├── DATA_LICENSE_AND_CITATION.md
│   ├── METHODS_AND_RESULTS.md
│   └── REVISION_REPRODUCIBILITY.md
├── results/
│   ├── figures/               Core publication figures
│   ├── tables/                Core metrics, predictions, and interval results
│   ├── revision_20261008/
│   │   ├── lstm_replicates/    Compact LSTM results and configuration
│   │   └── paired_moving_block_bootstrap.csv
│   └── revision_20261027/
│       ├── lstm_capacity_sensitivity/  Larger LSTM results, logs, checkpoints
│       └── reproduced_diagnostics/    Reproduced bootstrap and load statistics
├── source_data/               Place the separately downloaded workbook here
├── src/                       Core and revision experiment scripts
├── requirements.txt           Core-pipeline dependencies
├── requirements_lstm_revision.txt
└── run_all.py                 Core experiments and figures
```

The revised comparisons are documented in
[docs/REVISION_REPRODUCIBILITY.md](docs/REVISION_REPRODUCIBILITY.md).
Use the CSV files identified below for the revised numerical results. Older
overview workbooks or writing notes may predate the revision.

## Data and chronological partitions

The study uses 10,000 consecutive one-minute observations from the public
660-MW boiler dataset. These are **public benchmark data, not field data from
CHN Energy Bengbu Power Generation Co., Ltd.**

Download version 1 of the original workbook separately:

- [Dataset record, DOI: 10.17632/687fw2n74b.1](https://data.mendeley.com/datasets/687fw2n74b/1)
- [Associated article, DOI: 10.1016/j.energy.2019.04.020](https://doi.org/10.1016/j.energy.2019.04.020)
- Dataset licence: CC BY-NC 3.0, as stated on the dataset record.

Place `data.xlsx` at:

```text
source_data/Data for Dynamic modeling of NOX emission in a 660 MW coal-fired boiler with LSTM/data.xlsx
```

The original workbook and the raw-data CSV are not redistributed in the revision
archive. See [docs/DATA_LICENSE_AND_CITATION.md](docs/DATA_LICENSE_AND_CITATION.md)
for attribution and licence information.

| Partition | One-based sample numbers | Number of samples | Use |
|---|---:|---:|---|
| Feature warm-up | 1–10 | 10 | Construct historical features |
| Training | 11–6500 | 6,490 | Fit preprocessing, select delays, and train models |
| Calibration | 6501–7001 | 501 | Select the critical sensor and calibrate intervals |
| Test | 7002–10000 | 2,999 | Evaluate predictions on an aligned chronological interval |

Historical target inputs are recorded measurements, not recursive predictions.
Artificial missingness is applied to the original process time series before
delayed features are constructed. A missing observation therefore also affects
its subsequent delayed use.

## Reproduce the core experiments

Run commands from the repository root. The original pipeline used Python
3.12.13, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, and scikit-learn 1.8.0;
the saved environment is recorded in `results/tables/execution_summary.json`.

Create an environment:

```bash
python -m venv .venv-core
```

Activate it with `.\.venv-core\Scripts\Activate.ps1` in Windows PowerShell,
or `source .venv-core/bin/activate` on Linux/macOS. Then run:

```bash
python -m pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs the core numerical experiments and figure generation. The
revision LSTM and diagnostic scripts below are separate entry points. Stored
results correspond to the recorded environments; regenerated values and runtime
measurements can vary with library versions and hardware.

## Reproduce the revision analyses

### Independently implemented LSTM baselines

First, in the core environment, export the unchanged public workbook to the
input CSV required by the revision runners:

```bash
python src/prepare_revision_input.py
```

Use a separate Python 3.12 environment for the LSTM audit:

```bash
python -m venv .venv-lstm
```

Activate this environment, then run:

```bash
python -m pip install -r requirements_lstm_revision.txt
python src/run_lstm_replicates.py
python src/run_lstm_capacity_sensitivity.py
```

The recorded LSTM environment uses NumPy 1.26.4, pandas 3.0.3,
scikit-learn 1.5.1, and PyTorch 2.9.0 on CPU. The larger runner imports
`src/run_lstm_candidate.py`; keep that file alongside the other scripts.

Both LSTMs use the common outer chronological partition and exclude the current
target. Their input is a ten-step sequence of process measurements and the
preceding A- and B-side target values. Input/output scalers are fitted on the
outer training partition before its internal split; gradient updates use its
first 90%, while the final 10% selects the checkpoint. Neither the calibration
nor the test interval is passed to the fitting/early-stopping routines.

- **Compact:** two layers of 64 hidden units, dropout 0.10, batch size 256,
  maximum 90 epochs, early-stopping patience 12.
- **Larger:** two layers of 128 hidden units, dropout 0.20, batch size 512,
  maximum 220 epochs, early-stopping patience 25.
- Both use Adam, learning rate 0.001, weight decay 0.0001, standardised-output
  MSE, gradient-norm clipping at 1.0, and seeds 17, 42, and 73.

Both configurations are trained on complete process inputs and use
training-median filling during missingness tests. They are independent
implementations, not reconstructions of the archived Source-LSTM model.
The comparisons do not establish performance against a missingness-augmented
or exhaustively optimised LSTM.

### Bootstrap and measured load-band statistics

Using the supplied chronological prediction outputs, run:

```bash
python src/reproduce_revision_diagnostics.py
```

This script reproduces the archived bootstrap point estimates and interval
endpoints and exports measured NOx statistics by load band. The circular
moving-block bootstrap uses 2,000 resamples, block lengths of 30, 60, and
120 minutes, and seed 20261008. It is a sensitivity analysis on one record,
not population-level inference. Its strict numerical check refers to the
supplied prediction file; run it before overwriting those archived predictions
with results regenerated in another environment.

## Reported reference results

Metrics below use the same 2,999-sample test interval. Macro metrics average
the A- and B-side metrics. RMSE and MAE are in mg/Nm³.

### Complete process-sensor inputs

| Model | Macro RMSE | Macro MAE | Macro R² |
|---|---:|---:|---:|
| SA-DA-ARX | 9.650 | 6.477 | 0.9330 |
| Source-LSTM (archived predictions) | 12.199 | 9.221 | 0.8948 |
| Source-SVM (archived predictions) | 18.664 | 14.621 | 0.7512 |
| SVR-ARX (executed comparator) | 20.939 | 13.458 | 0.6884 |
| Implemented LSTM (compact; 3 seeds) | 14.420 ± 0.674 | 10.544 ± 0.531 | 0.8500 ± 0.0133 |
| Implemented LSTM (larger; 3 seeds) | 10.596 ± 0.059 | 7.557 ± 0.091 | 0.9196 ± 0.0004 |

The 20.9% RMSE reduction refers specifically to the archived Source-LSTM
predictions supplied in the public workbook, not to retraining that source
model under the present protocol. Relative to the larger independently
implemented LSTM configuration, the complete-input RMSE reduction is about 8.9%.
The shared outer protocol does not imply equal hyperparameter-search budgets
or identical optimisation procedures across model families.
The implemented-LSTM rows report mean ± sample standard deviation across
three seeds.

### Process-sensor missingness

| Model and scenario | Macro RMSE | Macro MAE | Macro R² |
|---|---:|---:|---:|
| SA-DA-ARX, 10% MCAR | 11.430 ± 0.064 | 8.032 ± 0.034 | 0.9065 |
| SA-DA-ARX, 20% MCAR | 11.907 ± 0.087 | 8.421 ± 0.044 | 0.8992 |
| SA-DA-ARX, 30% MCAR | 12.383 ± 0.095 | 8.819 ± 0.047 | 0.8914 |
| Non-augmented DA-ARX, 30% MCAR | 19.398 ± 0.309 | 13.435 ± 0.174 | 0.7323 |
| SA-DA-ARX, 20% block loss | 11.486 ± 0.146 | 8.000 ± 0.092 | 0.9056 |
| SA-DA-ARX, full F26 outage | 12.490 | 8.841 | 0.8898 |

MCAR means missing completely at random. Random and block RMSE/MAE results
report mean ± sample standard deviation across five masks on the same record;
their R² column reports the corresponding mean.
Complete inputs and full F26 outage are deterministic for SA-DA-ARX.

### Implemented LSTM stress tests

| Sensor condition | SA-DA-ARX RMSE | LSTM (compact) RMSE | LSTM (larger) RMSE |
|---|---:|---:|---:|
| Complete inputs | 9.650 | 14.420 ± 0.674 | 10.596 ± 0.059 |
| 30% random loss | 12.383 ± 0.095 | 16.303 ± 0.192 | 15.228 ± 0.212 |
| 20% block loss | 11.486 ± 0.146 | 15.958 ± 0.514 | 13.023 ± 0.418 |
| Full F26 outage | 12.490 | 19.453 ± 0.296 | 15.880 ± 0.059 |

For LSTMs, complete-input and F26 summaries use three seeds; random/block
summaries pool three seeds and five common masks (15 runs). These standard
deviations describe different sources of variability from the five-mask
SA-DA-ARX summaries and are not independent-record uncertainty estimates.

### Result files

- `results/tables/scenario_summary.csv`: core model/scenario point metrics.
- `results/tables/regime_summary.csv`: load- and ramp-dependent errors.
- `results/tables/load_band_measured_nox.csv`: actual measured NOx by load band.
- `results/tables/conformal_summary.csv`: interval half-width, empirical
  coverage, and normalised width.
- `results/revision_20261008/lstm_replicates/lstm_common_protocol_summary.csv`:
  compact LSTM results.
- `results/revision_20261027/lstm_capacity_sensitivity/lstm_summary.csv`:
  larger LSTM results; the same folder contains configuration, checkpoints,
  and training histories.
- `results/revision_20261008/paired_moving_block_bootstrap.csv`:
  bootstrap sensitivity results.

## Scope and limitations

- The task is contemporaneous prediction, not validated future forecasting.
  No fixed analyser delay or prediction lead time is claimed.
- The preceding ten minutes of measured NOx must remain available. The method
  has not been validated as a replacement for a failed NOx analyser.
- All 27 process variables are anonymised. F01 is used as an operational load
  proxy, not as a verified physical sensor mapping; F02–F27 are not assigned
  physical identities.
- Missingness scenarios are controlled stress tests, not labelled plant fault
  records. They do not establish real failure frequencies or robustness to
  value-dependent missingness, drift, or correlated multi-sensor failures.
- Calibration and test corruptions are separate episodes generated using the
  same scenario–replicate seed. Their masks are not independently sampled;
  for MCAR, the calibration mask matches the first 501 rows of the test mask.
- Interval half-width is fixed per output within a calibration scenario,
  rather than adapted at every minute. Empirical macro coverage of
  94.5%–97.2% is not a formal finite-sample coverage guarantee under temporal
  dependence.
- Five masks probe location sensitivity within approximately one week from one
  unit. They do not establish generalisation across seasons or generating units,
  completed plant deployment, closed-loop control, or regulatory compliance.

## Licence

The source code is released under the MIT License; see `LICENSE`.
The third-party dataset remains subject to its original CC BY-NC 3.0 licence.
Retain attribution to the dataset and associated article when using the
derived research outputs.
