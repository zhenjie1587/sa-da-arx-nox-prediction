# Sensor-Availability-Aware Dynamic NOx Prediction

Reproducible implementation of a sensor-availability-aware, delay-aware autoregressive model with exogenous inputs (SA-DA-ARX) for one-minute-ahead NOx prediction in a 660 MW coal-fired boiler under simulated sensor missingness.

The repository accompanies the manuscript:

> Sensor-Availability-Aware Dynamic Prediction of NOx Emissions for a 660-MW Coal-Fired Boiler under Sensor Missingness

**Author:** Shiyu Guo  
**Affiliation:** CHN Energy Bengbu Power Generation Co., Ltd., Bengbu, Anhui, China  
**Contact:** 13196243595@163.com

## Main features

- Chronological training, calibration, and test splits without random shuffling.
- Training-only process-delay selection.
- Ten-minute autoregressive NOx history.
- Clean-data and sensor-dropout-augmented ridge experts.
- Availability-aware hard gating between the two experts.
- Random, block, and critical-sensor outage stress tests.
- Blockwise residual calibration for empirical prediction intervals.
- Reproducible tables, figures, trained artifacts, and fixed random seeds.

## Repository structure

```text
.
├── artifacts/                 Trained experts, lag map, and model metadata
├── docs/                      Method and data-licence notes
├── results/
│   ├── figures/               Publication figures in PDF and PNG
│   ├── tables/                Predictions, metrics, and statistical outputs
│   └── results_summary.xlsx   Consolidated result workbook
├── source_data/               Place the public source workbook here
├── src/                       Data processing, models, experiments, and figures
├── requirements.txt
└── run_all.py                 One-command experiment entry point
```

## Data

The original workbook is not redistributed in this repository. Download version 1 of the public dataset from Mendeley Data:

- Dataset DOI: `10.17632/687fw2n74b.1`
- Associated article DOI: `10.1016/j.energy.2019.04.020`
- Dataset licence: CC BY-NC 3.0, as stated on the repository record

Place `data.xlsx` at exactly:

```text
source_data/Data for Dynamic modeling of NOX emission in a 660 MW coal-fired boiler with LSTM/data.xlsx
```

See `docs/DATA_LICENSE_AND_CITATION.md` for attribution and use restrictions.

## Environment

Python 3.12 is recommended.

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Reproduce the experiments

```bash
python run_all.py
```

The command regenerates the numerical experiments and publication figures under `results/`.

## Reported reference results

The final chronological test period contains 2,999 one-minute samples.

| Model or scenario | Macro RMSE (mg/Nm3) | Macro MAE (mg/Nm3) | R2 |
|---|---:|---:|---:|
| SA-DA-ARX, complete sensors | 9.650 | 6.477 | 0.9330 |
| Source LSTM, complete sensors | 12.199 | 9.221 | 0.8948 |
| SA-DA-ARX, 10% MCAR | 11.430 +/- 0.064 | 8.011 +/- 0.034 | 0.9065 |
| SA-DA-ARX, 20% MCAR | 11.907 +/- 0.087 | 8.344 +/- 0.047 | 0.8992 |
| SA-DA-ARX, 30% MCAR | 12.383 +/- 0.095 | 8.669 +/- 0.054 | 0.8914 |
| Non-augmented DA-ARX, 30% MCAR | 19.398 +/- 0.309 | 13.605 +/- 0.174 | 0.7323 |

## Scope and limitations

- The data are public benchmark data and are not field data from CHN Energy Bengbu Power Generation Co., Ltd.
- Process variables F02-F27 remain anonymous because the public workbook does not disclose their physical names.
- Sensor missingness is injected in controlled simulations; it is not based on labelled historical sensor faults.
- The model uses the previous ten minutes of measured NOx and is intended for short-term forecasting or temporary auxiliary estimation, not permanent analyser replacement.

## Licence

The source code is released under the MIT License. The third-party dataset remains subject to its original CC BY-NC 3.0 licence. Generated results should retain attribution to the source dataset and associated article.
