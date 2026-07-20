# ⚽ GoalGPT Engine: Complete Technical Architecture & Reference Manual

<div align="center">

![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?style=for-the-badge&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/Engine_Status-v8_Production-00C853?style=for-the-badge)

<br />

**A Championship-Grade Football Prediction Engine Powered by Calibrated Expected Goals ($xG$), Bivariate Poisson Matrices, ELO Dynamics, & Automatic Knockout Stage Inference.**

[Key Features](#-project-introduction) • [System Architecture](#-complete-system-architecture) • [Algorithms & Math](#-core-algorithms--math-v8) • [Validation Suite](#-the-11-test-diagnostic-suite) • [Quickstart Walkthrough](#-end-to-end-walkthrough)

</div>

---

Welcome to the official, presentation-ready technical documentation for the **GoalGPT Engine (v8 Production)**. This manual explains everything happening under the hood—from raw dataset ingestion and Bayesian shrinkage to the final stochastic scorelines shown on the web dashboard.

---

## 📋 Table of Contents
1. [Project Introduction](#1--project-introduction)
2. [Complete System Architecture](#2--complete-system-architecture)
3. [Datasets & Sources](#3--datasets--sources)
4. [Core Algorithms & Math (v8)](#4--core-algorithms--math-v8)
5. [Dynamic Stage Inference & Knockouts](#5--dynamic-stage-inference--knockouts)
6. [Model Serialization & Environment Shims](#6--model-serialization--environment-shims)
7. [The 11-Test Diagnostic Suite](#7--the-11-test-diagnostic-suite)
8. [End-to-End Walkthrough](#8--end-to-end-walkthrough)

---

## 1. 🚀 Project Introduction

GoalGPT is a predictive sports analytics engine built to forecast association football outcomes, specifically tailored for the **FIFA World Cup 2026** and global international fixtures.

Traditional models suffer from two primary failure modes:
1. **Oversimplified Machine Learning**: Standard classification trees and gradient boosters ignore the discrete physics and low-scoring Poisson distributions inherent to football matches.
2. **Rigid Historical Lookups**: Purely historical Poisson models fail to adapt to tournament form, small-sample variance, and tactical manager shifts.

GoalGPT v8 solves this with a **Hybrid Analytical-Stochastic Engine** centered around a calibrated **Bivariate Poisson probability grid**, **Bayesian shrinkage** on small-sample World Cup performances, exponential/linear time-decay weighting, and **automated knockout stage inference** to deliver championship-grade predictions.

---

## 2. 🏛️ Complete System Architecture

GoalGPT is organized into clean, highly modular layers within a zero-dependency architecture designed for portability across local CLI environments, Google Colab notebooks, and production web servers:

```
GoalGPT/
├── sports_prediction_colab_app.py   # Core prediction engine, CLI, & Flask Web API
├── test_engine.py                   # 11-Test Diagnostic Suite
├── update_datasets.py               # Data ingestion & normalization utility
├── GoalGPT_latest.pkl               # Production model symlink -> goalgpt_v8.pkl
├── goalgpt_v8.pkl                   # Latest trained & serialized model artifact (Version 8)
├── experiment_log.json              # Hyperparameter tuning & CV evaluation log
├── Output.json                      # Reference schema for API output
├── templates/                       # Web UI HTML templates
├── static/                          # Web UI styling & interactive scripts
└── DataSet/                         # Curated CSV datasets repository (16 files)
```

### Architectural Flow & Separation of Concerns

```
       +-------------------------------------------------------------+
       |                  Client Interfaces                          |
       |     [ CLI Terminal ]   [ Flask REST API ]   [ Colab/Jupyter ] |
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |                  PredictionService                          |
       |  • Fuzzy Alias Resolution (e.g., 'United States' -> 'USA')  |
       |  • Input Validation & Error Formatting                      |
       |  • JSON Schema Serialization Compliant with Output.json     |
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |                FootballPredictionModel                      |
       |  • Data Ingestion Layer (16 CSV datasets)                   |
       |  • 5-Fold Cross-Validation Hyperparameter Tuning            |
       |  • Analytical Expected Goals (xG) Calibration Engine         |
       |  • Bivariate Poisson Scoreline Distribution & Grid Mass     |
       |  • Automatic Stage Inference & Penalty Shootout Logic       |
       |  • Player Goalscorer & Goalkeeper Clean-Sheet Modeling      |
       +-------------------------------------------------------------+
```

---

## 3. 📊 Datasets & Sources

GoalGPT processes **16 carefully curated CSV datasets**. *Developers must ensure these datasets are updated via `python3 update_datasets.py` before running model retraining.*

| Ingestion Priority | Dataset File | Primary Purpose | Source / Directory |
| :---: | :--- | :--- | :--- |
| **1** | `Worldcup_2026_matches_until_now.csv` | Live tournament match logs (Group stage) | `DataSet/` |
| **2** | `Worldcup_2026_round_of_32.csv` | Live Round of 32 results & stage pairings | `DataSet/` |
| **3** | `Worldcup_2026_squads_and_players.csv` | Official rosters, player profiles, & positions | `DataSet/` |
| **4** | `Manager_dataset.csv` | Tactical performance ratings & tenure of national managers | `DataSet/` |
| **5** | `players_data-2024_2025.csv` | Comprehensive club season player statistics (2024–25) | `DataSet/` |
| **6** | `shootouts.csv` | Penalty shootout win/loss records & historical conversion rates | `DataSet/` |
| **7** | `former_names.csv` | Mappings for historical country name aliases (`USA` $\leftrightarrow$ `United States`) | `DataSet/` |
| **8** | `results.csv` | Full international match records (1872 – Present) | `DataSet/` |
| **9** | `goalscorers.csv` | Historical international goalscorers registry | `DataSet/` |
| **10** | `players.csv` | Active international squad rosters (Source of Truth) | `DataSet/` |
| **11** | `players_data_light-2024_2025.csv` | Club season light metrics ($xG$, minutes played 2024–25) | `DataSet/` |
| **12** | `fifa_ranking-2024-06-20.csv` | Official FIFA international ranking points | `DataSet/` |
| **13** | `Worldcup_2026_teams.csv` | Participating World Cup 2026 teams & base ELO ratings | `DataSet/` |
| **14** | `Output.json` | Reference verification schema for API validation | Root Directory |
| **15** | `experiment_log.json` | Cross-validation history and grid search logs | Root Directory |
| **16** | `GoalGPT_latest.pkl` | Binary serialized production model weight state | Root Directory |

---

## 4. 🧠 Core Algorithms & Math (v8)

GoalGPT v8 prioritizes analytical, mathematically consistent probability logic ($ML\_WEIGHT = 0.00$, $\text{Analytical} = 1.00$) over tree-based models to avoid small-sample volatility during tournament knockout stages.

### Expected Goals ($xG$) Calculation
For any match between Home ($H$) and Away ($A$), raw expected goals ($xG_H, xG_A$) are calculated across multiple distinct mathematical steps:

#### 1. Bayesian Shrinkage on World Cup Performance
Small-sample tournament data ($m_{wc} > 0$) can be volatile. GoalGPT blends recorded tournament goals ($40\%$) with tournament $xG$ ($60\%$):
$$\text{Raw}_{WC} = 0.60 \cdot xG_{for, wc} + 0.40 \cdot \text{Goals}_{for, wc}$$
This raw metric is then shrunken toward historical all-time averages using prior weight $M = \text{PRIOR\_M}$:
$$xG_{wc} = \frac{m_{wc} \cdot \text{Raw}_{WC} + M \cdot \text{HistAvg}}{m_{wc} + M}$$

#### 2. Recent International Form with Time Decay
Recent matches ($i \in [0, 9]$, where $0$ is the most recent) are weighted with linear decay ($w_i = 1.0 - 0.1 \cdot i$). To prioritize high-stakes intensity, matches played in the `FIFA World Cup` receive a **$2.0\times$ weight multiplier**:
$$W_{total} = \sum_{i=0}^9 \left(w_i \cdot \text{Mult}_{wc, i}\right), \quad xG_{form} = \frac{\sum_{i=0}^9 \left(w_i \cdot \text{Mult}_{wc, i} \cdot \text{Goals}_i\right)}{W_{total}}$$

#### 3. ELO Differential & Manager Delta Adjustments
Base expected goals are dynamically scaled using logistic ELO differential ($\Delta ELO = ELO_H - ELO_A$), FIFA ranking deltas, and national team manager impact coefficients ($\Delta_{mgr}$):
$$xG_{calibrated} = xG_{form} \cdot \left(\frac{1}{1 + 10^{-\Delta ELO / 400}}\right) \cdot \left(1 + \text{coef}_{mgr} \cdot \Delta_{mgr}\right) + \text{HomeAdvantage}$$

---

### Bivariate Poisson Scoreline Distribution
Once calibrated $xG_H$ and $xG_A$ are determined, the engine constructs a dynamic score grid up to $N = \max(9, \lceil\max(xG_H, xG_A) + 8\rceil)$:
$$P(h, a) = \left(\frac{e^{-xG_H} xG_H^h}{h!}\right) \left(\frac{e^{-xG_A} xG_A^a}{a!}\right)$$

The joint probability matrix is strictly normalized across the grid so that exact probability conservation holds:
$$\sum_{h=0}^N \sum_{a=0}^N P(h, a) = 1.000000$$

---

## 5. 🏆 Dynamic Stage Inference & Knockouts

GoalGPT v8 eliminates fragile client-side stage inputs (`Group Stage` vs. `Round of 32` vs. `Quarterfinals`). The engine inspects tournament match datasets (`Worldcup_2026_round_of_32.csv`, etc.) and **automatically infers** whether a fixture is a Group Stage match or a Knockout Stage match based on team pairings.

### Knockout Progression & Shootout Probability
If an inferred Knockout match ends in a regulation or extra-time draw ($h = a$), historical penalty shootout statistics from `shootouts.csv` combined with ELO differentials compute the advanced winner:
```json
{
   "penalty_shootout": {
      "show_shootout": true,
      "predicted_winner": "Argentina",
      "win_probability": 58,
      "historical_record": { "wins": 5, "total": 6 }
   }
}
```

---

## 6. 🛠️ Model Serialization & Environment Shims

To ensure seamless, error-free execution when loading serialized models between Google Colab (`scikit-learn 1.6.x`) and local production servers (`scikit-learn 1.9.x`), GoalGPT dynamically applies comprehensive warning filters and class alias namespaces to prevent unpickling crashes:

```python
import warnings
import sys
import pickle

# Suppress scikit-learn unpickling and version mismatch warnings
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", module=".*sklearn.*")

# Fix for scikit-learn loss function module restructuring across minor versions
try:
    import sklearn._loss._loss
    sys.modules['_loss'] = sys.modules.get('_loss', sys.modules['sklearn._loss._loss'])
except ImportError:
    pass

# Safely load the version 8 model artifact
with open("GoalGPT_latest.pkl", "rb") as f:
    model = pickle.load(f)
```

---

## 7. 🧪 The 11-Test Diagnostic Suite

Every GoalGPT release is rigorously verified against `test_engine.py`. This production suite validates mathematical conservation, probabilistic calibration, and schema stability across **11 comprehensive diagnostic tests**:

| Test ID | Test Name | Target Metric / Evaluation Requirement | v8 Status |
| :---: | :--- | :--- | :---: |
| **TEST 1** | **Matrix Normalization** | Joint Poisson probability grid $\sum P(h, a)$ must equal exactly `1.000000`. | ✅ **PASSED** |
| **TEST 2** | **Marginals Alignment** | Matrix marginal sums ($P(H > A), P(H = A), P(H < A)$) must align with headline win probabilities. | ✅ **PASSED** |
| **TEST 3** | **Scoreline Consistency** | Predicted exact scoreline ($h - a$) must match the highest-probability outcome category. | ✅ **PASSED** |
| **TEST 4** | **Expected Goals Derivation** | Matrix-derived expected goals $\sum(h \cdot P(h,a))$ must approximate analytical $xG$ inputs. | ✅ **PASSED** |
| **TEST 5** | **Dynamic Matrix Mass** | Grid truncation check ($0–10+$ goals) must retain $> 99.99\%$ of total Poisson mass per marginal. | ✅ **PASSED** |
| **TEST 6** | **Historical Calibration** | Evaluates Log Loss ($< 1.10$) and Brier Score ($< 0.60$) on historical test validation splits. | ✅ **PASSED** |
| **TEST 7** | **Repeatability & Determinism** | 100% deterministic output consistency under identical seeds across multiple executions. | ✅ **PASSED** |
| **TEST 8** | **Monte Carlo Convergence** | $10,000$ simulation iterations must match analytical matrix probabilities within $\pm 1.5\%$ tolerance. | ✅ **PASSED** |
| **TEST 9** | **Automatic Stage Inference** | Correctly detects and infers `Group Stage` vs. `Round of 32` knockout pairings without user flags. | ✅ **PASSED** |
| **TEST 10** | **Output Schema Compatibility** | Ensures JSON keys strictly match the specification defined in `Output.json`. | ✅ **PASSED** |
| **TEST 11** | **Fuzzy Alias Resolution** | Validates alias mappings (`'United States'` $\rightarrow$ `'USA'`) and rejects nonexistent team inputs (`'Atlantis FC'`). | ✅ **PASSED** |

---

## 8. 🎯 End-to-End Walkthrough

Get started with GoalGPT in under 60 seconds across Python scripts, CLI, or the interactive Flask web dashboard.

### 🐍 Option A: Python API / Colab Notebooks
```python
from sports_prediction_colab_app import FootballPredictionModel, PredictionService
import pickle

# Load production model
with open("GoalGPT_latest.pkl", "rb") as f:
    model = pickle.load(f)

# Instantiate prediction service
service = PredictionService(model)

# Generate detailed prediction
result = service.generate_prediction("Argentina", "Brazil")
print(result["output"]["match_prediction"])
print(result["output"]["score_prediction"])
```

### 💻 Option B: Terminal CLI Execution
```bash
# Run a single prediction via JSON input string
python3 sports_prediction_colab_app.py --input '{"home_team": "Spain", "away_team": "Germany"}'

# Retrain the engine across all 16 datasets and save version 9
python3 sports_prediction_colab_app.py --save-model goalgpt_v9.pkl

# Run diagnostic verification suite
python3 test_engine.py
```

### 🌐 Option C: Flask Web Dashboard & REST API
```bash
# Launch local web application on http://127.0.0.1:5000
python3 sports_prediction_colab_app.py --web --port 5000
```

#### REST API Example (`POST /api/predict`)
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"home_team": "Portugal", "away_team": "France"}'
```

---

<div align="center">
  <p><b>Built with ❤️ by the GoalGPT Engineering Team</b></p>
  <p><i>Empowering Data-Driven Football Insights for the 2026 World Cup & Beyond.</i></p>
</div>
