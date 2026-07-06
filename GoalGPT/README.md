# ⚽ GoalGPT — Football Prediction Engine

A championship-grade football prediction AI that forecasts match outcomes, scorelines, player scoring probabilities, clean sheets, and over/under goals — powered by Poisson statistics, ELO ratings, and optional ML gradient boosting.

---

## 📋 Project Specifications

- **Application Type**: Hybrid CLI and Flask Web Application.
- **Core AI/ML Methods**: Poisson Distribution (stochastic modeling), Gradient Boosting (optional via `scikit-learn`), Elo Ratings, Bayesian Shrinkage for small sample tournament data.
- **Model Version**: GoalGPT v4/v5 (supports advanced metrics like npxG, SoT rates, manager multipliers, and historical shootout predictive modeling).
- **Tournament Focus**: 2026 FIFA World Cup (incorporating real-time Round of 32 and Round of 16 datasets).
- **Core Stack**: Pure Python (Standard Library) with optional dependencies (`scikit-learn`, `numpy`). Web UI uses HTML/JS/CSS.
- **Data Integration**: Integrates historical CSV data (1872–present) with live tournament metrics, generating comprehensive JSON match predictions.

---

## 📁 Project Structure

```
GoalGPT/
├── sports_prediction_colab_app.py   # Main prediction engine (CLI & Web App)
├── test_engine.py                   # Testing framework for predictions
├── update_datasets.py               # Utility to manage/update dataset files
├── goalgpt_v5.pkl                   # Saved/trained model (latest version)
├── experiment_log.json              # Training metadata log (auto-generated)
├── static/                          # Web app static assets (CSS, JS)
├── templates/                       # Web app HTML templates
└── DataSet/
    ├── results.csv                       # Full international match history
    ├── goalscorers.csv                   # International goal scorers history
    ├── players.csv                       # Current active squad rosters (source of truth)
    ├── players_data_light-2024_2025.csv  # Club season xG / minutes (2024–25)
    ├── players_data-2024_2025.csv        # Full club season stats (2024–25)
    ├── fifa_ranking-*.csv                # FIFA rankings files
    ├── shootouts.csv                     # Penalty shootout records
    ├── Manager_dataset.csv               # Manager performance metrics
    ├── former_names.csv                  # Historical team name aliases
    ├── Worldcup_2026_teams.csv           # World Cup participating teams & ratings
    ├── Worldcup_2026_squads_and_players.csv # Squad list of WC players & profiles
    ├── Worldcup_2026_matches_until_now.csv # Completed tournament match logs
    ├── Worldcup_2026_round_of_32.csv     # Round of 32 matches
    └── Worldcup_2026_round_of_16.csv     # Round of 16 matches
```

---

## 🚀 How to Run

### Prerequisites

- Python 3.8+
- No external dependencies required (pure standard library)
- Optional (for ML-enhanced winner probabilities): `scikit-learn`, `numpy`

```bash
pip install scikit-learn numpy   # optional
```

---

### 1. Train & Save the Model

Trains on all datasets and saves the model to a `.pkl` file for fast future inference.

```bash
python3 sports_prediction_colab_app.py --save-model goalgpt_v5.pkl
```

> This also writes an `experiment_log.json` with training metadata.

---

### 2. Run a Prediction (from saved model)

```bash
python3 sports_prediction_colab_app.py --input '{"home_team": "Argentina", "away_team": "Brazil"}'
```

> If `goalgpt_v5.pkl` exists in the working directory, it is **auto-loaded** — no `--load-model` flag needed.

---

### 3. Explicitly Load a Saved Model

```bash
python3 sports_prediction_colab_app.py --load-model goalgpt_v5.pkl --input '{"home_team": "France", "away_team": "England"}'
```

---

### 4. Run the Web Interface (Flask)

Start the integrated Flask web interface to use GoalGPT from your browser:

```bash
python3 sports_prediction_colab_app.py --web
```
*(Or use the alias: `--serve`)*

The web application will automatically start on `http://localhost:5000`. You can change the port with `--port PORT`.

---

### 5. Programmatic Python Usage

How you load and run the model depends on how you are using the code in Google Colab or Jupyter Notebooks:

#### Option A: If you pasted the entire script into a notebook cell
If you already pasted the code and executed the cell, the model classes are already loaded in memory. You do **not** need to import anything. Just load the pickle file directly:

```python
import sys
import pickle

# Alias the current module namespace so pickle can find the pasted classes
sys.modules['sports_prediction_colab_app'] = sys.modules['__main__']

# Load the saved model file directly
with open("goalgpt_v5.pkl", "rb") as f:
    model = pickle.load(f)

# Run a prediction using a dictionary
result = model.predict({"home_team": "Spain", "away_team": "Japan"})
print(result["output"]["score_prediction"])
```

#### Option B: If you uploaded `sports_prediction_colab_app.py` as a file
If you uploaded the file to Colab's file explorer, import the class first before loading the pickle:

```python
import sys
import pickle
# Ensure current directory is in Python's search path
sys.path.append(".")

from sports_prediction_colab_app import FootballPredictionModel

# Load the saved model file
with open("goalgpt_v5.pkl", "rb") as f:
    model = pickle.load(f)

# Run a prediction using a dictionary
result = model.predict({"home_team": "Spain", "away_team": "Japan"})
print(result["output"]["score_prediction"])
```

---

### 6. Train on the Fly (without saving)

```bash
python3 sports_prediction_colab_app.py --input '{"home_team": "Spain", "away_team": "Germany"}'
```

> If no pre-trained model file exists, the engine trains fresh from the CSVs before predicting.

---

### 7. Custom Dataset Paths

```bash
python3 sports_prediction_colab_app.py \
  --results         path/to/results.csv \
  --goalscorers     path/to/goalscorers.csv \
  --players         path/to/players_data_light-2024_2025.csv \
  --current-players path/to/players.csv \
  --manager-data    path/to/Manager_dataset.csv \
  --wc-teams        path/to/Worldcup_2026_teams.csv \
  --wc-squads       path/to/Worldcup_2026_squads_and_players.csv \
  --wc-matches      path/to/Worldcup_2026_matches_until_now.csv \
  --wc-round-of-32  path/to/Worldcup_2026_round_of_32.csv \
  --wc-round-of-16  path/to/Worldcup_2026_round_of_16.csv \
  --save-model      my_model.pkl
```

---

### CLI Arguments Reference

| Argument | Description | Default |
|---|---|---|
| `--input` | JSON string with `home_team` and `away_team` | Argentina vs Brazil |
| `--results` | Path to match results CSV | `DataSet/results.csv` |
| `--goalscorers` | Path to goal scorers CSV | `DataSet/goalscorers.csv` |
| `--players` | Path to club season stats CSV (xG source) | `DataSet/players_data_light-2024_2025.csv` |
| `--current-players` | Path to current squad rosters CSV | `DataSet/players.csv` |
| `--manager-data` | Path to Manager metrics dataset | `DataSet/Manager_dataset.csv` |
| `--wc-teams` | Path to World Cup 2026 teams dataset | `DataSet/Worldcup_2026_teams.csv` |
| `--wc-squads` | Path to World Cup 2026 squads dataset | `DataSet/Worldcup_2026_squads_and_players.csv` |
| `--wc-matches` | Path to World Cup 2026 match results dataset | `DataSet/Worldcup_2026_matches_until_now.csv` |
| `--wc-round-of-32` | Path to World Cup 2026 R32 match dataset | `DataSet/Worldcup_2026_round_of_32.csv` |
| `--wc-round-of-16` | Path to World Cup 2026 R16 match dataset | `DataSet/Worldcup_2026_round_of_16.csv` |
| `--save-model FILE` | Train and save model to `.pkl` file | — |
| `--load-model FILE` | Load a pre-trained `.pkl` model | — |
| `--web` / `--serve` | Start the web application server | — |
| `--port PORT` | Port number for the web interface | `5000` |

---

## 📊 Prediction Output Format

```jsonc
{
  "output": {
    "match_prediction": {
      "home_team": "Argentina",
      "away_team": "Brazil",
      "winner_probabilities": { "home": 0.36, "draw": 0.24, "away": 0.40 },
      "expected_goals": { "home": 1.55, "away": 1.63 },
      "both_teams_to_score_probability": 0.63,
      "first_team_to_score_probabilities": { "home": 0.47, "away": 0.49, "none": 0.04 },
      "clean_sheet_probabilities": {
        "home": { "probability": 0.20, "goalkeeper": "Emiliano Martínez" },
        "away": { "probability": 0.21, "goalkeeper": "Alisson" }
      },
      "total_goals_over_under": {
        "1.5": { "over": 0.83, "under": 0.17 },
        "2.5": { "over": 0.62, "under": 0.38 },
        "3.5": { "over": 0.39, "under": 0.61 }
      }
    },
    "score_prediction": {
      // predicted_scoreline is always the #1 most likely scoreline
      "predicted_scoreline": "1-1",
      // Only the TOP 2 most probable scorelines are returned
      "scoreline_probabilities": [
        { "score": "1-1", "probability": 0.1051 },
        { "score": "1-2", "probability": 0.0858 }
      ]
    },
    // player_prediction count mirrors the predicted scoreline:
    // "1-1" → 1 home scorer, 1 away scorer (most likely to score)
    // "2-0" → 2 home scorers, 0 away scorers
    // "0-0" → empty lists
    "player_prediction": {
      "home_scorers": [
        {
          "player": "Lionel Messi",
          "goals": 73,
          "probability": 0.0556,
          "prob_1_goal": 0.0525,
          "prob_2_or_more": 0.0015
        }
        // ...N entries = home goals in predicted scoreline
      ],
      "away_scorers": [
        // ...N entries = away goals in predicted scoreline
      ]
    }
  }
}
```

---

## 🧠 Methods Used for Each Prediction

### 1. 🏆 Winner Probabilities (`winner_probabilities`)

| Layer | Method | Weight |
|---|---|---|
| Primary | **Poisson Convolution** — simulate all scorelines (0–0 to 7–7) and sum win/draw/loss probabilities | 32% (100% if no ML) |
| Boosted | **Gradient Boosting Classifier** (`sklearn`) — trained on ELO ratings + attack/defence averages | 68% (if scikit-learn installed) |

**Features used in ML model:** Home ELO, Away ELO, home goals-for/against per match, away goals-for/against per match.

---

**Method: Blended Multi-Factor Model with World Cup Priority**

Expected Goals (xG) is computed by combining long-term historical stats, active squad ratings, pre-tournament Elo, manager metrics, and live tournament results:

1. **Historical/Squad Attacking ($Att_{hist}$) & Defending ($Def_{hist}$)**:
   $$Att_{hist} = 0.40 \times (\text{Historical Goals Scored}) + 0.35 \times (\text{Form Goals Scored}) + 0.25 \times (\text{Active Squad Player Attack Rating})$$
   $$Def_{hist} = 0.40 \times (\text{Historical Goals Conceded}) + 0.35 \times (\text{Form Goals Conceded}) + 0.25 \times (\text{Active Squad Player Def Rating})$$

2. **World Cup 2026 Blend (70% current / 30% historical)**:
   If a team has active matches in the World Cup 2026 dataset, its ratings are blended:
   $$Att_{blended} = 0.70 \times Att_{tourney\_2026} + 0.30 \times Att_{hist}$$
   $$Def_{blended} = 0.70 \times Def_{tourney\_2026} + 0.30 \times Def_{hist}$$
   * *Tournament Stats* include actual goals scored/conceded and expected goals (xG) from World Cup 2026 matches.

3. **Goalkeeper Performance Adjustment (GK Clean Sheet & Defense Boost)**:
   Goalkeepers winning Goalkeeper of the Match (GK PotM) awards scale clean sheet probabilities and strengthen the team's defensive rating (reducing conceded xG):
   $$GK\_Factor = 1.0 + 0.10 \times Count_{GK\_PotM} \quad (\text{capped at } 1.35)$$
   $$CleanSheet\_P = CleanSheet\_P_{base} \times GK\_Factor$$
   $$Def_{blended} = \frac{Def_{blended}}{GK\_Factor}$$

4. **Base xG Blend**:
   $$\text{home\_xg} = 0.50 \times Att_{home} + 0.40 \times Def_{away} + 0.10 \times \text{Global\_Avg}$$
   $$\text{away\_xg} = 0.50 \times Att_{away} + 0.40 \times Def_{home} + 0.10 \times \text{Global\_Avg}$$

5. **Sigmoid ELO Scaling**:
   Adjusts expected goals via a sigmoid multiplier capped between `0.7` and `1.3` using pre-tournament / live ELO ratings:
   $$\text{ELO\_Mult}_{home} = 0.7 + \frac{0.6}{1.0 + e^{-(\Delta ELO / 400.0)}}$$

6. **Head-to-Head Blend (10% weight)**:
   Blends the base xG with the H2H average goals (90% base xG + 10% H2H avg).

7. **Manager Performance (15% weight)**:
   Adjusts expected goals using the manager's offensive rating and the opponent's defensive rating.

8. **xG Clamp**:
   Clamps final xG to the realistic range of `[0.15, 3.5]` to avoid extreme predictions.

---

### 3. 📋 Scoreline Probabilities (`score_prediction`)

**Method: Two-Stage Stochastic Bivariate Poisson**

To ensure realistic, non-deterministic scorelines that favor the stronger team while leaving a realistic chance of upsets:
1. The match outcome (Win / Draw / Loss) is drawn stochastically from the simulated Poisson probabilities.
2. The scoreline is sampled stochastically from the corresponding outcome category.
3. Scorelines with probability $< 1.5\%$ are excluded to avoid highly improbable scoreline combinations.

---

### 4. 👤 Player Scoring Probabilities (`player_prediction`)

**Method: Squad-Restricted Poisson Rates & Player of the Match Boosts**

1. **Squad Filter**: Players are strictly filtered to the official active squads participating in the World Cup (`Worldcup_2026_squads_and_players.csv`).
2. **Player of the Match (PotM) Boost**: Goalscoring and assist rates are scaled using tournament PotM awards:
   $$Multiplier = 1.0 + 0.15 \times Count_{PotM}$$
   $$P_{goal} = \min(0.99, P_{base} \times Multiplier)$$



### 4. 🛡️ Clean Sheet Probabilities (`clean_sheet_probabilities`)

**Method: Poisson Zero-Goal Probability**

```
P(clean sheet) = P(opponent scores 0 goals) = e^(−opponent_xG)
```

- **Goalkeeper** is resolved from `DataSet/players.csv` — the GK with the most `gk_minutes` for that national team.
- Falls back to club-level minutes from `players_data_light-2024_2025.csv` if the team is not in the current roster file.

---

### 5. 🎯 Both Teams to Score (`both_teams_to_score_probability`)

**Method: Poisson Joint Probability**

```
P(BTTS) = sum of P(h-a) for all scores where h > 0 AND a > 0
```

Derived directly from the same Poisson score distribution used in scoreline predictions.

---

### 6. 🚦 First Team to Score (`first_team_to_score_probabilities`)

**Method: Proportional xG Split**

```
P(home scores first) = (home_xG / total_xG) × P(at least one goal)
P(away scores first) = (away_xG / total_xG) × P(at least one goal)
P(no goal)           = P(0-0)
```

---

### 7. 📈 Over/Under Goals (`total_goals_over_under`)

**Method: Poisson CDF Convolution**

For thresholds 1.5, 2.5, and 3.5:

```
P(under N.5) = sum of P(h-a) for all scores where h + a <= floor(N.5)
P(over N.5)  = 1 − P(under N.5)
```

Both teams' goal distributions are convolved independently using the Poisson PMF.

---

### 8. 👤 Player Scoring Probabilities (`player_prediction`)

**Method: Roster-Filtered Poisson Player Model**

Only players listed in `DataSet/players.csv` (current active squad) are included. Retired or historical-only players are **excluded**.

For each active player, a match-level scoring rate `λ_match` is calculated:

```
λ_player = player_xG_season / team_matches          # per-match club xG rate
λ_match  = λ_player × (match_xG / team_avg_goals)  # scaled to this specific match

P(anytime scorer) = 1 − e^(−λ_match)
P(exactly 1 goal) = λ_match × e^(−λ_match)
P(2+ goals)       = 1 − P(0) − P(1)
```

**Data sources ranked by priority:**
1. `players_data_light-2024_2025.csv` → Club season xG (primary signal)
2. `DataSet/goalscorers.csv` → Historical international goals (tiebreaker)
3. `DataSet/players.csv` → Current tournament goals (bonus signal)

Players with no xG data fall back to a goals-per-match rate derived from international history.

Results are **sorted by `probability` descending** — highest likelihood scorers appear first.

**Scoreline-linked count rule:**

The number of scorers returned per team is determined directly by the `predicted_scoreline`:

| Predicted Scoreline | `home_scorers` returned | `away_scorers` returned |
|---|---|---|
| `1-1` | 1 (most likely home scorer) | 1 (most likely away scorer) |
| `2-0` | 2 (top 2 home) | 0 (empty) |
| `3-2` | 3 (top 3 home) | 2 (top 2 away) |
| `0-0` | 0 (empty) | 0 (empty) |

---

## 🗂️ Data Sources

| File | Description | Used For |
|---|---|---|
| `DataSet/results.csv` | Full international match results (1872–present) | ELO training, team stats |
| `DataSet/goalscorers.csv` | Historical international goal scorers | Player historical goals |
| `DataSet/players.csv` | **Current active squad rosters** | Player filtering, GK selection |
| `DataSet/players_data_light-2024_2025.csv` | Club season xG & minutes (2024–25) | Player xG scoring rates |
| `DataSet/Manager_dataset.csv` | Manager performance metrics | Adjusting Team attack & defense based on managers |
| `DataSet/Worldcup_2026_teams.csv` | Pre-tournament FIFA rankings, Elo, and managers | Primary source for participating teams |
| `DataSet/Worldcup_2026_squads_and_players.csv` | Official squads, positions, goals, and awards | Restricting active players, GK and PotM lookup |
| `DataSet/Worldcup_2026_matches_until_now.csv` | Live match logs, goals, and player stats | Blending tournament performance & awards |
| `DataSet/Worldcup_2026_round_of_32.csv` | World Cup 2026 Round of 32 Matches | Updating tournament form |
| `DataSet/Worldcup_2026_round_of_16.csv` | World Cup 2026 Round of 16 Matches | Updating tournament form |

---

## 🔬 Experiment Tracking

Every `--save-model` run automatically writes `experiment_log.json`:

```json
{
  "timestamp": "2026-06-25T10:05:39Z",
  "training_matches": 46742,
  "teams_tracked": 312,
  "elo_k_factor": 32,
  "xg_blend": { "attack": 0.56, "defence": 0.34, "average": 0.10 },
  "h2h_blend": { "statistical": 0.82, "h2h": 0.18 },
  "ml_blend": { "ml": 0.68, "poisson": 0.32 },
  "ml_enabled": false,
  "training_seconds": 4.12,
  "model_file": "goalgpt_v5.pkl"
}
```

---

## ⚡ Performance

| Mode | Typical Time |
|---|---|
| First run (train + predict) | ~4–6 seconds |
| Load from `goalgpt_v5.pkl` + predict | < 1 second |

---

## ❌ Error Handling

If a team name is not recognised, the engine returns structured suggestions:

```json
{
  "error": "Prediction failed",
  "validation_errors": [
    {
      "field": "home_team",
      "value": "Argentinaa",
      "message": "Team 'Argentinaa' not found.",
      "suggestions": ["Argentina"]
    }
  ]
}
```
