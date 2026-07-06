import argparse
import cloudpickle
import csv
import json
import math
import pickle
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def normalize_name(name: str) -> str:
    """Normalize names to be lowercase and accent-insensitive."""
    name = name.strip().lower()
    return "".join(c for c in unicodedata.normalize('NFD', name)
                  if unicodedata.category(c) != 'Mn')


class _EloDict(dict):
    """dict subclass with a default of 1500.0 – fully picklable."""
    def __missing__(self, key):
        self[key] = 1500.0
        return 1500.0

# ── Optional ML libraries ────────────────────────────────────────────────────
HAS_ML = False
try:
    # pyrefly: ignore [missing-import]
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    HAS_ML = True
except ImportError:
    pass

# ── Task / capability metadata ───────────────────────────────────────────────
PREDICTION_TASKS = {
    "match_winner":   {"problem_type": "classification",  "preferred_models": ["XGBoost", "CatBoost"]},
    "expected_goals": {"problem_type": "regression",      "preferred_models": ["XGBoost Regressor"]},
    "clean_sheets":   {"problem_type": "classification",  "preferred_models": ["CatBoost"]},
    "scoreline":      {"problem_type": "classification",  "preferred_models": ["Poisson"]},
    "over_under":     {"problem_type": "regression",      "preferred_models": ["Poisson CDF"]},
}


@dataclass
class TeamStats:
    matches: int = 0
    goals_for: int = 0
    goals_against: int = 0
    clean_sheets: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0


# ── Manager Impact ────────────────────────────────────────────────────────────
class ManagerImpact:
    """Read-only loader and rating calculator for Manager_dataset.csv.

    All CSVs are opened in read mode only — no writes, no modifications.
    Missing or unparseable data always returns NEUTRAL (no effect on predictions).
    """

    # Neutral record — rating 1.0 means no change to any prediction
    NEUTRAL = {
        "manager":       None,
        "rating":        1.0,
        "off_mult":      1.0,   # offensive xG multiplier
        "def_mult":      1.0,   # defensive xG multiplier
        "win_pct":       0.5,
        "loss_pct":      0.25,
        "avg_scored":    1.5,
        "avg_conceded":  1.5,
    }

    def __init__(self, path: Path):
        self.records: dict = {}   # normalized_team_name → stats dict
        self._load(path)

    # ── CSV loader ────────────────────────────────────────────────────────────
    def _load(self, path: Path) -> None:
        if not path.exists():
            print(f"[ManagerImpact] Dataset not found: {path} — using neutral defaults.",
                  file=sys.stderr)
            return
        try:
            with open(path, encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        team_raw = row.get("National Team", "").strip()
                        if not team_raw:
                            continue
                        key = normalize_name(team_raw)

                        def _pct(s):
                            return float(s.strip().rstrip("%")) / 100.0 if s.strip() else 0.5

                        def _flt(s, default=1.5):
                            try: return float(s.strip()) if s.strip() else default
                            except ValueError: return default

                        win_pct      = _pct(row.get("Win %", ""))
                        loss_pct     = _pct(row.get("Loss %", ""))
                        avg_scored   = _flt(row.get("Avg Goals Scored Per Game", ""), 1.5)
                        avg_conceded = _flt(row.get("Avg Goals Conceded Per Game", ""), 1.5)

                        # ── Rating formula ────────────────────────────────────
                        # raw: 0-100 scale
                        #   win%×40 + (1−loss%)×20 + (scored/3.0)×25 + (1−conceded/2.5)×15
                        raw = (win_pct * 40.0
                               + (1.0 - loss_pct) * 20.0
                               + min(avg_scored / 3.0, 1.0) * 25.0
                               + max(0.0, (1.0 - avg_conceded / 2.5)) * 15.0)

                        # Normalise to 0.5 – 1.5 (1.0 = neutral)
                        rating = 0.5 + raw / 100.0

                        # Per-xG multipliers: 0.80 – 1.20 range
                        off_mult = 0.80 + min(avg_scored   / 2.5, 1.0) * 0.40
                        def_mult = 0.80 + max(0.0, (2.5 - avg_conceded) / 2.5) * 0.40

                        self.records[key] = {
                            "manager":       row.get("Manager", "").strip(),
                            "rating":        round(rating, 4),
                            "off_mult":      round(off_mult, 4),
                            "def_mult":      round(def_mult, 4),
                            "win_pct":       round(win_pct, 4),
                            "loss_pct":      round(loss_pct, 4),
                            "avg_scored":    avg_scored,
                            "avg_conceded":  avg_conceded,
                        }
                    except Exception:
                        continue
        except Exception as exc:
            print(f"[ManagerImpact] Could not read {path}: {exc}", file=sys.stderr)

    # ── Public API ────────────────────────────────────────────────────────────
    def get(self, team: str) -> dict:
        """Return manager stats for *team*, or NEUTRAL if not found."""
        return self.records.get(normalize_name(team), self.NEUTRAL)

    def rating(self, team: str) -> float:
        return self.get(team)["rating"]

    def off_mult(self, team: str) -> float:
        return self.get(team)["off_mult"]

    def def_mult(self, team: str) -> float:
        return self.get(team)["def_mult"]


class FootballPredictionModel:
    """Championship-grade football prediction engine."""

    # ── Construction ──────────────────────────────────────────────────────────
    def __init__(self, results_path=None, goalscorers_path=None, players_path=None,
                 current_players_path=None, manager_path=None,
                 worldcup_teams_path=None, worldcup_squads_path=None,
                 worldcup_matches_path=None, worldcup_round_of_32_path=None,
                 worldcup_round_of_16_path=None):
        def _auto(name):
            ds = Path("DataSet") / name
            return str(ds) if ds.exists() else name

        self.results_path         = Path(results_path         or _auto("results.csv"))
        self.goalscorers_path     = Path(goalscorers_path     or _auto("goalscorers.csv"))
        self.players_path         = Path(players_path         or _auto("players_data_light-2024_2025.csv"))
        self.current_players_path = Path(current_players_path or _auto("players.csv"))
        self.manager_path         = Path(manager_path         or _auto("Manager_dataset.csv"))
        
        self.worldcup_teams_path       = Path(worldcup_teams_path       or _auto("Worldcup_2026_teams.csv"))
        self.worldcup_squads_path      = Path(worldcup_squads_path      or _auto("Worldcup_2026_squads_and_players.csv"))
        self.worldcup_matches_path     = Path(worldcup_matches_path     or _auto("Worldcup_2026_matches_until_now.csv"))
        self.worldcup_round_of_32_path = Path(worldcup_round_of_32_path or _auto("Worldcup_2026_round_of_32.csv"))
        self.worldcup_round_of_16_path = Path(worldcup_round_of_16_path or _auto("Worldcup_2026_round_of_16.csv"))

        # Manager impact module (read-only; gracefully neutral if file absent)
        self.manager_impact = ManagerImpact(self.manager_path)

        self.matches: list          = []
        self.team_stats             = defaultdict(TeamStats)
        self.elo_ratings            = _EloDict()
        self.h2h_matches            = defaultdict(list)
        self.player_goals           = defaultdict(Counter)
        self.player_xg              = defaultdict(dict)
        self.player_npxg            = defaultdict(dict)
        self.player_sot             = defaultdict(dict)
        self.player_nineties        = defaultdict(dict)
        self.team_goalkeeper        = {}
        self.fallback_goalkeepers   = {}
        self.current_team_players   = defaultdict(dict)
        self.total_goals_scored: int = 0
        self.total_matches: int     = 0

        # World Cup 2026 structures
        self.wc_teams_info          = {}
        self.wc_team_ids            = {}
        self.wc_squads              = defaultdict(set)
        self.wc_player_info         = defaultdict(dict)
        self.wc_team_stats          = defaultdict(dict)
        self.potm_counts            = Counter()
        
        # Configurable Blending & Tuning Hyperparameters
        self.ML_WEIGHT = 0.60
        self.ANALYTICAL_WEIGHT = 0.40
        self.MANAGER_DELTA_COEF = 0.05
        self.PRIOR_M = 5.0
        self.PRIOR_FORM_M = 3.0
        self.DRAW_CALIB_MULT = 1.0
        self.STOCHASTIC_VAR = 0.02

        self.fifa_rankings          = {}
        self.shootout_stats         = defaultdict(lambda: {"wins": 0, "total": 0})
        self.gk_potm_counts         = Counter()
        self.wc_players_appeared    = set()
        self.r32_team_pairs         = frozenset()  # populated by _load_worldcup_round_of_32
        self.r16_team_pairs         = frozenset()

        self.ml_winner_model        = None
        self.ml_goals_model         = None
        self.wc_teams_list          = []

    # ── Data loading ──────────────────────────────────────────────────────────
    def _load_results(self):
        if not self.results_path.exists():
            print(f"Error: {self.results_path} not found.", file=sys.stderr)
            sys.exit(1)
        with open(self.results_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    self.matches.append({
                        "date":       datetime.strptime(row["date"], "%Y-%m-%d"),
                        "home_team":  row["home_team"],
                        "away_team":  row["away_team"],
                        "home_score": int(row["home_score"]),
                        "away_score": int(row["away_score"]),
                    })
                except (ValueError, TypeError, KeyError):
                    continue
        self.matches.sort(key=lambda x: x["date"])

    def _load_current_players(self):
        """Loads active player rosters from players.csv."""
        if not self.current_players_path.exists():
            print(f"Warning: {self.current_players_path} not found.", file=sys.stderr)
            return
        
        with open(self.current_players_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    name = row.get("player", "").strip()
                    team = row.get("team", "").strip()
                    team_country = row.get("team_country", "").strip()
                    pos = row.get("position", "").strip()
                    if not name or not team:
                        continue
                    
                    try:
                        mins = float(row.get("minutes", 0) or 0)
                    except ValueError:
                        mins = 0.0
                    try:
                        gk_mins = float(row.get("gk_minutes", 0) or 0)
                    except ValueError:
                        gk_mins = 0.0
                    try:
                        goals = int(float(row.get("goals", 0) or 0))
                    except ValueError:
                        goals = 0
                    try:
                        assists = int(float(row.get("assists", 0) or 0))
                    except ValueError:
                        assists = 0
                    try:
                        gk_cs = int(float(row.get("gk_clean_sheets", 0) or 0))
                    except ValueError:
                        gk_cs = 0
                    try:
                        gk_games = int(float(row.get("gk_games", 0) or 0))
                    except ValueError:
                        gk_games = 0

                    norm_name = normalize_name(name)
                    player_dict = {
                        "original_name": name,
                        "position": pos,
                        "minutes": mins,
                        "gk_minutes": gk_mins,
                        "goals": goals,
                        "assists": assists,
                        "gk_clean_sheets": gk_cs,
                        "gk_games": gk_games,
                    }
                    self.current_team_players[team][norm_name] = player_dict
                    if team_country and team_country != team:
                        self.current_team_players[team_country][norm_name] = player_dict
                except Exception:
                    continue

        # Resolve starting GK per team based on gk_minutes / minutes from players.csv
        for team, roster in self.current_team_players.items():
            gks = []
            for norm_name, p_info in roster.items():
                if "GK" in p_info["position"]:
                    gk_mins = p_info["gk_minutes"] or p_info["minutes"]
                    gks.append((gk_mins, p_info["original_name"]))
            if gks:
                gks.sort(reverse=True)
                self.team_goalkeeper[team] = gks[0][1]

    def _load_player_stats(self):
        """Load player xG and identify top GK per nation from the players CSV."""
        if not self.players_path.exists():
            return
        # Nation-code → country name mapping (abbreviated)
        # We'll build team → players using the Nation column (e.g. "br BRA" → "Brazil")
        nation_map = {
            "ARG": "Argentina", "BRA": "Brazil",  "FRA": "France",
            "ENG": "England",   "GER": "Germany",  "ESP": "Spain",
            "ITA": "Italy",     "POR": "Portugal", "NED": "Netherlands",
            "BEL": "Belgium",   "URU": "Uruguay",  "COL": "Colombia",
            "MEX": "Mexico",    "USA": "USA",       "SCO": "Scotland",
            "CRO": "Croatia",   "DEN": "Denmark",   "SWE": "Sweden",
            "SEN": "Senegal",   "NGA": "Nigeria",   "GHA": "Ghana",
            "MAR": "Morocco",   "EGY": "Egypt",     "JPN": "Japan",
            "KOR": "South Korea","AUS": "Australia","IRN": "Iran",
            "POL": "Poland",    "CZE": "Czech Republic","HUN": "Hungary",
            "TUR": "Turkey",    "UKR": "Ukraine",   "RUS": "Russia",
            "SRB": "Serbia",    "SVK": "Slovakia",  "AUT": "Austria",
            "SUI": "Switzerland","NOR": "Norway",   "FIN": "Finland",
            "CIV": "Ivory Coast","CMR": "Cameroon","ALG": "Algeria",
            "TUN": "Tunisia",   "ZAM": "Zambia",    "ANG": "Angola",
            "ISL": "Iceland",   "IRL": "Republic of Ireland",
            "WAL": "Wales",     "CAN": "Canada",    "ECU": "Ecuador",
            "CHI": "Chile",     "PAR": "Paraguay",  "BOL": "Bolivia",
            "PER": "Peru",      "VEN": "Venezuela", "HON": "Honduras",
            "CRC": "Costa Rica","PAN": "Panama",    "JAM": "Jamaica",
            "CHN": "China",     "SAU": "Saudi Arabia","QAT": "Qatar",
            "UAE": "United Arab Emirates","IRQ": "Iraq",
        }
        gk_stats = defaultdict(list)  # country → [(minutes, name)]

        with open(self.players_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    nation_raw = row.get("Nation", "")
                    parts = nation_raw.strip().split()
                    code = parts[-1].upper() if parts else ""
                    country = nation_map.get(code)
                    if not country:
                        continue

                    pos  = row.get("Pos", "")
                    name = row.get("Player", "").strip()
                    if not name:
                        continue

                    norm_name = normalize_name(name)

                    try:
                        xg_val = float(row.get("xG", 0) or 0)
                    except ValueError:
                        xg_val = 0.0
                    try:
                        npxg_val = float(row.get("npxG", 0) or 0)
                    except ValueError:
                        npxg_val = 0.0
                    try:
                        sot_val = float(row.get("SoT", 0) or 0)
                    except ValueError:
                        sot_val = 0.0
                    try:
                        nineties_val = float(row.get("90s", 0) or 0)
                    except ValueError:
                        nineties_val = 0.0

                    if xg_val > 0 or npxg_val > 0 or sot_val > 0:
                        self.player_xg[country][norm_name] = self.player_xg[country].get(norm_name, 0.0) + xg_val
                        self.player_npxg[country][norm_name] = self.player_npxg[country].get(norm_name, 0.0) + npxg_val
                        self.player_sot[country][norm_name] = self.player_sot[country].get(norm_name, 0.0) + sot_val
                        self.player_nineties[country][norm_name] = self.player_nineties[country].get(norm_name, 0.0) + nineties_val

                    # Track GK by total minutes played
                    if "GK" in pos:
                        try:
                            mins = float(row.get("Min", 0) or 0)
                        except ValueError:
                            mins = 0.0
                        gk_stats[country].append((mins, name))
                except Exception:
                    continue

        # Pick starting GK (most minutes) per country as fallback
        for country, entries in gk_stats.items():
            entries.sort(reverse=True)
            self.fallback_goalkeepers[country] = entries[0][1]

    def _load_worldcup_teams(self):
        if not self.worldcup_teams_path.exists():
            return
        with open(self.worldcup_teams_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    t_id = row["team_id"].strip()
                    name = row["team_name"].strip()
                    norm_name = normalize_name(name)
                    fifa_rank = int(row["fifa_ranking_pre_tournament"])
                    elo = float(row["elo_rating"])
                    manager = row["manager_name"].strip()
                    
                    self.wc_teams_info[norm_name] = {
                        "team_id": t_id,
                        "team_name": name,
                        "fifa_ranking": fifa_rank,
                        "elo": elo,
                        "manager": manager
                    }
                    self.wc_team_ids[t_id] = norm_name
                    
                    self.elo_ratings[name] = elo
                    self.elo_ratings[norm_name] = elo
                except Exception:
                    continue
        
        # Cache list of participating World Cup teams to support fully standalone prediction
        self.wc_teams_list = sorted(list({
            info["team_name"]
            for info in self.wc_teams_info.values()
            if info.get("team_name")
        }))

    def _load_fifa_rankings(self):
        """Load global FIFA rankings from the 2024 snapshot as a fallback for non-World Cup teams."""
        path = Path("DataSet/fifa_ranking-2024-06-20.csv")
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if row.get("rank_date") == "2024-06-20":
                        rank = int(float(row.get("rank", 100)))
                        country = row.get("country_full", "").strip()
                        if country:
                            self.fifa_rankings[normalize_name(country)] = rank
                except Exception:
                    continue

    def _load_shootouts(self):
        """Load historical penalty shootouts to calculate team shootout stats."""
        path = Path("DataSet/shootouts.csv")
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ht = row.get("home_team", "").strip()
                    at = row.get("away_team", "").strip()
                    w = row.get("winner", "").strip()
                    if not ht or not at or not w:
                        continue
                    hn = normalize_name(ht)
                    an = normalize_name(at)
                    wn = normalize_name(w)
                    for team in (hn, an):
                        self.shootout_stats[team]["total"] += 1
                        if team == wn:
                            self.shootout_stats[team]["wins"] += 1
                except Exception:
                    continue

    def _load_worldcup_squads(self):
        if not self.worldcup_squads_path.exists():
            return
        with open(self.worldcup_squads_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    t_id = row["team_id"].strip()
                    p_name = row["player_name"].strip()
                    pos = row["position"].strip()
                    norm_team = self.wc_team_ids.get(t_id)
                    if not norm_team:
                        continue
                    norm_player = normalize_name(p_name)
                    self.wc_squads[norm_team].add(norm_player)
                    
                    # Store player info
                    try:
                        caps = int(row.get("caps", 0) or 0)
                    except ValueError:
                        caps = 0
                    try:
                        goals = int(row.get("goals", 0) or 0)
                    except ValueError:
                        goals = 0
                        
                    self.wc_player_info[norm_team][norm_player] = {
                        "original_name": p_name,
                        "position": pos,
                        "caps": caps,
                        "goals": goals
                    }
                    
                    # Ensure starting GK fallback is picked from the WC squad if available
                    if "GK" in pos and norm_player not in self.fallback_goalkeepers.get(norm_team, ""):
                        # If no GK has been set yet, or we find one with more caps
                        current_fallback = self.fallback_goalkeepers.get(norm_team)
                        if not current_fallback:
                            self.fallback_goalkeepers[norm_team] = p_name
                        else:
                            # Compare caps
                            curr_norm = normalize_name(current_fallback)
                            curr_info = self.wc_player_info[norm_team].get(curr_norm, {})
                            if caps > curr_info.get("caps", -1):
                                self.fallback_goalkeepers[norm_team] = p_name
                except Exception:
                    continue

    def _load_worldcup_matches(self):
        if not self.worldcup_matches_path.exists():
            return
        with open(self.worldcup_matches_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    status = row["status"].strip()
                    if status != "Completed":
                        continue
                    
                    ht = row["home_team_name"].strip()
                    at = row["away_team_name"].strip()
                    hn = normalize_name(ht)
                    an = normalize_name(at)
                    hs = int(row["home_score"])
                    as_ = int(row["away_score"])
                    h_xg = float(row["home_xg"])
                    a_xg = float(row["away_xg"])
                    h_gk = row["home_goalkeeper"].strip()
                    a_gk = row["away_goalkeeper"].strip()
                    potm = row["player_of_the_match_name"].strip()

                    # Append to training matches
                    try:
                        match_date = datetime.strptime(row["date"], "%Y-%m-%d")
                    except Exception:
                        match_date = datetime.now()
                    self.matches.append({
                        "date":       match_date,
                        "home_team":  ht,
                        "away_team":  at,
                        "home_score": hs,
                        "away_score": as_,
                        "tournament": "FIFA World Cup",
                        "stage":      row.get("stage_name", "Group Stage"),
                    })
                    
                    # Track statistics
                    for team_norm in (hn, an):
                        if team_norm not in self.wc_team_stats:
                            self.wc_team_stats[team_norm] = {
                                "matches": 0,
                                "goals_for": 0.0,
                                "goals_against": 0.0,
                                "clean_sheets": 0,
                                "wins": 0,
                                "draws": 0,
                                "losses": 0,
                                "xg_for": 0.0,
                                "xg_against": 0.0
                            }
                    
                    # Home stats
                    h_stats = self.wc_team_stats[hn]
                    h_stats["matches"] += 1
                    h_stats["goals_for"] += hs
                    h_stats["goals_against"] += as_
                    h_stats["xg_for"] += h_xg
                    h_stats["xg_against"] += a_xg
                    if as_ == 0:
                        h_stats["clean_sheets"] += 1
                    
                    # Away stats
                    a_stats = self.wc_team_stats[an]
                    a_stats["matches"] += 1
                    a_stats["goals_for"] += as_
                    a_stats["goals_against"] += hs
                    a_stats["xg_for"] += a_xg
                    a_stats["xg_against"] += h_xg
                    if hs == 0:
                        a_stats["clean_sheets"] += 1
                        
                    if hs > as_:
                        h_stats["wins"] += 1
                        a_stats["losses"] += 1
                    elif hs < as_:
                        h_stats["losses"] += 1
                        a_stats["wins"] += 1
                    else:
                        h_stats["draws"] += 1
                        a_stats["draws"] += 1
                        
                    # PotM track
                    if potm:
                        self.potm_counts[normalize_name(potm)] += 1
                        
                    # Goalkeeper tracking (if the PotM was one of the goalkeepers)
                    if potm and normalize_name(potm) == normalize_name(h_gk):
                        self.gk_potm_counts[normalize_name(h_gk)] += 1
                    elif potm and normalize_name(potm) == normalize_name(a_gk):
                        self.gk_potm_counts[normalize_name(a_gk)] += 1
                        
                    # Let's keep a record of current goalkeepers
                    if h_gk:
                        self.team_goalkeeper[ht] = h_gk
                        self.team_goalkeeper[hn] = h_gk
                    if a_gk:
                        self.team_goalkeeper[at] = a_gk
                        self.team_goalkeeper[an] = a_gk
                        
                    if hasattr(self, "wc_players_appeared"):
                        if h_gk: self.wc_players_appeared.add(normalize_name(h_gk))
                        if a_gk: self.wc_players_appeared.add(normalize_name(a_gk))
                        if potm: self.wc_players_appeared.add(normalize_name(potm))
                        
                except Exception:
                    continue

    def _load_worldcup_round_of_32(self):
        """Load Worldcup_2026_round_of_32.csv — one row per goal event.

        Schema differs from Worldcup_2026_matches_until_now.csv:
          date, stage, home_team, away_team, home_score, away_score,
          extra_time, penalties, winner, scorer, scorer_team, goal_minute, venue

        Multiple rows share the same match (one per scorer). This loader:
          1. Deduplicates rows into unique match records.
          2. Updates wc_team_stats with goals, clean sheets, results.
          3. Registers scorers in player_goals and wc_players_appeared.
          4. Tags each match as tournament='FIFA World Cup' for 2x recency boost.
        All datasets are read-only — no writes are performed.
        """
        if not self.worldcup_round_of_32_path.exists():
            print(f"[R32] Dataset not found: {self.worldcup_round_of_32_path} — skipping.",
                  file=sys.stderr)
            return

        # ── Pass 1: Aggregate unique matches ─────────────────────────────────
        match_index = {}   # (date_str, ht, at) → match dict
        scorer_rows = []   # deferred scorer processing

        with open(self.worldcup_round_of_32_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    ht  = row["home_team"].strip()
                    at  = row["away_team"].strip()
                    hs  = int(row["home_score"])
                    as_ = int(row["away_score"])
                    dt  = row["date"].strip()
                    key = (dt, ht, at)

                    if key not in match_index:
                        try:
                            match_date = datetime.strptime(dt, "%Y-%m-%d")
                        except Exception:
                            match_date = datetime.now()

                        # Estimate xG from historical averages (xG not in this CSV)
                        avg = self.total_goals_scored / (2 * max(1, self.total_matches))
                        if avg == 0: avg = 1.3

                        match_index[key] = {
                            "date":       match_date,
                            "home_team":  ht,
                            "away_team":  at,
                            "home_score": hs,
                            "away_score": as_,
                            "tournament": "FIFA World Cup",
                            "stage":      "Round of 32",
                            "est_home_xg": avg,  # fallback estimate
                            "est_away_xg": avg,
                        }

                    # Collect scorer events for Pass 2
                    scorer = row.get("scorer", "").strip()
                    scorer_team = row.get("scorer_team", "").strip()
                    if scorer and scorer_team:
                        scorer_rows.append((scorer, scorer_team))

                except Exception:
                    continue

        # ── Pass 2: Update wc_team_stats and self.matches ────────────────────
        for match in match_index.values():
            ht  = match["home_team"]
            at  = match["away_team"]
            hn  = normalize_name(ht)
            an  = normalize_name(at)
            hs  = match["home_score"]
            as_ = match["away_score"]

            # Initialise wc_team_stats bucket if missing
            for team_norm in (hn, an):
                if team_norm not in self.wc_team_stats:
                    self.wc_team_stats[team_norm] = {
                        "matches": 0, "goals_for": 0.0, "goals_against": 0.0,
                        "clean_sheets": 0, "wins": 0, "draws": 0, "losses": 0,
                        "xg_for": 0.0, "xg_against": 0.0,
                    }

            est_xg = match["est_home_xg"]

            # Home
            h_stats = self.wc_team_stats[hn]
            h_stats["matches"]       += 1
            h_stats["goals_for"]     += hs
            h_stats["goals_against"] += as_
            h_stats["xg_for"]        += est_xg
            h_stats["xg_against"]    += est_xg
            if as_ == 0: h_stats["clean_sheets"] += 1

            # Away
            a_stats = self.wc_team_stats[an]
            a_stats["matches"]       += 1
            a_stats["goals_for"]     += as_
            a_stats["goals_against"] += hs
            a_stats["xg_for"]        += est_xg
            a_stats["xg_against"]    += est_xg
            if hs == 0: a_stats["clean_sheets"] += 1

            # Win/draw/loss
            if hs > as_:
                h_stats["wins"]   += 1; a_stats["losses"] += 1
            elif hs < as_:
                h_stats["losses"] += 1; a_stats["wins"]   += 1
            else:
                h_stats["draws"]  += 1; a_stats["draws"]  += 1

            # Append to self.matches so Elo + team stats recalculate correctly
            self.matches.append({
                "date":       match["date"],
                "home_team":  ht,
                "away_team":  at,
                "home_score": hs,
                "away_score": as_,
                "tournament": "FIFA World Cup",
                "stage":      "Round of 32",
            })

        # ── Pass 3: Register scorers ──────────────────────────────────────────
        for scorer, scorer_team in scorer_rows:
            norm_scorer = normalize_name(scorer)
            norm_team   = normalize_name(scorer_team)

            # Find canonical team name from the match index for player_goals key
            canonical_team = scorer_team  # default
            for (_, ht, at), _ in match_index.items():
                if normalize_name(ht) == norm_team:
                    canonical_team = ht; break
                if normalize_name(at) == norm_team:
                    canonical_team = at; break

            self.player_goals[canonical_team][norm_scorer] += 1

            # Mark player as appeared in WC 2026 — makes them eligible for prediction
            self.wc_players_appeared.add(norm_scorer)

        r32_matches = len(match_index)
        scorers_added = len(scorer_rows)
        print(f"[R32] Loaded {r32_matches} Round of 32 matches, "
              f"{scorers_added} goal events registered.", file=sys.stderr)

        # Store fixture team-pairs for automatic knockout detection at prediction time
        self.r32_team_pairs = frozenset(
            (normalize_name(m["home_team"]), normalize_name(m["away_team"]))
            for m in match_index.values()
        )

    def _load_worldcup_round_of_16(self):
        """Load Worldcup_2026_round_of_16.csv — one row per goal event."""
        if not self.worldcup_round_of_16_path.exists():
            print(f"[R16] Dataset not found: {self.worldcup_round_of_16_path} — skipping.",
                  file=sys.stderr)
            return

        match_index = {}
        scorer_rows = []

        with open(self.worldcup_round_of_16_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    ht  = row["home_team"].strip()
                    at  = row["away_team"].strip()
                    hs  = int(row["home_score"])
                    as_ = int(row["away_score"])
                    dt  = row["date"].strip()
                    key = (dt, ht, at)

                    if key not in match_index:
                        try:
                            match_date = datetime.strptime(dt, "%Y-%m-%d")
                        except Exception:
                            match_date = datetime.now()

                        avg = self.total_goals_scored / (2 * max(1, self.total_matches))
                        if avg == 0: avg = 1.3

                        match_index[key] = {
                            "date":       match_date,
                            "home_team":  ht,
                            "away_team":  at,
                            "home_score": hs,
                            "away_score": as_,
                            "tournament": "FIFA World Cup",
                            "stage":      "Round of 16",
                            "est_home_xg": avg,
                            "est_away_xg": avg,
                        }

                    scorer = row.get("scorer", "").strip()
                    scorer_team = row.get("scorer_team", "").strip()
                    if scorer and scorer_team:
                        scorer_rows.append((scorer, scorer_team))

                except Exception:
                    continue

        for match in match_index.values():
            ht  = match["home_team"]
            at  = match["away_team"]
            hn  = normalize_name(ht)
            an  = normalize_name(at)
            hs  = match["home_score"]
            as_ = match["away_score"]

            for team_norm in (hn, an):
                if team_norm not in self.wc_team_stats:
                    self.wc_team_stats[team_norm] = {
                        "matches": 0, "goals_for": 0.0, "goals_against": 0.0,
                        "clean_sheets": 0, "wins": 0, "draws": 0, "losses": 0,
                        "xg_for": 0.0, "xg_against": 0.0,
                    }

            est_xg = match["est_home_xg"]
            h_stats = self.wc_team_stats[hn]
            h_stats["matches"]       += 1
            h_stats["goals_for"]     += hs
            h_stats["goals_against"] += as_
            h_stats["xg_for"]        += est_xg
            h_stats["xg_against"]    += est_xg
            if as_ == 0: h_stats["clean_sheets"] += 1

            a_stats = self.wc_team_stats[an]
            a_stats["matches"]       += 1
            a_stats["goals_for"]     += as_
            a_stats["goals_against"] += hs
            a_stats["xg_for"]        += est_xg
            a_stats["xg_against"]    += est_xg
            if hs == 0: a_stats["clean_sheets"] += 1

            if hs > as_:
                h_stats["wins"]   += 1; a_stats["losses"] += 1
            elif hs < as_:
                h_stats["losses"] += 1; a_stats["wins"]   += 1
            else:
                h_stats["draws"]  += 1; a_stats["draws"]  += 1

            self.matches.append({
                "date":       match["date"],
                "home_team":  ht,
                "away_team":  at,
                "home_score": hs,
                "away_score": as_,
                "tournament": "FIFA World Cup",
                "stage":      "Round of 16",
            })

        for scorer, scorer_team in scorer_rows:
            norm_scorer = normalize_name(scorer)
            norm_team   = normalize_name(scorer_team)

            canonical_team = scorer_team
            for (_, ht, at), _ in match_index.items():
                if normalize_name(ht) == norm_team:
                    canonical_team = ht; break
                if normalize_name(at) == norm_team:
                    canonical_team = at; break

            self.player_goals[canonical_team][norm_scorer] += 1
            self.wc_players_appeared.add(norm_scorer)

        r16_matches = len(match_index)
        scorers_added = len(scorer_rows)
        print(f"[R16] Loaded {r16_matches} Round of 16 matches, "
              f"{scorers_added} goal events registered.", file=sys.stderr)

        self.r16_team_pairs = frozenset(
            (normalize_name(m["home_team"]), normalize_name(m["away_team"]))
            for m in match_index.values()
        )

    def _train_team_models(self):
        K = 32.0
        for match in self.matches:
            ht, at = match["home_team"], match["away_team"]
            hs, as_ = match["home_score"], match["away_score"]

            h = self.team_stats[ht]
            a = self.team_stats[at]
            h.matches += 1;  a.matches += 1
            h.goals_for += hs; h.goals_against += as_
            a.goals_for += as_; a.goals_against += hs
            if hs == 0: a.clean_sheets += 1
            if as_ == 0: h.clean_sheets += 1

            if hs > as_:   h.wins += 1; a.losses += 1; actual = 1.0
            elif hs < as_: h.losses += 1; a.wins += 1; actual = 0.0
            else:          h.draws += 1; a.draws += 1; actual = 0.5

            self.total_goals_scored += hs + as_
            self.total_matches += 1

            h_elo = self.elo_ratings[ht]
            a_elo = self.elo_ratings[at]
            exp_h = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo) / 400.0))
            self.elo_ratings[ht] = h_elo + K * (actual - exp_h)
            self.elo_ratings[at] = a_elo + K * ((1 - actual) - (1 - exp_h))

            key = frozenset([ht, at])
            self.h2h_matches[key].append(match)

        if self.goalscorers_path.exists():
            with open(self.goalscorers_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    team   = row.get("team", "").strip()
                    scorer = row.get("scorer", "").strip()
                    if team and scorer:
                        norm_scorer = normalize_name(scorer)
                        self.player_goals[team][norm_scorer] += 1
        else:
            print(f"Warning: {self.goalscorers_path} not found.", file=sys.stderr)

    def _train_supervised_models(self):
        if not HAS_ML:
            print("No ML libs – using statistical fallback.", file=sys.stderr)
            return
        features, targets = [], []
        for m in self.matches:
            ht, at = m["home_team"], m["away_team"]
            hs = self.team_stats[ht]; as_ = self.team_stats[at]
            features.append([
                self.elo_ratings[ht], self.elo_ratings[at],
                hs.goals_for / max(1, hs.matches), hs.goals_against / max(1, hs.matches),
                as_.goals_for / max(1, as_.matches), as_.goals_against / max(1, as_.matches),
            ])
            if m["home_score"] > m["away_score"]: targets.append(2)
            elif m["home_score"] < m["away_score"]: targets.append(0)
            else: targets.append(1)
        X = np.array(features)
        self.ml_winner_model = GradientBoostingClassifier(random_state=42)
        self.ml_winner_model.fit(X, np.array(targets))

    def calibrate_xg(self):
        """Calculate MAE and RMSE of predicted raw xG vs actual goals for the 90 Group Stage matches."""
        wc_group_matches = [
            m for m in self.matches 
            if m.get("tournament") == "FIFA World Cup" and m.get("stage") == "Group Stage"
        ]
        if not wc_group_matches:
            print("No World Cup Group Stage matches found for calibration.", file=sys.stderr)
            return 0.0, 0.0

        ae_sum = 0.0
        se_sum = 0.0
        n = len(wc_group_matches)
        for m in wc_group_matches:
            ht, at = m["home_team"], m["away_team"]
            hs, as_ = m["home_score"], m["away_score"]
            try:
                h_xg, a_xg = self._expected_goals_raw(ht, at)
            except Exception:
                h_xg, a_xg = 1.3, 1.3
            ae_sum += abs(h_xg - hs) + abs(a_xg - as_)
            se_sum += (h_xg - hs) ** 2 + (a_xg - as_) ** 2

        mae = ae_sum / (2 * n)
        rmse = math.sqrt(se_sum / (2 * n))
        print(f"--- Expected Goals (xG) Calibration Report ({n} Matches) ---", file=sys.stderr)
        print(f"Mean Absolute Error (MAE): {mae:.4f} goals/match", file=sys.stderr)
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f} goals/match", file=sys.stderr)
        print("----------------------------------------------------------------", file=sys.stderr)
        return mae, rmse

    def _optimize_hyperparameters(self):
        """Perform 5-fold cross-validation coordinate descent to find optimal parameters."""
        wc_group_matches = [
            m for m in self.matches 
            if m.get("tournament") == "FIFA World Cup" and m.get("stage") == "Group Stage"
        ]
        if not wc_group_matches:
            return

        # Split into 5 folds
        n = len(wc_group_matches)
        fold_size = n // 5
        folds = []
        for i in range(5):
            start = i * fold_size
            end = (i + 1) * fold_size if i < 4 else n
            val_set = wc_group_matches[start:end]
            train_set = wc_group_matches[:start] + wc_group_matches[end:]
            folds.append((train_set, val_set))

        # Save original tournament stats to restore later
        orig_wc_team_stats = dict(self.wc_team_stats)

        # Candidate parameters
        ml_w_choices = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
        mgr_choices = [0.0, 0.03, 0.05, 0.08, 0.12]
        prior_m_choices = [1.0, 3.0, 5.0, 8.0]
        prior_form_m_choices = [1.0, 2.0, 3.0, 5.0]
        draw_choices = [0.6, 0.8, 1.0, 1.2]
        var_choices = [0.01, 0.02, 0.03, 0.05]

        # Initial parameters
        best_ml_w = 0.60
        best_mgr = 0.05
        best_prior_m = 5.0
        best_prior_form_m = 3.0
        best_draw = 1.0
        best_var = 0.02

        def compute_wc_stats(matches):
            stats = defaultdict(dict)
            for m in matches:
                hn = normalize_name(m["home_team"])
                an = normalize_name(m["away_team"])
                hs = m["home_score"]
                as_ = m["away_score"]
                for t in (hn, an):
                    if t not in stats:
                        stats[t] = {"matches": 0, "goals_for": 0.0, "goals_against": 0.0}
                stats[hn]["matches"] += 1
                stats[hn]["goals_for"] += hs
                stats[hn]["goals_against"] += as_
                stats[an]["matches"] += 1
                stats[an]["goals_for"] += as_
                stats[an]["goals_against"] += hs
            return stats

        # Precompute folds stats
        fold_stats = [compute_wc_stats(train_set) for train_set, _ in folds]

        def evaluate(ml_w, mgr, prior_m, prior_form_m, draw_m, var_v):
            # Temporarily set attributes
            self.ML_WEIGHT = ml_w
            self.ANALYTICAL_WEIGHT = 1.0 - ml_w
            self.MANAGER_DELTA_COEF = mgr
            self.PRIOR_M = prior_m
            self.PRIOR_FORM_M = prior_form_m
            self.DRAW_CALIB_MULT = draw_m
            self.STOCHASTIC_VAR = var_v

            total_loss = 0.0
            for idx, (train_set, val_set) in enumerate(folds):
                # Set stats for this fold
                self.wc_team_stats = fold_stats[idx]
                for m in val_set:
                    ht, at = m["home_team"], m["away_team"]
                    hs, as_ = m["home_score"], m["away_score"]
                    try:
                        h_xg, a_xg = self._expected_goals_raw(ht, at)
                    except Exception:
                        h_xg, a_xg = 1.3, 1.3

                    # Winner loss
                    score_probs = self._score_distribution(h_xg, a_xg)
                    wp = self._winner_probabilities(ht, at, h_xg, a_xg, score_probs)
                    if hs > as_:
                        p = wp.get("home", 0.33)
                    elif hs < as_:
                        p = wp.get("away", 0.33)
                    else:
                        p = wp.get("draw", 0.33)
                    p = max(1e-5, min(1.0 - 1e-5, p))
                    log_loss = -math.log(p)

                    # xG MSE loss
                    mse = (h_xg - hs) ** 2 + (a_xg - as_) ** 2

                    total_loss += log_loss + 0.1 * mse
            return total_loss

        # Coordinate descent (2 passes)
        for pass_num in range(2):
            # Tune ML_WEIGHT
            best_val_loss = float('inf')
            for w in ml_w_choices:
                loss = evaluate(w, best_mgr, best_prior_m, best_prior_form_m, best_draw, best_var)
                if loss < best_val_loss:
                    best_val_loss = loss
                    best_ml_w = w

            # Tune MANAGER_DELTA_COEF
            best_val_loss = float('inf')
            for m_coef in mgr_choices:
                loss = evaluate(best_ml_w, m_coef, best_prior_m, best_prior_form_m, best_draw, best_var)
                if loss < best_val_loss:
                    best_val_loss = loss
                    best_mgr = m_coef

            # Tune PRIOR_M
            best_val_loss = float('inf')
            for pm in prior_m_choices:
                loss = evaluate(best_ml_w, best_mgr, pm, best_prior_form_m, best_draw, best_var)
                if loss < best_val_loss:
                    best_val_loss = loss
                    best_prior_m = pm

            # Tune PRIOR_FORM_M
            best_val_loss = float('inf')
            for pfm in prior_form_m_choices:
                loss = evaluate(best_ml_w, best_mgr, best_prior_m, pfm, best_draw, best_var)
                if loss < best_val_loss:
                    best_val_loss = loss
                    best_prior_form_m = pfm

            # Tune DRAW_CALIB_MULT
            best_val_loss = float('inf')
            for dc in draw_choices:
                loss = evaluate(best_ml_w, best_mgr, best_prior_m, best_prior_form_m, dc, best_var)
                if loss < best_val_loss:
                    best_val_loss = loss
                    best_draw = dc

            # Tune STOCHASTIC_VAR
            best_val_loss = float('inf')
            for sv in var_choices:
                loss = evaluate(best_ml_w, best_mgr, best_prior_m, best_prior_form_m, best_draw, sv)
                if loss < best_val_loss:
                    best_val_loss = loss
                    best_var = sv

        # Set the final best parameters
        self.ML_WEIGHT = best_ml_w
        self.ANALYTICAL_WEIGHT = 1.0 - best_ml_w
        self.MANAGER_DELTA_COEF = best_mgr
        self.PRIOR_M = best_prior_m
        self.PRIOR_FORM_M = best_prior_form_m
        self.DRAW_CALIB_MULT = best_draw
        self.STOCHASTIC_VAR = best_var

        # Restore original stats
        self.wc_team_stats = orig_wc_team_stats

        print("--- Optimal Hyperparameters Found via 5-Fold CV ---", file=sys.stderr)
        print(f"ML Weight: {self.ML_WEIGHT:.2f} (Analytical: {self.ANALYTICAL_WEIGHT:.2f})", file=sys.stderr)
        print(f"Manager Delta Coef: {self.MANAGER_DELTA_COEF:.4f}", file=sys.stderr)
        print(f"World Cup Prior M: {self.PRIOR_M:.2f}", file=sys.stderr)
        print(f"Recent Form Prior M: {self.PRIOR_FORM_M:.2f}", file=sys.stderr)
        print(f"Draw Calibration Mult: {self.DRAW_CALIB_MULT:.2f}", file=sys.stderr)
        print(f"Stochastic Variance Range: {self.STOCHASTIC_VAR:.4f}", file=sys.stderr)
        print("-----------------------------------------------------", file=sys.stderr)

    def train(self):
        self._load_results()
        self._load_current_players()
        self._load_player_stats()
        self._load_worldcup_teams()
        self._load_fifa_rankings()
        self._load_shootouts()
        self._load_worldcup_squads()
        self._load_worldcup_matches()
        self._load_worldcup_round_of_32()  # Highest-priority dataset — loaded last, dated latest
        self._load_worldcup_round_of_16()
        self.matches.sort(key=lambda x: x["date"])
        self._train_team_models()
        self._train_supervised_models()
        # Reload manager data on every train so it stays current without retraining
        self.manager_impact = ManagerImpact(self.manager_path)
        self._optimize_hyperparameters()
        self.calibrate_xg()
        print("Model training complete.", file=sys.stderr)

    # ── Maths helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _poisson_pmf(k: int, lam: float) -> float:
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return math.exp(-lam) * (lam ** k) / math.factorial(k)

    # ── Core prediction components ────────────────────────────────────────────
    def _team_recent_form(self, team):
        # Find last 10 matches for team to apply recency weighting
        recent = []
        norm_team = normalize_name(team)
        for m in reversed(self.matches):
            if "home_norm" not in m:
                m["home_norm"] = normalize_name(m["home_team"])
            if "away_norm" not in m:
                m["away_norm"] = normalize_name(m["away_team"])
            if m["home_norm"] == norm_team or m["away_norm"] == norm_team:
                recent.append(m)
                if len(recent) == 10:
                    break
        if not recent:
            h = self.team_stats[team]
            return h.goals_for / max(1, h.matches), h.goals_against / max(1, h.matches), 0.0
        
        weighted_gf = 0.0
        weighted_ga = 0.0
        total_weight = 0.0
        
        # Calculate recency based on index (0 is most recent, up to 9 is oldest)
        for i, m in enumerate(recent):
            # Time decay: most recent match weight is 1.0, decays linearly to 0.1
            base_weight = 1.0 - (i * 0.1)
            
            # Boost World Cup matches
            if m.get("tournament") == "FIFA World Cup":
                base_weight *= 2.0
                
            total_weight += base_weight
            
            if normalize_name(m["home_team"]) == normalize_name(team):
                weighted_gf += m["home_score"] * base_weight
                weighted_ga += m["away_score"] * base_weight
            else:
                weighted_gf += m["away_score"] * base_weight
                weighted_ga += m["home_score"] * base_weight
                
        return weighted_gf / total_weight, weighted_ga / total_weight, total_weight

    def _squad_attack_rating(self, team):
        roster = self.current_team_players.get(team, {})
        if not roster:
            h = self.team_stats[team]
            return h.goals_for / max(1, h.matches)
        
        sum_score = 0.0
        for norm_name, p_info in roster.items():
            p_goals = p_info.get("goals", 0)
            p_assists = p_info.get("assists", 0)
            p_xg = self.player_xg[team].get(norm_name, 0.0) if team in self.player_xg else 0.0
            # Attack score: goals + 0.5 * assists + club xG
            sum_score += p_goals + 0.5 * p_assists + p_xg
            
        return min(3.0, max(0.2, sum_score / 20.0))

    def _squad_defense_rating(self, team):
        roster = self.current_team_players.get(team, {})
        if not roster:
            h = self.team_stats[team]
            return h.goals_against / max(1, h.matches)
            
        gk_factor = 1.0
        df_minutes = 0.0
        
        for norm_name, p_info in roster.items():
            pos = p_info.get("position", "")
            mins = p_info.get("minutes", 0.0)
            
            if "GK" in pos and mins > 0:
                gk_cs = p_info.get("gk_clean_sheets", 0)
                gk_games = p_info.get("gk_games", 0)
                if gk_games > 0:
                    cs_pct = gk_cs / gk_games
                    gk_factor = 1.2 - 0.4 * cs_pct
            
            if "DF" in pos:
                df_minutes += mins
                
        df_exp = min(1.2, max(0.8, 1.5 - (df_minutes / 1000.0)))
        combined = gk_factor * df_exp
        
        h = self.team_stats[team]
        base_ga = h.goals_against / max(1, h.matches)
        return max(0.2, min(3.0, base_ga * combined))

    def _expected_goals_raw(self, home_team, away_team):
        avg = self.total_goals_scored / (2 * max(1, self.total_matches))
        if avg == 0: avg = 1.3

        hn = normalize_name(home_team)
        an = normalize_name(away_team)
        
        # 1. Offensive & Defensive Form (Historical)
        h_hist_gf = self.team_stats[home_team].goals_for / max(1, self.team_stats[home_team].matches)
        h_hist_ga = self.team_stats[home_team].goals_against / max(1, self.team_stats[home_team].matches)
        
        a_hist_gf = self.team_stats[away_team].goals_for / max(1, self.team_stats[away_team].matches)
        a_hist_ga = self.team_stats[away_team].goals_against / max(1, self.team_stats[away_team].matches)
        
        # 2. Recent Form (Time-Decayed with Bayesian Shrinkage)
        h_form_raw_gf, h_form_raw_ga, h_form_wt = self._team_recent_form(home_team)
        a_form_raw_gf, a_form_raw_ga, a_form_wt = self._team_recent_form(away_team)
        
        h_form_gf = (h_form_wt * h_form_raw_gf + self.PRIOR_FORM_M * h_hist_gf) / (h_form_wt + self.PRIOR_FORM_M)
        h_form_ga = (h_form_wt * h_form_raw_ga + self.PRIOR_FORM_M * h_hist_ga) / (h_form_wt + self.PRIOR_FORM_M)
        a_form_gf = (a_form_wt * a_form_raw_gf + self.PRIOR_FORM_M * a_hist_gf) / (a_form_wt + self.PRIOR_FORM_M)
        a_form_ga = (a_form_wt * a_form_raw_ga + self.PRIOR_FORM_M * a_hist_ga) / (a_form_wt + self.PRIOR_FORM_M)

        # 3. Manager Impact Multipliers (Multiplicative Interaction)
        man_mult_h, man_mult_a = 1.0, 1.0
        if getattr(self, "manager_impact", None) is not None:
            home_man_off = self.manager_impact.off_mult(home_team)
            away_man_def = self.manager_impact.def_mult(away_team)
            man_mult_h = (home_man_off + away_man_def) / 2.0
            
            away_man_off = self.manager_impact.off_mult(away_team)
            home_man_def = self.manager_impact.def_mult(home_team)
            man_mult_a = (away_man_off + home_man_def) / 2.0
        
        # 4. Team Strength (Squad Rating with Multiplicative Manager Interaction)
        h_squad_att = self._squad_attack_rating(home_team) * man_mult_h
        h_squad_def = self._squad_defense_rating(home_team)
        a_squad_att = self._squad_attack_rating(away_team) * man_mult_a
        a_squad_def = self._squad_defense_rating(away_team)
        
        # 5. WC Performance (with Bayesian Shrinkage towards historical average)
        h_wc = self.wc_team_stats.get(hn, {})
        a_wc = self.wc_team_stats.get(an, {})
        
        h_wc_m = h_wc.get("matches", 0)
        h_wc_gf_raw = (h_wc.get("goals_for", 0) / max(1, h_wc_m)) if h_wc_m > 0 else avg
        h_wc_ga_raw = (h_wc.get("goals_against", 0) / max(1, h_wc_m)) if h_wc_m > 0 else avg
        h_wc_gf = (h_wc_m * h_wc_gf_raw + self.PRIOR_M * h_hist_gf) / (h_wc_m + self.PRIOR_M)
        h_wc_ga = (h_wc_m * h_wc_ga_raw + self.PRIOR_M * h_hist_ga) / (h_wc_m + self.PRIOR_M)
        
        a_wc_m = a_wc.get("matches", 0)
        a_wc_gf_raw = (a_wc.get("goals_for", 0) / max(1, a_wc_m)) if a_wc_m > 0 else avg
        a_wc_ga_raw = (a_wc.get("goals_against", 0) / max(1, a_wc_m)) if a_wc_m > 0 else avg
        a_wc_gf = (a_wc_m * a_wc_gf_raw + self.PRIOR_M * a_hist_gf) / (a_wc_m + self.PRIOR_M)
        a_wc_ga = (a_wc_m * a_wc_ga_raw + self.PRIOR_M * a_hist_ga) / (a_wc_m + self.PRIOR_M)
        
        # 6. H2H Dynamic
        key = frozenset([home_team, away_team])
        h2h = self.h2h_matches[key]
        n_h2h = len(h2h)
        h2h_w = min(0.08, 0.08 * (n_h2h / 3.0))
        h2h_h_gf, h2h_a_gf = avg, avg
        if n_h2h > 0:
            hg, ag = [], []
            for m in h2h:
                if m["home_team"] == home_team:
                    hg.append(m["home_score"]); ag.append(m["away_score"])
                else:
                    hg.append(m["away_score"]); ag.append(m["home_score"])
            h2h_h_gf = sum(hg) / n_h2h
            h2h_a_gf = sum(ag) / n_h2h
            
        rem_w = 0.08 - h2h_w
        form_w = 0.15 + (rem_w / 2)
        strength_w = 0.10 + (rem_w / 2)
        
        # 7. Base xG construction using exact weights
        home_base_xg = (
            0.25 * h_hist_gf + 
            0.20 * a_hist_ga + 
            form_w * ((h_form_gf + a_form_ga) / 2.0) +
            strength_w * (h_squad_att * avg) +
            0.04 * ((h_wc_gf + a_wc_ga) / 2.0) + 
            h2h_w * h2h_h_gf
        ) / 0.82
        
        away_base_xg = (
            0.25 * a_hist_gf + 
            0.20 * h_hist_ga + 
            form_w * ((a_form_gf + h_form_ga) / 2.0) +
            strength_w * (a_squad_att * avg) +
            0.04 * ((a_wc_gf + h_wc_ga) / 2.0) + 
            h2h_w * h2h_a_gf
        ) / 0.82

        # 8. Multipliers: Elo (10%), FIFA Rank (3% with Fallback)
        elo_gap = (self.elo_ratings[home_team] - self.elo_ratings[away_team]) / 400.0
        try:
            sigmoid_val = 1.0 / (1.0 + math.exp(-elo_gap))
        except OverflowError:
            sigmoid_val = 1.0 if elo_gap > 0 else 0.0
        elo_mult = 0.7 + 0.6 * sigmoid_val
        
        h_rank = 100
        a_rank = 100
        if hasattr(self, "wc_teams_info") and hn in self.wc_teams_info:
            h_rank = int(self.wc_teams_info[hn].get("fifa_ranking", 100) or 100)
        elif hn in self.fifa_rankings:
            h_rank = self.fifa_rankings[hn]

        if hasattr(self, "wc_teams_info") and an in self.wc_teams_info:
            a_rank = int(self.wc_teams_info[an].get("fifa_ranking", 100) or 100)
        elif an in self.fifa_rankings:
            a_rank = self.fifa_rankings[an]

        rank_gap = (a_rank - h_rank) / 100.0 # Positive if home is better
        rank_mult = max(0.5, min(1.5, 1.0 + rank_gap))

        # 9. Reallocated blend weights (removing additive manager component)
        home_xg = 0.87 * home_base_xg + 0.10 * (elo_mult * avg) + 0.03 * (rank_mult * avg)
        away_xg = 0.87 * away_base_xg + 0.10 * ((2.0 - elo_mult) * avg) + 0.03 * ((2.0 - rank_mult) * avg)

        return home_xg, away_xg

    def _expected_goals(self, home_team, away_team):
        hn = normalize_name(home_team)
        an = normalize_name(away_team)
        
        home_xg, away_xg = self._expected_goals_raw(home_team, away_team)

        # Apply Goalkeeper PotM Adjustment
        if hasattr(self, "gk_potm_counts"):
            a_gk = self.team_goalkeeper.get(away_team) or self.team_goalkeeper.get(an) or getattr(self, "fallback_goalkeepers", {}).get(an)
            if a_gk:
                a_gk_potm = self.gk_potm_counts.get(normalize_name(a_gk), 0)
                home_xg /= min(1.35, 1.0 + 0.10 * a_gk_potm)

            h_gk = self.team_goalkeeper.get(home_team) or self.team_goalkeeper.get(hn) or getattr(self, "fallback_goalkeepers", {}).get(hn)
            if h_gk:
                h_gk_potm = self.gk_potm_counts.get(normalize_name(h_gk), 0)
                away_xg /= min(1.35, 1.0 + 0.10 * h_gk_potm)

        # Apply Controlled Statistical Variation using self.STOCHASTIC_VAR
        import random as _random
        var_range = self.STOCHASTIC_VAR
        home_xg *= _random.uniform(1.0 - var_range, 1.0 + var_range)
        away_xg *= _random.uniform(1.0 - var_range, 1.0 + var_range)

        return max(0.15, min(3.5, home_xg)), max(0.15, min(3.5, away_xg))

    def _score_distribution(self, home_xg, away_xg):
        max_goals = max(9, math.ceil(max(home_xg, away_xg) + 8))
        probs = {}
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                probs[f"{h}-{a}"] = self._poisson_pmf(h, home_xg) * self._poisson_pmf(a, away_xg)
        return sorted(probs.items(), key=lambda x: x[1], reverse=True)

    def _select_scoreline(self, home_team, away_team, calibrated_matrix, outcome):
        import hashlib
        import random as _random

        today_str = datetime.now(timezone.utc).date().isoformat()
        
        # Determine canonical ordering for reproducible but stochastic sampling seed
        team_a, team_b = sorted([home_team, away_team])
        seed_str  = f"{team_a}{team_b}{today_str}"
        seed      = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2 ** 32)
        rng       = _random.Random(seed)

        # Filter the matrix for cells matching the predicted outcome category
        candidates = []
        for score_str, prob in calibrated_matrix.items():
            h, a = map(int, score_str.split("-"))
            if outcome == "home" and h > a:
                candidates.append((score_str, prob))
            elif outcome == "away" and h < a:
                candidates.append((score_str, prob))
            elif outcome == "draw" and h == a:
                candidates.append((score_str, prob))

        # The predicted scoreline is the mode (highest-probability scoreline) within this category
        best_score, _ = max(candidates, key=lambda x: x[1])
        return best_score, rng

    def _allocate_scorers(self, candidates, num_goals, rng):
        """Allocate num_goals goals to players using weighted probability sampling."""
        if num_goals == 0 or not candidates:
            return []

        goal_tally = {}
        weights = {c["player"]: max(0.001, c["probability"]) for c in candidates}
        player_data = {c["player"]: c for c in candidates}

        for _ in range(num_goals):
            names = list(weights.keys())
            wts = [weights[n] for n in names]
            chosen = rng.choices(names, weights=wts, k=1)[0]

            goal_tally[chosen] = goal_tally.get(chosen, 0) + 1
            if goal_tally[chosen] == 1:
                weights[chosen] *= 0.40
            elif goal_tally[chosen] == 2:
                weights[chosen] *= 0.10
            else:
                weights[chosen] *= 0.01

        results = []
        for player, n_goals in goal_tally.items():
            base = player_data[player]
            p1  = round(base["prob_1_goal"] * 100)
            p2  = round(base["prob_2_or_more"] * 100)
            preds = [{"goal_count": 1, "probability": p1}]
            if p2 > 0:
                preds.append({"goal_count": 2, "probability": p2})
            results.append({"name": player, "predictions": preds, "_goals": n_goals})

        results.sort(key=lambda x: (-x["_goals"], -player_data[x["name"]]["probability"]))
        return [{"name": r["name"], "predictions": r["predictions"]} for r in results]

    def _winner_probabilities(self, home_team, away_team, home_xg, away_xg, score_probs):
        ph = pd = pa = 0.0
        for s, p in score_probs:
            h, a = map(int, s.split("-"))
            if h > a: ph += p
            elif h < a: pa += p
            else: pd += p
        tot = ph + pd + pa
        if tot > 0: ph /= tot; pd /= tot; pa /= tot
        poisson = {"home": ph, "draw": pd, "away": pa}
        
        ml_d = dict(poisson)
        if HAS_ML and self.ml_winner_model:
            hs = self.team_stats[home_team]; as_ = self.team_stats[away_team]
            x_fwd = np.array([[
                self.elo_ratings[home_team], self.elo_ratings[away_team],
                hs.goals_for / max(1, hs.matches), hs.goals_against / max(1, hs.matches),
                as_.goals_for / max(1, as_.matches), as_.goals_against / max(1, as_.matches),
            ]])
            x_rev = np.array([[
                self.elo_ratings[away_team], self.elo_ratings[home_team],
                as_.goals_for / max(1, as_.matches), as_.goals_against / max(1, as_.matches),
                hs.goals_for / max(1, hs.matches), hs.goals_against / max(1, hs.matches),
            ]])
            
            ml_fwd = self.ml_winner_model.predict_proba(x_fwd)[0]
            ml_rev = self.ml_winner_model.predict_proba(x_rev)[0]
            
            # Neutralize implicit home bias
            ml_home = (ml_fwd[2] + ml_rev[0]) / 2.0
            ml_draw = (ml_fwd[1] + ml_rev[1]) / 2.0
            ml_away = (ml_fwd[0] + ml_rev[2]) / 2.0
            ml_d = {"away": ml_away, "draw": ml_draw, "home": ml_home}
            
        # 1. Configurable Hybrid Blend using optimized weights
        ml_w = self.ML_WEIGHT
        ana_w = self.ANALYTICAL_WEIGHT
        blended = {k: ml_w * ml_d[k] + ana_w * poisson[k] for k in poisson}
        
        # 2. Manager rating delta using optimized manager coefficient
        mi = getattr(self, "manager_impact", None)
        if mi is not None:
            delta = (mi.rating(home_team) - mi.rating(away_team)) * self.MANAGER_DELTA_COEF
            blended["home"] = max(0.0, blended["home"] + delta)
            blended["away"] = max(0.0, blended["away"] - delta)
            tot2 = sum(blended.values())
            if tot2 > 0:
                blended = {k: v / tot2 for k, v in blended.items()}
                
        # 3. Dynamic Historical Draw Calibration using optimized threshold multiplier
        if not hasattr(self, "_cached_hist_draw_rate"):
            historical_draws = 0
            total_matches = len(self.matches) if hasattr(self, 'matches') else 0
            if total_matches > 0:
                for m in self.matches:
                    if m.get("home_score") == m.get("away_score"):
                        historical_draws += 1
                self._cached_hist_draw_rate = historical_draws / total_matches
            else:
                self._cached_hist_draw_rate = 0.25
        hist_draw_rate = self._cached_hist_draw_rate
            
        if blended["draw"] < hist_draw_rate * 0.8 * self.DRAW_CALIB_MULT:
            diff_teams = abs(blended["home"] - blended["away"])
            if diff_teams < 0.20:
                blended["draw"] = (blended["draw"] + hist_draw_rate) / 2.0
                tot3 = sum(blended.values())
                blended = {k: v / tot3 for k, v in blended.items()}

        # 4. Generate the calibrated joint probability matrix using rounded probabilities
        p_home_round = round(blended["home"], 4)
        p_away_round = round(blended["away"], 4)
        p_draw_round = round(blended["draw"], 4)
        
        # Ensure they sum to exactly 1.0
        round_sum = p_home_round + p_away_round + p_draw_round
        if abs(round_sum - 1.0) > 1e-9:
            diff = round(1.0 - round_sum, 4)
            max_key = max(blended, key=blended.get)
            if max_key == "home": p_home_round = round(p_home_round + diff, 4)
            elif max_key == "away": p_away_round = round(p_away_round + diff, 4)
            else: p_draw_round = round(p_draw_round + diff, 4)

        calibrated = {}
        for score_str, prob in score_probs:
            h, a = map(int, score_str.split("-"))
            if h > a:
                calibrated[score_str] = prob * (p_home_round / ph) if ph > 0 else prob
            elif h < a:
                calibrated[score_str] = prob * (p_away_round / pa) if pa > 0 else prob
            else:
                calibrated[score_str] = prob * (p_draw_round / pd) if pd > 0 else prob

        total_sum = sum(calibrated.values())
        if total_sum > 0:
            calibrated = {k: v / total_sum for k, v in calibrated.items()}
        else:
            calibrated = dict(score_probs)

        # 5. Advanced Confidence Calculation
        # Build blended dict with rounded values for consistency
        blended_round = {"home": p_home_round, "draw": p_draw_round, "away": p_away_round}
        sorted_probs = sorted(blended_round.values(), reverse=True)
        margin_score = sorted_probs[0] - sorted_probs[1]

        mad = (abs(ml_d["home"] - poisson["home"]) + abs(ml_d["draw"] - poisson["draw"]) + abs(ml_d["away"] - poisson["away"])) / 3.0
        agreement_score = max(0.0, 1.0 - (mad / (2.0 / 3.0)))

        entropy = 0.0
        for prob in calibrated.values():
            if prob > 1e-9:
                entropy -= prob * math.log2(prob)
        entropy_score = max(0.0, 1.0 - (entropy / 6.0))

        confidence_val = 0.40 * margin_score + 0.30 * agreement_score + 0.30 * entropy_score
        if confidence_val >= 0.70:
            confidence = "Very High"
        elif confidence_val >= 0.50:
            confidence = "High"
        elif confidence_val >= 0.30:
            confidence = "Medium"
        else:
            confidence = "Low"

        final = {k: v for k, v in blended_round.items()}
        final["confidence"] = confidence
        final["calibrated_matrix"] = calibrated
        return final

    def _first_team_to_score(self, home_xg, away_xg):
        no_goal = self._poisson_pmf(0, home_xg) * self._poisson_pmf(0, away_xg)
        scoring = 1.0 - no_goal
        total = home_xg + away_xg
        if total == 0:
            return {"home": 0.0, "away": 0.0, "none": 1.0}
        return {
            "home": round(home_xg / total * scoring, 4),
            "away": round(away_xg / total * scoring, 4),
            "none": round(no_goal, 4),
        }

    def _clean_sheet_probability(self, opponent_xg, team=None):
        """Poisson-based clean sheet; annotates GK name if available."""
        p = round(self._poisson_pmf(0, opponent_xg), 4)
        result = {"probability": p}
        if team:
            gk_name = None
            if team in self.team_goalkeeper:
                gk_name = self.team_goalkeeper[team]
            elif normalize_name(team) in self.team_goalkeeper:
                gk_name = self.team_goalkeeper[normalize_name(team)]
            elif team in self.fallback_goalkeepers:
                gk_name = self.fallback_goalkeepers[team]
            elif normalize_name(team) in self.fallback_goalkeepers:
                gk_name = self.fallback_goalkeepers[normalize_name(team)]

            if gk_name:
                result["goalkeeper"] = gk_name
                # Goalkeeper of the match boost
                if hasattr(self, "gk_potm_counts"):
                    gk_potm = self.gk_potm_counts.get(normalize_name(gk_name), 0)
                    gk_factor = min(1.35, 1.0 + 0.10 * gk_potm)
                    result["probability"] = round(min(0.99, p * gk_factor), 4)
        return result

    def _over_under(self, home_xg, away_xg):
        """P(total goals > threshold) for 1.5, 2.5, 3.5 using Poisson convolution."""
        results = {}
        for threshold in (1.5, 2.5, 3.5):
            p_under = 0.0
            cap = int(threshold)  # floor: goals <= cap means 'under'
            for total in range(cap + 1):
                for h in range(total + 1):
                    a = total - h
                    p_under += self._poisson_pmf(h, home_xg) * self._poisson_pmf(a, away_xg)
            p_over = round(max(0.0, 1.0 - p_under), 4)
            p_under = round(min(1.0, p_under), 4)
            results[f"{threshold}"] = {"over": p_over, "under": p_under}
        return results

    def _top_scorers(self, home_team, away_team, home_xg, away_xg, home_limit=10, away_limit=10):
        """Top scorers with historical goals + xG-based probability distributions."""
        def build(team, match_xg):
            hist = self.player_goals[team]
            xg_map = self.player_xg.get(team, {})
            t_stats = self.team_stats[team]
            avg_goals = (t_stats.goals_for / max(1, t_stats.matches))

            norm_team = normalize_name(team)
            wc_player_set = self.wc_squads.get(norm_team, set()) if hasattr(self, "wc_squads") else set()
            h_wc = self.wc_team_stats.get(norm_team, {})
            wc_matches = h_wc.get("matches", 0)

            if wc_player_set:
                appeared = {p for p in wc_player_set if p in getattr(self, "wc_players_appeared", set())}
                if len(appeared) >= 11:
                    eligible = appeared
                else:
                    eligible = wc_player_set
                    
                players = {}
                for norm_name in eligible:
                    p_info = self.wc_player_info[norm_team].get(norm_name, {})
                    orig_name = p_info.get("original_name", norm_name)
                    g_hist = hist.get(norm_name, 0)
                    g_tourney = p_info.get("goals", 0)

                    # Extract club-season stats
                    club_npxg = self.player_npxg[team].get(norm_name, 0.0)
                    club_sot = self.player_sot[team].get(norm_name, 0.0)
                    club_nineties = self.player_nineties[team].get(norm_name, 0.0)

                    # Calculate club prior rate
                    if club_nineties > 0:
                        npxg_per90 = club_npxg / club_nineties
                        sot_per90 = club_sot / club_nineties
                        club_prior_rate = 0.7 * npxg_per90 + 0.3 * (sot_per90 * 0.15)
                    else:
                        club_prior_rate = 0.0

                    # Calculate tournament rate
                    tourney_rate = (g_tourney / max(1, wc_matches)) if wc_matches > 0 else 0.0

                    # Blend club prior and tournament rate
                    if club_nineties > 0:
                        if wc_matches > 0:
                            blended_rate = 0.6 * club_prior_rate + 0.4 * tourney_rate
                        else:
                            blended_rate = club_prior_rate
                    else:
                        blended_rate = tourney_rate

                    # Fallback to historical rate if blended rate is 0.0 but player has historical goals
                    if blended_rate == 0.0 and g_hist > 0:
                        blended_rate = g_hist / max(1, t_stats.matches)

                    # Base rate from goals if everything is 0.0
                    if blended_rate == 0.0:
                        pos = p_info.get("position", "")
                        if "FW" in pos:
                            blended_rate = 0.15
                        elif "MF" in pos:
                            blended_rate = 0.05
                        else:
                            blended_rate = 0.01

                    # Expected goals for the match for this player
                    player_match_xg = blended_rate * (match_xg / max(0.5, avg_goals))
                    
                    # Poisson probabilities
                    prob_score = 1.0 - self._poisson_pmf(0, player_match_xg)
                    prob_1_goal = self._poisson_pmf(1, player_match_xg)
                    prob_2plus = max(0.0, 1.0 - self._poisson_pmf(0, player_match_xg) - self._poisson_pmf(1, player_match_xg))

                    players[orig_name] = {
                        "goals": g_tourney,
                        "prob_score": prob_score,
                        "prob_1_goal": prob_1_goal,
                        "prob_2plus": prob_2plus,
                        "xg_season": club_npxg
                    }
            else:
                players = {}

            all_entries = []
            for player, info in players.items():
                g = info["goals"]
                prob_score = info["prob_score"]
                prob_1_goal = info["prob_1_goal"]
                prob_2plus = info["prob_2plus"]

                # Player of the Match (PotM) boost
                if hasattr(self, "potm_counts"):
                    potm_count = self.potm_counts.get(normalize_name(player), 0)
                    potm_mult = 1.0 + 0.15 * potm_count
                    prob_score = round(min(0.99, prob_score * potm_mult), 4)
                    prob_1_goal = round(min(0.99, prob_1_goal * potm_mult), 4)
                    prob_2plus = round(min(0.99, prob_2plus * potm_mult), 4)

                entry = {
                    "player":           player,
                    "goals":            g,
                    "probability":      round(prob_score * 100.0, 1),
                    "prob_1_goal":      round(max(0.0, prob_1_goal) * 100.0, 1),
                    "prob_2_or_more":   round(max(0.0, prob_2plus) * 100.0, 1),
                }
                if info["xg_season"] > 0:
                    entry["xg_season"] = round(info["xg_season"], 3)
                all_entries.append(entry)

            # Sort by probability descending, goals descending
            all_entries.sort(key=lambda x: (-x["probability"], -x["goals"]))
            return all_entries[:10]

        def build_limited(team, match_xg, limit):
            if limit == 0:
                return []
            return build(team, match_xg)[:limit]

        return {
            "home_scorers": build_limited(home_team, home_xg, home_limit),
            "away_scorers": build_limited(away_team, away_xg, away_limit),
        }

    # ── Public API ────────────────────────────────────────────────────────────
    def available_teams(self):
        if getattr(self, "wc_teams_list", None):
            return self.wc_teams_list
        teams = set()
        if hasattr(self, "worldcup_teams_path") and self.worldcup_teams_path.exists():
            with open(self.worldcup_teams_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    name = row.get("team_name", "").strip()
                    if name:
                        teams.add(name)
        if not teams:
            teams = {t for t in self.team_stats.keys() if t}
        return sorted(list(teams))

    def _infer_stage(self, home_team, away_team):
        """Automatically determine tournament stage from loaded fixture data.

        Returns 'Round of 32' if both teams appear in the stored R32 fixture list,
        otherwise returns 'Group Stage' as the safe default.
        """
        hn = normalize_name(home_team)
        an = normalize_name(away_team)
        pairs = getattr(self, "r32_team_pairs", frozenset())
        if (hn, an) in pairs or (an, hn) in pairs:
            return "Round of 32"
        return "Group Stage"

    def predict(self, home_team, away_team=None, stage=None):
        if isinstance(home_team, dict):
            away_team = home_team.get("away_team", "")
            home_team = home_team.get("home_team", "")

        # Auto-infer stage from fixture data if not explicitly supplied
        if stage is None:
            stage = self._infer_stage(home_team, away_team)

        if not home_team or not away_team:
            raise ValueError(json.dumps({
                "validation_errors": [
                    {"field": "input", "message": "Both home_team and away_team must be provided."}
                ]
            }))
        # Safeguard for old models or unpopulated players roster:
        if not hasattr(self, "current_team_players") or not self.current_team_players:
            self.current_team_players = defaultdict(dict)
            self._load_current_players()
            # Also rebuild starting goalkeepers if team_goalkeeper is empty or missing
            if not self.team_goalkeeper:
                self.team_goalkeeper = {}
                for team, roster in self.current_team_players.items():
                    gks = []
                    for norm_name, p_info in roster.items():
                        if "GK" in p_info["position"]:
                            gk_mins = p_info.get("gk_minutes", 0.0) or p_info.get("minutes", 0.0)
                            gks.append((gk_mins, p_info["original_name"]))
                    if gks:
                        gks.sort(reverse=True)
                        self.team_goalkeeper[team] = gks[0][1]

        known = self.team_stats
        errors = []
        for label, team in [("home_team", home_team), ("away_team", away_team)]:
            if team not in known:
                close = [t for t in known if team.lower() in t.lower()][:5]
                errors.append({"field": label, "value": team,
                               "message": f"Team '{team}' not found.",
                               "suggestions": close or self.available_teams()[:10]})
        if errors:
            raise ValueError(json.dumps({"validation_errors": errors}))

        home_xg, away_xg = self._expected_goals(home_team, away_team)
        score_probs       = self._score_distribution(home_xg, away_xg)

        # ── Winner probabilities (computed first) ──────────────────────────────
        wp = self._winner_probabilities(home_team, away_team, home_xg, away_xg, score_probs)
        win_prob = {
            "home_team": {"team": home_team, "probability": round(wp["home"] * 100)},
            "draw":      {"probability":      round(wp["draw"] * 100)},
            "away_team": {"team": away_team,  "probability": round(wp["away"] * 100)},
        }

        calibrated_matrix = wp["calibrated_matrix"]

        # Derive expected goals directly from the calibrated joint matrix
        home_xg_cal = sum(int(s.split("-")[0]) * p for s, p in calibrated_matrix.items())
        away_xg_cal = sum(int(s.split("-")[1]) * p for s, p in calibrated_matrix.items())

        # Determine the winner category (highest probability outcome category)
        p_home = wp["home"]
        p_draw = wp["draw"]
        p_away = wp["away"]
        if p_home > p_away and p_home > p_draw:
            predicted_outcome = "home"
        elif p_away > p_home and p_away > p_draw:
            predicted_outcome = "away"
        else:
            predicted_outcome = "draw"

        # ── Probability-consistent scoreline selection (Mode of winner category) ──
        predicted_line, rng = self._select_scoreline(home_team, away_team, calibrated_matrix, predicted_outcome)
        pred_h, pred_a = map(int, predicted_line.split("-"))

        # Derive BTTS from calibrated matrix
        btts = sum(p for s, p in calibrated_matrix.items() if int(s.split("-")[0]) > 0 and int(s.split("-")[1]) > 0)
        btts_pct = round(btts * 100)

        # ── First team to score (derived from calibrated matrix) ─────────────────
        fts_none = sum(p for s, p in calibrated_matrix.items() if int(s.split("-")[0]) == 0 and int(s.split("-")[1]) == 0)
        scoring_prob = 1.0 - fts_none
        if home_xg_cal + away_xg_cal > 0:
            fts_home = (home_xg_cal / (home_xg_cal + away_xg_cal)) * scoring_prob
            fts_away = (away_xg_cal / (home_xg_cal + away_xg_cal)) * scoring_prob
        else:
            fts_home = 0.0
            fts_away = 0.0
        
        if fts_home >= fts_away:
            first_scorer = {"team": home_team, "probability": round(fts_home * 100)}
        else:
            first_scorer = {"team": away_team, "probability": round(fts_away * 100)}

        # ── Clean sheet per team (derived from calibrated matrix) ───────────────
        cs_home_prob = sum(p for s, p in calibrated_matrix.items() if int(s.split("-")[1]) == 0)
        cs_away_prob = sum(p for s, p in calibrated_matrix.items() if int(s.split("-")[0]) == 0)

        # Retrieve goalkeeper names
        def get_gk_name(team):
            hn = normalize_name(team)
            if team in self.team_goalkeeper: return self.team_goalkeeper[team]
            if hn in self.team_goalkeeper: return self.team_goalkeeper[hn]
            if team in self.fallback_goalkeepers: return self.fallback_goalkeepers[team]
            if hn in self.fallback_goalkeepers: return self.fallback_goalkeepers[hn]
            return None

        gk_home = get_gk_name(home_team) or "—"
        gk_away = get_gk_name(away_team) or "—"

        def _fmt_gk(name):
            parts = name.split()
            return f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) > 1 else name

        home_cs_pct = round(cs_home_prob * 100)
        away_cs_pct = round(cs_away_prob * 100)

        # ── Player scorers (derived from the matrix-calibrated xG values) ───────
        raw_scorers = self._top_scorers(
            home_team, away_team, home_xg_cal, away_xg_cal,
            home_limit=100, away_limit=100
        )

        home_goals_list = self._allocate_scorers(raw_scorers["home_scorers"], pred_h, rng)
        away_goals_list = self._allocate_scorers(raw_scorers["away_scorers"], pred_a, rng)

        player_prediction = {
            "home_team": {
                "team": home_team,
                "goal": home_goals_list,
                "clean_sheet_prediction": {
                    "goalkeeper":  _fmt_gk(gk_home),
                    "prediction":  home_cs_pct >= 30,
                    "probability": home_cs_pct,
                },
            },
            "away_team": {
                "team": away_team,
                "goal": away_goals_list,
                "clean_sheet_prediction": {
                    "goalkeeper":  _fmt_gk(gk_away),
                    "prediction":  away_cs_pct >= 30,
                    "probability": away_cs_pct,
                },
            },
        }

        # ── Penalty shootout forecast for knockout draws ──────────────────────
        is_knockout = stage != "Group Stage"
        is_predicted_draw = (pred_h == pred_a)
        show_shootout = is_knockout and is_predicted_draw
        
        penalty_shootout = {"show_shootout": False}
        if show_shootout:
            hn = normalize_name(home_team)
            an = normalize_name(away_team)
            
            def get_rate(team_norm):
                stats = self.shootout_stats.get(team_norm, {"wins": 0, "total": 0})
                if stats.get("total", 0) == 0:
                    return 0.5
                return stats["wins"] / stats["total"]

            p1 = get_rate(hn)
            p2 = get_rate(an)
            home_shootout_prob = p1 / (p1 + p2) if (p1 + p2) > 0 else 0.5
            
            penalty_shootout = {
                "show_shootout": True,
                "home_win_probability": round(home_shootout_prob * 100),
                "away_win_probability": round((1.0 - home_shootout_prob) * 100),
                "predicted_winner": home_team if home_shootout_prob >= 0.5 else away_team
            }

        return {
            "output": {
                "match_prediction": {
                    "win_probabilities": win_prob,
                    "confidence": wp.get("confidence", "Medium"),
                },
                "score_prediction": {
                    "predicted_scoreline": {
                        "home_team":  home_team,
                        "home_goals": pred_h,
                        "away_team":  away_team,
                        "away_goals": pred_a,
                    },
                    "total_goals": pred_h + pred_a,
                },
                "goal_insights": {
                    "first_team_to_score": first_scorer,
                    "both_teams_to_score": {
                        "prediction": btts_pct >= 50,
                        "probability": btts_pct,
                    },
                },
                "player_prediction": player_prediction,
                "penalty_shootout": penalty_shootout,
            }
        }

def get_char():
    import sys
    import tty
    import termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def interactive_select_team(prompt, teams):
    import sys
    if not sys.stdin.isatty():
        while True:
            val = input(f"{prompt} (available: {', '.join(teams[:5])}...): ").strip()
            matches = [t for t in teams if t.lower() == val.lower()]
            if matches:
                return matches[0]
            print(f"Invalid team. Please select from: {', '.join(teams)}")

    query = ""
    selected_idx = 0
    
    while True:
        filtered = [t for t in teams if query.lower() in t.lower()]
        if not filtered:
            filtered = ["No matching teams found"]
            
        selected_idx = min(selected_idx, len(filtered) - 1)
        selected_idx = max(0, selected_idx)
        
        sys.stdout.write(f"\r{prompt}: {query}\x1b[0J\n")
        
        display_limit = 10
        for i, team in enumerate(filtered[:display_limit]):
            if i == selected_idx:
                sys.stdout.write(f" > \033[1;36m{team}\033[0m\n")
            else:
                sys.stdout.write(f"   {team}\n")
        if len(filtered) > display_limit:
            sys.stdout.write(f"   ... ({len(filtered) - display_limit} more matches)\n")
            
        lines_to_go_up = min(len(filtered), display_limit) + (1 if len(filtered) > display_limit else 0) + 1
        sys.stdout.write(f"\033[{lines_to_go_up}A")
        cursor_col = len(prompt) + len(query) + 3
        sys.stdout.write(f"\033[{cursor_col}C")
        sys.stdout.flush()
        
        ch = get_char()
        
        if ch in ('\r', '\n'):
            if filtered and filtered[0] != "No matching teams found":
                sys.stdout.write(f"\r\033[0J")
                sys.stdout.write(f"\033[{lines_to_go_up - 1}B\r\033[0J")
                sys.stdout.write(f"\033[{lines_to_go_up - 1}A")
                sys.stdout.write(f"{prompt}: \033[1;32m{filtered[selected_idx]}\033[0m\n")
                sys.stdout.flush()
                return filtered[selected_idx]
        elif ch in ('\x7f', '\x08'):
            query = query[:-1]
            selected_idx = 0
        elif ch == '\x1b[A':
            selected_idx = max(0, selected_idx - 1)
        elif ch == '\x1b[B':
            selected_idx = min(len(filtered) - 1, selected_idx + 1)
        elif len(ch) == 1 and ch.isprintable():
            query += ch
            selected_idx = 0


# Bind classes to sports_prediction_colab_app and register the module to ensure
# seamless pickling/unpickling compatibility in Google Colab, notebooks, and scripts.
import sys
sys.modules['sports_prediction_colab_app'] = sys.modules.get('sports_prediction_colab_app', sys.modules[__name__])
FootballPredictionModel.__module__ = 'sports_prediction_colab_app'
ManagerImpact.__module__ = 'sports_prediction_colab_app'
TeamStats.__module__ = 'sports_prediction_colab_app'
_EloDict.__module__ = 'sports_prediction_colab_app'
try:
    import __main__
    __main__.FootballPredictionModel = FootballPredictionModel
    __main__.ManagerImpact = ManagerImpact
    __main__.TeamStats = TeamStats
    __main__._EloDict = _EloDict
except Exception:
    pass


# ── Model versioning helpers ─────────────────────────────────────────────────
_VERSION_PATTERN = re.compile(r"GoalGPT_version_(\d+)\.pkl$")


def _next_version(directory: Path) -> int:
    """Return the next version number (highest existing + 1, or 1 if none found)."""
    versions = [
        int(m.group(1))
        for f in directory.glob("GoalGPT_version_*.pkl")
        for m in [_VERSION_PATTERN.match(f.name)]
        if m
    ]
    return max(versions, default=0) + 1


def _resolve_auto_pkl(directory: Path) -> "Path | None":
    """Locate the model to auto-load.

    GoalGPT_latest.pkl is the sole auto-load trigger.  Versioned files
    (GoalGPT_version_N.pkl) are treated as immutable archives; their presence
    does NOT suppress a new training run.  Deleting GoalGPT_latest.pkl is the
    canonical way to request a fresh retrain that increments the version counter.
    """
    latest = directory / "GoalGPT_latest.pkl"
    return latest if latest.exists() else None



class PredictionService:
    """Service layer between UI (CLI/Flask) and FootballPredictionModel."""
    def __init__(self, model):
        self.model = model

    def get_available_teams(self):
        # Return sorted list of available teams
        return sorted(self.model.available_teams())

    def generate_prediction(self, home_team, away_team, stage=None):
        known = self.model.team_stats
        errors = []
        for label, team in [("home_team", home_team), ("away_team", away_team)]:
            if team not in known:
                close = [t for t in known if team.lower() in t.lower()][:5]
                errors.append({
                    "field": label,
                    "value": team,
                    "message": f"Team '{team}' not found.",
                    "suggestions": close or self.get_available_teams()[:10]
                })
        if errors:
            return {"status": "error", "validation_errors": errors}

        if home_team == away_team:
            return {
                "status": "error",
                "validation_errors": [
                    {"field": "away_team", "value": away_team, "message": "Home and Away teams must be different."}
                ]
            }

        try:
            pred_res = self.model.predict(home_team, away_team, stage=stage)
        except Exception as e:
            return {"status": "error", "message": f"Prediction failed: {str(e)}"}

        # Extract normalized names for WC details
        hn = normalize_name(home_team)
        an = normalize_name(away_team)

        # Team stats comparisons
        home_ts = self.model.team_stats[home_team]
        away_ts = self.model.team_stats[away_team]

        home_rank = None
        home_manager = None
        if hn in self.model.wc_teams_info:
            home_rank = self.model.wc_teams_info[hn].get("fifa_ranking")
            home_manager = self.model.wc_teams_info[hn].get("manager")

        away_rank = None
        away_manager = None
        if an in self.model.wc_teams_info:
            away_rank = self.model.wc_teams_info[an].get("fifa_ranking")
            away_manager = self.model.wc_teams_info[an].get("manager")

        home_elo = self.model.elo_ratings.get(home_team, 1500.0)
        away_elo = self.model.elo_ratings.get(away_team, 1500.0)

        home_mgr_info = self.model.manager_impact.get(home_team)
        away_mgr_info = self.model.manager_impact.get(away_team)

        home_form_gf, home_form_ga, _ = self.model._team_recent_form(home_team)
        away_form_gf, away_form_ga, _ = self.model._team_recent_form(away_team)

        home_wc_stats = self.model.wc_team_stats.get(hn, {})
        away_wc_stats = self.model.wc_team_stats.get(an, {})

        h2h_key = frozenset([home_team, away_team])
        h2h_matches = self.model.h2h_matches.get(h2h_key, [])
        h2h_list = []
        for m in h2h_matches:
            h2h_list.append({
                "date": m["date"].strftime("%Y-%m-%d") if isinstance(m["date"], datetime) else str(m["date"]),
                "home_team": m["home_team"],
                "away_team": m["away_team"],
                "home_score": m["home_score"],
                "away_score": m["away_score"]
            })

        # Calculate Elo delta and Sigmoid details
        elo_diff = home_elo - away_elo

        # Dynamic Explanations Builder
        explanations = []
        if abs(elo_diff) > 100:
            dominant = home_team if elo_diff > 0 else away_team
            explanations.append(f"{dominant} holds a dominant historical strength advantage, leading the Elo ratings by {int(abs(elo_diff))} points.")
        elif abs(elo_diff) > 30:
            dominant = home_team if elo_diff > 0 else away_team
            explanations.append(f"{dominant} has a moderate rating edge, leading the Elo ratings by {int(abs(elo_diff))} points.")
        else:
            explanations.append(f"Both squads are extremely evenly matched historically, with a negligible Elo gap of {int(abs(elo_diff))} points.")

        home_net_form = home_form_gf - home_form_ga
        away_net_form = away_form_gf - away_form_ga
        if home_net_form > away_net_form + 0.6:
            explanations.append(f"{home_team} enters this clash in superior form, maintaining a +{home_net_form:.1f} net goal average recently compared to {away_team}'s +{away_net_form:.1f}.")
        elif away_net_form > home_net_form + 0.6:
            explanations.append(f"{away_team} has shown better recent consistency, carrying a +{away_net_form:.1f} net goal average recently compared to {home_team}'s +{home_net_form:.1f}.")

        home_mgr_rating = home_mgr_info.get("rating", 1.0)
        away_mgr_rating = away_mgr_info.get("rating", 1.0)
        if home_mgr_rating > away_mgr_rating + 0.15:
            explanations.append(f"Tactical edge goes to {home_team} under manager {home_mgr_info.get('manager')} (rating {home_mgr_rating:.2f} vs {away_mgr_rating:.2f}).")
        elif away_mgr_rating > home_mgr_rating + 0.15:
            explanations.append(f"Tactical edge goes to {away_team} under manager {away_mgr_info.get('manager')} (rating {away_mgr_rating:.2f} vs {home_mgr_rating:.2f}).")

        # Key players info
        home_players = pred_res["output"]["player_prediction"]["home_team"]["goal"]
        away_players = pred_res["output"]["player_prediction"]["away_team"]["goal"]
        if home_players:
            explanations.append(f"Goalscorer threat: {home_players[0]['name']} is predicted to lead {home_team}'s attack ({home_players[0]['predictions'][0]['probability']}% scoring chance).")
        if away_players:
            explanations.append(f"Goalscorer threat: {away_players[0]['name']} is predicted to lead {away_team}'s attack ({away_players[0]['predictions'][0]['probability']}% scoring chance).")

        # Overall confidence
        probs = pred_res["output"]["match_prediction"]["win_probabilities"]
        home_p = probs["home_team"]["probability"]
        away_p = probs["away_team"]["probability"]
        draw_p = probs["draw"]["probability"]
        max_p = max(home_p, away_p, draw_p)

        if max_p >= 60:
            confidence = "Very High"
        elif max_p >= 48:
            confidence = "High"
        elif max_p >= 38:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Determine winner from the predicted scoreline to ensure mathematical consistency
        pred_h = pred_res["output"]["score_prediction"]["predicted_scoreline"]["home_goals"]
        pred_a = pred_res["output"]["score_prediction"]["predicted_scoreline"]["away_goals"]
        if pred_h > pred_a:
            winner = home_team
        elif pred_h < pred_a:
            winner = away_team
        else:
            # If it's a draw, check if a shootout occurred (for knockout matches)
            shootout = pred_res["output"].get("penalty_shootout", {})
            if shootout.get("show_shootout"):
                winner = shootout.get("predicted_winner", "Draw")
            else:
                winner = "Draw"

        h2h_count = len(h2h_matches)
        home_h2h_wins = sum(1 for m in h2h_matches if (m["home_team"] == home_team and m["home_score"] > m["away_score"]) or (m["away_team"] == home_team and m["away_score"] > m["home_score"]))
        away_h2h_wins = sum(1 for m in h2h_matches if (m["home_team"] == away_team and m["home_score"] > m["away_score"]) or (m["away_team"] == away_team and m["away_score"] > m["home_score"]))
        h2h_draws = h2h_count - home_h2h_wins - away_h2h_wins

        analytics = {
            "confidence": confidence,
            "winner": winner,
            "home_stats": {
                "fifa_rank": home_rank or "N/A",
                "elo": int(home_elo),
                "manager": home_manager or "N/A",
                "manager_win_pct": f"{home_mgr_info.get('win_pct', 0.5) * 100:.1f}%",
                "manager_loss_pct": f"{home_mgr_info.get('loss_pct', 0.25) * 100:.1f}%",
                "manager_avg_scored": home_mgr_info.get("avg_scored", 1.5),
                "manager_avg_conceded": home_mgr_info.get("avg_conceded", 1.5),
                "manager_rating": home_mgr_info.get("rating", 1.0),
                "matches": home_ts.matches,
                "goals_for": home_ts.goals_for,
                "goals_against": home_ts.goals_against,
                "recent_form_gf": round(home_form_gf, 2),
                "recent_form_ga": round(home_form_ga, 2),
                "wc_matches": home_wc_stats.get("matches", 0),
                "wc_goals_for": home_wc_stats.get("goals_for", 0),
                "wc_goals_against": home_wc_stats.get("goals_against", 0)
            },
            "away_stats": {
                "fifa_rank": away_rank or "N/A",
                "elo": int(away_elo),
                "manager": away_manager or "N/A",
                "manager_win_pct": f"{away_mgr_info.get('win_pct', 0.5) * 100:.1f}%",
                "manager_loss_pct": f"{away_mgr_info.get('loss_pct', 0.25) * 100:.1f}%",
                "manager_avg_scored": away_mgr_info.get("avg_scored", 1.5),
                "manager_avg_conceded": away_mgr_info.get("avg_conceded", 1.5),
                "manager_rating": away_mgr_info.get("rating", 1.0),
                "matches": away_ts.matches,
                "goals_for": away_ts.goals_for,
                "goals_against": away_ts.goals_against,
                "recent_form_gf": round(away_form_gf, 2),
                "recent_form_ga": round(away_form_ga, 2),
                "wc_matches": away_wc_stats.get("matches", 0),
                "wc_goals_for": away_wc_stats.get("goals_for", 0),
                "wc_goals_against": away_wc_stats.get("goals_against", 0)
            },
            "comparisons": {
                "elo_diff": elo_diff,
                "h2h_count": h2h_count,
                "h2h_home_wins": home_h2h_wins,
                "h2h_away_wins": away_h2h_wins,
                "h2h_draws": h2h_draws,
                "h2h_list": h2h_list
            },
            "explanations": explanations
        }

        return {
            "status": "success",
            "prediction": pred_res["output"],
            "analytics": analytics
        }


def run_web_server(model, port=5000):
    """Start light Flask REST API server and launch browser."""
    # pyrefly: ignore [missing-import]
    from flask import Flask, request, jsonify, render_template, send_from_directory
    import webbrowser
    import threading

    app = Flask(__name__, template_folder="templates", static_folder="static")
    service = PredictionService(model)

    @app.route("/")
    def index():
        return render_template("index.html", active_page="home")

    @app.route("/documentation")
    def documentation():
        return render_template("documentation.html", active_page="documentation")

    @app.route("/api/teams", methods=["GET"])
    def get_teams():
        return jsonify({"teams": service.get_available_teams()})

    @app.route("/api/predict", methods=["POST"])
    def predict():
        data = request.get_json() or {}
        home_team = data.get("home_team", "").strip()
        away_team = data.get("away_team", "").strip()
        # stage is inferred automatically by the engine; ignore any client-supplied value
        res = service.generate_prediction(home_team, away_team)
        if res.get("status") == "error":
            return jsonify(res), 400
        return jsonify(res)

    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://127.0.0.1:{port}/")

    threading.Thread(target=open_browser, daemon=True).start()

    print(f"Starting GoalGPT Web Server on http://127.0.0.1:{port}...", file=sys.stderr)
    app.run(host="127.0.0.1", port=port, debug=False)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GoalGPT Football Prediction AI")
    parser.add_argument("--input",      type=str, default=None,
                        help='JSON string: \'{"home_team":"Argentina","away_team":"Brazil"}\'')
    parser.add_argument("--results",    type=str, default=None, help="Path to results CSV")
    parser.add_argument("--goalscorers",type=str, default=None, help="Path to goalscorers CSV")
    parser.add_argument("--players",    type=str, default=None, help="Path to players stats CSV")
    parser.add_argument("--current-players", type=str, default=None, help="Path to players.csv roster")
    parser.add_argument("--manager-data",    type=str, default=None, help="Path to Manager_dataset.csv")
    parser.add_argument("--wc-teams",        type=str, default=None, help="Path to Worldcup_2026_teams.csv")
    parser.add_argument("--wc-squads",       type=str, default=None, help="Path to Worldcup_2026_squads_and_players.csv")
    parser.add_argument("--wc-matches",      type=str, default=None, help="Path to Worldcup_2026_matches_until_now.csv")
    parser.add_argument("--wc-round-of-32",  type=str, default=None, help="Path to Worldcup_2026_round_of_32.csv")
    parser.add_argument("--wc-round-of-16",  type=str, default=None, help="Path to Worldcup_2026_round_of_16.csv")
    parser.add_argument("--save-model", type=str, default=None, metavar="FILE",
                        help="Train and save model to FILE (.pkl)")
    parser.add_argument("--load-model", type=str, default=None, metavar="FILE",
                        help="Load pre-trained model from FILE (.pkl)")
    parser.add_argument("--web",        action="store_true", help="Start web interface")
    parser.add_argument("--serve",      action="store_true", help="Start web interface (alias)")
    parser.add_argument("--port",       type=int, default=5000, help="Web server port")
    # If running in Jupyter/Colab, ignore Jupyter's internal kernel arguments
    is_jupyter = any("ipykernel" in arg or "-f" in arg for arg in sys.argv)
    if is_jupyter:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    # ── Model load/train ──────────────────────────────────────────────────────
    _cwd = Path(".")

    if args.load_model:
        pkl_path = Path(args.load_model)
        if not pkl_path.exists():
            print(json.dumps({"error": f"Model file not found: {pkl_path}"}))
            sys.exit(1)
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {pkl_path}.", file=sys.stderr)
        # Refresh manager data from CSV only if requested or if no serialized records exist
        if args.manager_data or not getattr(model, "manager_impact", None) or not getattr(model.manager_impact, "records", None):
            _mgr_path = Path(args.manager_data) if args.manager_data else Path("DataSet/Manager_dataset.csv")
            model.manager_impact = ManagerImpact(_mgr_path)

    elif not args.save_model and (auto_pkl := _resolve_auto_pkl(_cwd)) is not None:
        # Auto-load the latest available model (GoalGPT_latest.pkl or highest version)
        with open(auto_pkl, "rb") as f:
            model = pickle.load(f)
        print(f"Auto-loaded {auto_pkl.name}.", file=sys.stderr)
        # Refresh manager data from CSV only if requested or if no serialized records exist
        if args.manager_data or not getattr(model, "manager_impact", None) or not getattr(model.manager_impact, "records", None):
            _mgr_path = Path(args.manager_data) if args.manager_data else Path("DataSet/Manager_dataset.csv")
            model.manager_impact = ManagerImpact(_mgr_path)

    else:
        t0 = time.time()
        model = FootballPredictionModel(
            results_path=args.results,
            goalscorers_path=args.goalscorers,
            players_path=args.players,
            current_players_path=args.current_players,
            manager_path=args.manager_data,
            worldcup_teams_path=args.wc_teams,
            worldcup_squads_path=args.wc_squads,
            worldcup_matches_path=args.wc_matches,
            worldcup_round_of_32_path=args.wc_round_of_32,
            worldcup_round_of_16_path=args.wc_round_of_16,
        )
        # Training must complete before any file is written
        try:
            model.train()
        except Exception as exc:
            print(json.dumps({"error": "Training failed — active model unchanged.",
                              "details": str(exc)}), file=sys.stderr)
            sys.exit(1)
        duration = round(time.time() - t0, 2)

        # Register the module so cloudpickle serializes classes and functions by value
        if 'sports_prediction_colab_app' in sys.modules:
            cloudpickle.register_pickle_by_value(sys.modules['sports_prediction_colab_app'])

        if args.save_model:
            # User supplied an explicit path — save there, skip auto-versioning
            pkl_path = Path(args.save_model)
            with open(pkl_path, "wb") as f:
                cloudpickle.dump(model, f)
            print(f"Model saved to {pkl_path}.", file=sys.stderr)
        else:
            # Auto-versioning: detect next version, save, update the stable pointer
            next_v   = _next_version(_cwd)
            pkl_path = _cwd / f"GoalGPT_version_{next_v}.pkl"
            with open(pkl_path, "wb") as f:
                cloudpickle.dump(model, f)
            print(f"Model saved to {pkl_path.name}.", file=sys.stderr)

            # Atomically replace the stable pointer so prediction always loads latest
            latest_ptr = _cwd / "GoalGPT_latest.pkl"
            shutil.copy2(pkl_path, latest_ptr)
            print(f"GoalGPT_latest.pkl updated → {pkl_path.name}.", file=sys.stderr)

        # Experiment log
        log = {
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "version":          next_v if not args.save_model else None,
            "training_matches": model.total_matches,
            "teams_tracked":    len(model.team_stats),
            "elo_k_factor":     32,
            "xg_blend":         {"attack": 0.56, "defence": 0.34, "average": 0.10},
            "h2h_blend":        {"statistical": 0.82, "h2h": 0.18},
            "ml_blend":         {"ml": 0.68, "poisson": 0.32},
            "ml_enabled":       HAS_ML and model.ml_winner_model is not None,
            "training_seconds": duration,
            "model_file":       str(pkl_path),
        }
        log_path = pkl_path.parent / "experiment_log.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Experiment log written to {log_path}.", file=sys.stderr)

    # ── Web Mode Execution ───────────────────────────────────────────────────
    if args.web or args.serve:
        run_web_server(model, port=args.port)
        sys.exit(0)

    # ── Resolve teams ─────────────────────────────────────────────────────────
    if args.input:
        try:
            data = json.loads(args.input)
            if "home_team" not in data or "away_team" not in data:
                raise KeyError("missing home_team or away_team")
            home_team = str(data["home_team"]).strip()
            away_team = str(data["away_team"]).strip()
        except (json.JSONDecodeError, KeyError) as e:
            print(json.dumps({
                "error":   "Invalid --input format.",
                "details": str(e),
                "example": '{"home_team":"Argentina","away_team":"Brazil"}',
            }, indent=2))
            sys.exit(1)
    else:
        # Resolve participating teams from Worldcup_2026_teams.csv (active tournament list)
        wc_teams = model.available_teams()
        
        home_team = interactive_select_team("Select Home Team", wc_teams)
        away_options = [t for t in wc_teams if t != home_team]
        away_team = interactive_select_team("Select Away Team", away_options)

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        result = model.predict(home_team, away_team)
        print(json.dumps(result, indent=3))
    except ValueError as e:
        # Structured validation error (team not found)
        try:
            detail = json.loads(str(e))
        except Exception:
            detail = {"message": str(e)}
        print(json.dumps({"error": "Prediction failed", **detail}, indent=2))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": "Unexpected error", "details": str(e)}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
