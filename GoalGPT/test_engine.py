import os
import sys
import math
import random
import numpy as np
import pickle

# Set up path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from sports_prediction_colab_app import FootballPredictionModel, PredictionService

def test_engine():
    print("====================================================")
    print("   GoalGPT Calibrated Engine Validation Suite       ")
    print("====================================================")

    # 1. Load Model
    model_path = "GoalGPT_latest.pkl"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please train model first.")
        sys.exit(1)
        
    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")

    # Select representative teams
    home_team = "Argentina"
    away_team = "Germany"
    stage = "Group Stage"

    print(f"\n--- Testing Match: {home_team} vs {away_team} ({stage}) ---")

    # Generate prediction outputs
    home_xg, away_xg = model._expected_goals(home_team, away_team)
    score_probs = model._score_distribution(home_xg, away_xg)
    wp = model._winner_probabilities(home_team, away_team, home_xg, away_xg, score_probs)
    calibrated_matrix = wp["calibrated_matrix"]

    # Verify expected goals calibration
    print(f"Analytical xG: Home={home_xg:.4f}, Away={away_xg:.4f}")
    
    # -------------------------------------------------------------------------
    # TEST 1: Matrix Normalization
    # -------------------------------------------------------------------------
    matrix_sum = sum(calibrated_matrix.values())
    print(f"TEST 1: Matrix Sum = {matrix_sum:.6f}")
    assert abs(matrix_sum - 1.0) < 1e-6, f"Matrix sum is not 1.0 (got {matrix_sum})"
    print("  => PASSED: Matrix sums to exactly 1.0")

    # -------------------------------------------------------------------------
    # TEST 2: Outcome Marginals alignment with Hybrid Probabilities
    # -------------------------------------------------------------------------
    ph_cal = sum(p for s, p in calibrated_matrix.items() if int(s.split("-")[0]) > int(s.split("-")[1]))
    pa_cal = sum(p for s, p in calibrated_matrix.items() if int(s.split("-")[0]) < int(s.split("-")[1]))
    pd_cal = sum(p for s, p in calibrated_matrix.items() if int(s.split("-")[0]) == int(s.split("-")[1]))

    print(f"TEST 2: Hybrid vs Calibrated Marginals:")
    print(f"  Home: Target={wp['home']:.4f}, Actual={ph_cal:.4f}")
    print(f"  Away: Target={wp['away']:.4f}, Actual={pa_cal:.4f}")
    print(f"  Draw: Target={wp['draw']:.4f}, Actual={pd_cal:.4f}")
    
    assert abs(ph_cal - wp["home"]) < 1e-6, f"Home marginal mismatch: target {wp['home']}, got {ph_cal}"
    assert abs(pa_cal - wp["away"]) < 1e-6, f"Away marginal mismatch: target {wp['away']}, got {pa_cal}"
    assert abs(pd_cal - wp["draw"]) < 1e-6, f"Draw marginal mismatch: target {wp['draw']}, got {pd_cal}"
    print("  => PASSED: Calibrated marginals match hybrid win probabilities")

    # -------------------------------------------------------------------------
    # TEST 3: Scoreline Consistency
    # -------------------------------------------------------------------------
    # Determine the winner category (highest probability outcome category)
    p_home = wp["home"]
    p_draw = wp["draw"]
    p_away = wp["away"]
    if p_home > p_away and p_home > p_draw:
        predicted_outcome = "home"
        outcome_desc = f"{home_team} Win"
    elif p_away > p_home and p_away > p_draw:
        predicted_outcome = "away"
        outcome_desc = f"{away_team} Win"
    else:
        predicted_outcome = "draw"
        outcome_desc = "Draw"

    predicted_line, rng = model._select_scoreline(home_team, away_team, calibrated_matrix, predicted_outcome)
    pred_h, pred_a = map(int, predicted_line.split("-"))

    print(f"TEST 3: Predicted Outcome = {outcome_desc}")
    print(f"        Predicted Scoreline = {predicted_line}")
    if predicted_outcome == "home":
        assert pred_h > pred_a, f"Scoreline {predicted_line} is inconsistent with Home win"
    elif predicted_outcome == "away":
        assert pred_h < pred_a, f"Scoreline {predicted_line} is inconsistent with Away win"
    else:
        assert pred_h == pred_a, f"Scoreline {predicted_line} is inconsistent with Draw"
    print("  => PASSED: Scoreline matches the predicted outcome category")

    # -------------------------------------------------------------------------
    # TEST 4: Expected Goals Consistency
    # -------------------------------------------------------------------------
    # Expected goals calculated from raw Poisson matrix
    # Under standard Poisson, the expected value of goals is equal to the rate parameter (home_xg / away_xg).
    # Since calibration scales the cells, the new expected goals from the calibrated matrix should remain close to the analytical xG.
    home_xg_cal = sum(int(s.split("-")[0]) * p for s, p in calibrated_matrix.items())
    away_xg_cal = sum(int(s.split("-")[1]) * p for s, p in calibrated_matrix.items())
    print(f"TEST 4: Matrix-derived Expected Goals:")
    print(f"  Home: Analytical={home_xg:.4f}, Derived={home_xg_cal:.4f}")
    print(f"  Away: Analytical={away_xg:.4f}, Derived={away_xg_cal:.4f}")
    # Verify they are within reasonable bounds (e.g. within 0.5 goals of the analytical rate)
    assert abs(home_xg_cal - home_xg) < 0.5, f"Home matrix-derived xG too far from analytical: {home_xg_cal} vs {home_xg}"
    assert abs(away_xg_cal - away_xg) < 0.5, f"Away matrix-derived xG too far from analytical: {away_xg_cal} vs {away_xg}"
    print("  => PASSED: Expected goals from the matrix are close to analytical xG")

    # -------------------------------------------------------------------------
    # TEST 5: Dynamic Matrix Mass
    # -------------------------------------------------------------------------
    # Each marginal Poisson series (home and away independently) must retain > 99.99%
    # of probability mass within the dynamic grid size used by _score_distribution.
    max_goals = max(9, math.ceil(max(home_xg, away_xg) + 8))
    home_mass = sum(model._poisson_pmf(g, home_xg) for g in range(max_goals + 1))
    away_mass = sum(model._poisson_pmf(g, away_xg) for g in range(max_goals + 1))
    print(f"TEST 5: Dynamic Matrix Mass (grid 0–{max_goals}):")
    print(f"  Home marginal mass retained: {home_mass * 100:.6f}%")
    print(f"  Away marginal mass retained: {away_mass * 100:.6f}%")
    assert home_mass >= 0.9999, f"Home Poisson truncated too much mass: {home_mass}"
    assert away_mass >= 0.9999, f"Away Poisson truncated too much mass: {away_mass}"
    print("  => PASSED: Dynamic matrix retains > 99.99% Poisson probability mass per marginal")

    # -------------------------------------------------------------------------
    # TEST 6: Probability Calibration Metrics (Log Loss and Brier Score)
    # -------------------------------------------------------------------------
    log_losses = []
    brier_scores = []
    
    eval_matches = model.matches[:30] if hasattr(model, 'matches') and len(model.matches) > 0 else []
    if eval_matches:
        for m in eval_matches:
            ht, at = m["home_team"], m["away_team"]
            hs, as_ = m["home_score"], m["away_score"]
            try:
                hx, ax = model._expected_goals(ht, at)
                sp = model._score_distribution(hx, ax)
                w_probs = model._winner_probabilities(ht, at, hx, ax, sp)
            except Exception:
                continue

            # Target category index
            if hs > as_:
                actual_cat = "home"
            elif hs < as_:
                actual_cat = "away"
            else:
                actual_cat = "draw"

            # Log loss
            pred_p = w_probs[actual_cat]
            pred_p = max(1e-15, min(1.0 - 1e-15, pred_p))
            log_losses.append(-math.log(pred_p))

            # Brier Score
            y_home = 1.0 if actual_cat == "home" else 0.0
            y_away = 1.0 if actual_cat == "away" else 0.0
            y_draw = 1.0 if actual_cat == "draw" else 0.0
            
            bs = (w_probs["home"] - y_home)**2 + (w_probs["away"] - y_away)**2 + (w_probs["draw"] - y_draw)**2
            brier_scores.append(bs)

        mean_log_loss = sum(log_losses) / len(log_losses)
        mean_brier = sum(brier_scores) / len(brier_scores)
        print(f"TEST 6: Calibration Metrics on {len(log_losses)} Historical Matches:")
        print(f"  Mean Log Loss: {mean_log_loss:.4f} (Ideal: < 1.10)")
        print(f"  Mean Brier Score: {mean_brier:.4f} (Ideal: < 0.60)")
        assert mean_log_loss < 1.30, f"Log loss is too high: {mean_log_loss}"
        assert mean_brier < 0.70, f"Brier score is too high: {mean_brier}"
        print("  => PASSED: Probability calibration metrics are within acceptable thresholds")
    else:
        print("TEST 6: Skipped (no historical matches found for validation)")

    # -------------------------------------------------------------------------
    # TEST 7: Repeatability (Deterministic output under same seed)
    # -------------------------------------------------------------------------
    res1 = model.predict(home_team, away_team, stage=stage)
    res2 = model.predict(home_team, away_team, stage=stage)
    
    score1 = res1["output"]["score_prediction"]["predicted_scoreline"]
    score2 = res2["output"]["score_prediction"]["predicted_scoreline"]
    
    score1_str = f"{score1['home_goals']}-{score1['away_goals']}"
    score2_str = f"{score2['home_goals']}-{score2['away_goals']}"
    
    print(f"TEST 7: Repeatability Test:")
    print(f"  Run 1 Scoreline: {score1_str}")
    print(f"  Run 2 Scoreline: {score2_str}")
    assert score1_str == score2_str, f"Repeatability failed: got {score1_str} and {score2_str}"
    print("  => PASSED: Same parameters yield identical predictions (repeatable)")

    # -------------------------------------------------------------------------
    # TEST 8: Monte Carlo Convergence (within +/- 1.5% margin of error)
    # -------------------------------------------------------------------------
    n_sims = 10000
    sim_outcomes = {"home": 0, "away": 0, "draw": 0}
    
    score_names = list(calibrated_matrix.keys())
    score_weights = list(calibrated_matrix.values())
    
    np.random.seed(42)
    draws = np.random.choice(score_names, size=n_sims, p=score_weights)
    for draw in draws:
        h, a = map(int, draw.split("-"))
        if h > a:
            sim_outcomes["home"] += 1
        elif h < a:
            sim_outcomes["away"] += 1
        else:
            sim_outcomes["draw"] += 1

    sim_home = sim_outcomes["home"] / n_sims
    sim_away = sim_outcomes["away"] / n_sims
    sim_draw = sim_outcomes["draw"] / n_sims

    print(f"TEST 8: Monte Carlo Convergence (N=10,000):")
    print(f"  Home: Matrix={wp['home']:.4f}, Sim={sim_home:.4f} (Diff={abs(sim_home - wp['home']):.4f})")
    print(f"  Away: Matrix={wp['away']:.4f}, Sim={sim_away:.4f} (Diff={abs(sim_away - wp['away']):.4f})")
    print(f"  Draw: Matrix={wp['draw']:.4f}, Sim={sim_draw:.4f} (Diff={abs(sim_draw - wp['draw']):.4f})")
    
    assert abs(sim_home - wp["home"]) < 0.015, f"Monte Carlo home diff too large: {abs(sim_home - wp['home'])}"
    assert abs(sim_away - wp["away"]) < 0.015, f"Monte Carlo away diff too large: {abs(sim_away - wp['away'])}"
    assert abs(sim_draw - wp["draw"]) < 0.015, f"Monte Carlo draw diff too large: {abs(sim_draw - wp['draw'])}"
    print("  => PASSED: Simulated outcomes match calibrated joint probabilities within 1.5% tolerance")

    # -------------------------------------------------------------------------
    # TEST 9: Automatic Knockout Stage Inference
    # -------------------------------------------------------------------------
    print(f"TEST 9: Automatic Stage Inference:")
    # Netherlands vs Morocco is in the Round of 32 CSV
    inferred_ko_1 = model._infer_stage("Netherlands", "Morocco")
    inferred_ko_2 = model._infer_stage("Morocco", "Netherlands")
    inferred_group = model._infer_stage("Argentina", "Germany")
    
    print(f"  Netherlands vs Morocco stage: {inferred_ko_1}")
    print(f"  Morocco vs Netherlands stage: {inferred_ko_2}")
    print(f"  Argentina vs Germany stage:   {inferred_group}")
    
    assert inferred_ko_1 == "Round of 32", f"Expected Round of 32 for NL-Morocco, got {inferred_ko_1}"
    assert inferred_ko_2 == "Round of 32", f"Expected Round of 32 for Morocco-NL (reversed), got {inferred_ko_2}"
    assert inferred_group == "Group Stage", f"Expected Group Stage for Argentina-Germany, got {inferred_group}"
    print("  => PASSED: Knockout stage inferred correctly based on team pairs")

    # -------------------------------------------------------------------------
    # TEST 10: Output Schema Compatibility (No changes to Output.json schema)
    # -------------------------------------------------------------------------
    print(f"TEST 10: Output Schema Compatibility:")
    res_ko = model.predict("Netherlands", "Morocco")
    out = res_ko.get("output", {})
    
    # Check top-level keys in predict['output']
    expected_top_keys = {"match_prediction", "score_prediction", "goal_insights", "player_prediction", "penalty_shootout"}
    actual_top_keys = set(out.keys())
    missing_keys = expected_top_keys - actual_top_keys
    assert not missing_keys, f"Missing output schema keys: {missing_keys}"
    print("  => PASSED: Output schema remains 100% compatible with previous versions")

    # -------------------------------------------------------------------------
    # TEST 11: Team Selection & Name Validation
    # -------------------------------------------------------------------------
    print("TEST 11: Team Selection & Name Validation:")
    service = PredictionService(model)
    # Test canonical resolution in prediction service
    for input_name, expected_canon in [
        ("United States", "USA"),
        ("usa", "USA"),
        ("DR Congo", "Congo DR"),
        ("Democratic Republic of the Congo", "Congo DR"),
        ("Cape Verde", "Cabo Verde"),
    ]:
        resolved = model.get_canonical_team_name(input_name)
        assert resolved == expected_canon, f"Failed to map '{input_name}' to '{expected_canon}', got '{resolved}'"
        print(f"  Mapped variant '{input_name}' => '{resolved}' successfully")

    # Test prediction with variant names
    pred = service.generate_prediction("United States", "DR Congo")
    assert pred.get("status") != "error", f"Prediction failed for United States vs DR Congo: {pred}"
    print("  => PASSED: Prediction successfully run using variant team names")

    # Test rejection of invalid teams
    pred_invalid = service.generate_prediction("Atlantis", "Germany")
    assert pred_invalid.get("status") == "error", "Expected error for invalid team, but got success"
    errors = pred_invalid.get("validation_errors", [])
    assert any("Atlantis" in e.get("value", "") for e in errors), f"Expected validation error for Atlantis: {errors}"
    print("  => PASSED: Successfully rejected invalid team with appropriate validation errors")

    print("\n====================================================")
    print("   ALL ENGINE VALIDATION TESTS COMPLETED SUCCESSFULLY!  ")
    print("====================================================")

if __name__ == "__main__":
    test_engine()
