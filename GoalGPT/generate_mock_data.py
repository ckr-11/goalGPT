import csv
import random
from datetime import datetime, timedelta

# List of teams and their strength parameters
TEAMS = {
    "Argentina": {"attack": 2.0, "defence": 0.8},
    "Brazil": {"attack": 1.9, "defence": 0.9},
    "France": {"attack": 2.1, "defence": 0.9},
    "England": {"attack": 1.8, "defence": 0.8},
    "Germany": {"attack": 1.7, "defence": 1.1},
    "Spain": {"attack": 1.7, "defence": 0.8},
    "Italy": {"attack": 1.3, "defence": 0.9},
    "Croatia": {"attack": 1.3, "defence": 1.0},
    "Netherlands": {"attack": 1.6, "defence": 1.2},
    "Portugal": {"attack": 1.8, "defence": 1.0}
}

# Players list with goal shares (must sum to 1.0 per team)
PLAYERS = {
    "Argentina": [("Messi", 0.40), ("Alvarez", 0.25), ("Martinez", 0.25), ("Di Maria", 0.10)],
    "Brazil": [("Neymar", 0.35), ("Vinicius", 0.30), ("Rodrygo", 0.20), ("Raphinha", 0.15)],
    "France": [("Mbappe", 0.50), ("Giroud", 0.25), ("Griezmann", 0.20), ("Dembele", 0.05)],
    "England": [("Kane", 0.55), ("Saka", 0.20), ("Bellingham", 0.15), ("Foden", 0.10)],
    "Germany": [("Fullkrug", 0.30), ("Musiala", 0.25), ("Wirtz", 0.25), ("Havertz", 0.20)],
    "Spain": [("Morata", 0.35), ("Olmo", 0.25), ("Williams", 0.20), ("Yamal", 0.20)],
    "Italy": [("Chiesa", 0.30), ("Retegui", 0.30), ("Barella", 0.20), ("Scamacca", 0.20)],
    "Croatia": [("Kramaric", 0.35), ("Modric", 0.25), ("Perisic", 0.20), ("Pasalic", 0.20)],
    "Netherlands": [("Depay", 0.40), ("Gakpo", 0.30), ("Malen", 0.15), ("Simons", 0.15)],
    "Portugal": [("Ronaldo", 0.45), ("Fernandes", 0.25), ("Ramos", 0.20), ("Leao", 0.10)]
}

def poisson_random(lam):
    L = 2.718281828459045 ** (-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1

def select_scorer(team):
    players = PLAYERS[team]
    r = random.random()
    cumulative = 0.0
    for name, share in players:
        cumulative += share
        if r <= cumulative:
            return name
    return players[-1][0]

def generate_data():
    results = []
    goalscorers = []
    
    start_date = datetime(2020, 1, 1)
    team_names = list(TEAMS.keys())
    
    # Generate around 400 matches
    random.seed(42)  # Seed for reproducibility
    
    current_date = start_date
    for match_id in range(400):
        # Move date forward slightly
        current_date += timedelta(days=random.randint(3, 7))
        
        home_team = random.choice(team_names)
        away_team = random.choice(team_names)
        while away_team == home_team:
            away_team = random.choice(team_names)
            
        home_stats = TEAMS[home_team]
        away_stats = TEAMS[away_team]
        
        # Calculate lambda (expected goals)
        # Home advantage multiplier of 1.15
        home_lambda = 1.15 * home_stats["attack"] * away_stats["defence"] * 0.7
        away_lambda = away_stats["attack"] * home_stats["defence"] * 0.7
        
        home_score = poisson_random(home_lambda)
        away_score = poisson_random(away_lambda)
        
        date_str = current_date.strftime("%Y-%m-%d")
        results.append({
            "date": date_str,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score
        })
        
        # Generate goals/scorers
        # Home team goals
        for _ in range(home_score):
            scorer = select_scorer(home_team)
            minute = random.randint(1, 90)
            goalscorers.append({
                "date": date_str,
                "home_team": home_team,
                "away_team": away_team,
                "team": home_team,
                "scorer": scorer,
                "minute": minute
            })
            
        # Away team goals
        for _ in range(away_score):
            scorer = select_scorer(away_team)
            minute = random.randint(1, 90)
            goalscorers.append({
                "date": date_str,
                "home_team": home_team,
                "away_team": away_team,
                "team": away_team,
                "scorer": scorer,
                "minute": minute
            })

    # Write results.csv
    with open("results.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "home_team", "away_team", "home_score", "away_score"])
        writer.writeheader()
        writer.writerows(results)
        
    # Write goalscorers.csv
    with open("goalscorers.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "home_team", "away_team", "team", "scorer", "minute"])
        writer.writeheader()
        writer.writerows(goalscorers)
        
    print(f"Generated {len(results)} matches in results.csv")
    print(f"Generated {len(goalscorers)} goals in goalscorers.csv")

if __name__ == "__main__":
    generate_data()
