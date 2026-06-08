"""
IPL Synthetic Dataset Generator
Generates realistic IPL player stats + auction price data for 2018-2023
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ─── Player Pool ─────────────────────────────────────────────────────────────
batsmen = [
    ("Virat Kohli","India","Batsman",35),("Rohit Sharma","India","Batsman",36),
    ("KL Rahul","India","Batsman",32),("Shubman Gill","India","Batsman",24),
    ("Faf du Plessis","South Africa","Batsman",39),("David Warner","Australia","Batsman",37),
    ("Jos Buttler","England","WK-Batsman",33),("Quinton de Kock","South Africa","WK-Batsman",31),
    ("Rishabh Pant","India","WK-Batsman",26),("Ishan Kishan","India","WK-Batsman",25),
    ("Shreyas Iyer","India","Batsman",29),("Suryakumar Yadav","India","Batsman",33),
    ("Tilak Varma","India","Batsman",21),("Ruturaj Gaikwad","India","Batsman",27),
    ("Devdutt Padikkal","India","Batsman",24),("Prithvi Shaw","India","Batsman",24),
    ("Manish Pandey","India","Batsman",34),("Ambati Rayudu","India","Batsman",38),
    ("Robin Uthappa","India","Batsman",38),("Ajinkya Rahane","India","Batsman",35),
]
allrounders = [
    ("Hardik Pandya","India","All-Rounder",30),("Ravindra Jadeja","India","All-Rounder",35),
    ("Axar Patel","India","All-Rounder",30),("Washington Sundar","India","All-Rounder",24),
    ("Glenn Maxwell","Australia","All-Rounder",35),("Marcus Stoinis","Australia","All-Rounder",34),
    ("Sam Curran","England","All-Rounder",25),("Ben Stokes","England","All-Rounder",32),
    ("Andre Russell","West Indies","All-Rounder",36),("Kieron Pollard","West Indies","All-Rounder",36),
    ("Shakib Al Hasan","Bangladesh","All-Rounder",36),("Shahrukh Khan","India","All-Rounder",28),
    ("Deepak Hooda","India","All-Rounder",29),
]
bowlers = [
    ("Jasprit Bumrah","India","Bowler",30),("Mohammed Shami","India","Bowler",33),
    ("Yuzvendra Chahal","India","Bowler",33),("Ravichandran Ashwin","India","Bowler",37),
    ("Bhuvneshwar Kumar","India","Bowler",34),("Trent Boult","New Zealand","Bowler",34),
    ("Pat Cummins","Australia","Bowler",30),("Mitchell Starc","Australia","Bowler",34),
    ("Kagiso Rabada","South Africa","Bowler",28),("Anrich Nortje","South Africa","Bowler",30),
    ("Rashid Khan","Afghanistan","Bowler",25),("Mohammad Nabi","Afghanistan","Bowler",38),
    ("Dwayne Bravo","West Indies","Bowler",40),("Umesh Yadav","India","Bowler",35),
    ("Shardul Thakur","India","Bowler",32),("Deepak Chahar","India","Bowler",31),
    ("Arshdeep Singh","India","Bowler",24),("T Natarajan","India","Bowler",32),
    ("Avesh Khan","India","Bowler",27),
]

all_players = batsmen + allrounders + bowlers
seasons = [2018, 2019, 2020, 2021, 2022, 2023]
teams = ["MI","CSK","RCB","KKR","DC","PBKS","RR","SRH","GT","LSG"]

# ─── Generate Season Stats ────────────────────────────────────────────────────
rows = []
for season in seasons:
    for name, country, role, age in all_players:
        player_age = age - (2023 - season)
        matches = np.random.randint(8, 16)

        if role in ["Batsman", "WK-Batsman"]:
            runs     = int(np.random.normal(380, 120))
            sr       = round(np.random.normal(140, 18), 1)
            avg      = round(np.random.normal(38, 10), 1)
            fifties  = np.random.randint(1, 5)
            hundreds = np.random.randint(0, 2)
            wickets  = 0; economy = 0.0; bowl_avg = 0.0
        elif role == "Bowler":
            runs     = int(np.random.normal(80, 40))
            sr       = round(np.random.normal(100, 15), 1)
            avg      = round(np.random.normal(15, 8), 1)
            fifties  = 0; hundreds = 0
            wickets  = np.random.randint(8, 22)
            economy  = round(np.random.normal(8.2, 1.2), 2)
            bowl_avg = round(np.random.normal(24, 7), 1)
        else:  # All-Rounder
            runs     = int(np.random.normal(250, 80))
            sr       = round(np.random.normal(145, 20), 1)
            avg      = round(np.random.normal(28, 8), 1)
            fifties  = np.random.randint(0, 3); hundreds = 0
            wickets  = np.random.randint(5, 16)
            economy  = round(np.random.normal(8.8, 1.0), 2)
            bowl_avg = round(np.random.normal(28, 8), 1)

        runs    = max(runs, 0); sr  = max(sr, 50)
        avg     = max(avg, 5);  wickets = max(wickets, 0)
        economy = max(economy, 5.0)

        rows.append({
            "player_name": name, "country": country, "role": role,
            "age": player_age, "season": season, "team": np.random.choice(teams),
            "matches": matches, "runs": runs, "strike_rate": sr,
            "batting_avg": avg, "fifties": fifties, "hundreds": hundreds,
            "wickets": wickets, "economy_rate": economy, "bowling_avg": bowl_avg,
        })

stats_df = pd.DataFrame(rows)

# ─── Generate Auction Prices ──────────────────────────────────────────────────
auction_rows = []
for name, country, role, age in all_players:
    base = {"Batsman": 50, "WK-Batsman": 75, "All-Rounder": 100, "Bowler": 50}[role]
    star_players = ["Virat Kohli", "Rohit Sharma", "Jasprit Bumrah",
                    "Hardik Pandya", "Andre Russell", "Jos Buttler", "Rashid Khan"]
    is_star = 1 if name in star_players else 0
    fame_bonus = 400 if is_star else 0
    age_factor = max(0.5, 1.2 - (age - 25) * 0.03)
    noise      = np.random.normal(0, 30)
    price_cr   = round(max(20, base * age_factor + fame_bonus + noise), 2)
    auction_rows.append({
        "player_name": name, "role": role, "country": country,
        "age": age, "base_price_lakhs": base * 10,
        "sold_price_cr": price_cr, "is_star": is_star
    })

auction_df = pd.DataFrame(auction_rows)

# ─── Save ─────────────────────────────────────────────────────────────────────
os.makedirs("data/raw", exist_ok=True)
stats_df.to_csv("data/raw/ipl_player_stats.csv", index=False)
auction_df.to_csv("data/raw/ipl_auction_data.csv", index=False)
print(f"✅ Stats dataset   : {stats_df.shape[0]} rows x {stats_df.shape[1]} cols")
print(f"✅ Auction dataset : {auction_df.shape[0]} rows x {auction_df.shape[1]} cols")
print("✅ CSVs saved to data/raw/")
