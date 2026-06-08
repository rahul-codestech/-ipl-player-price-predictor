"""
Step 2: EDA + Feature Engineering
- Exploratory analysis with visualizations
- Custom Impact Score metric
- Form Index (weighted recent performance)
- Saves processed dataset
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os, warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────────────────
stats   = pd.read_csv("data/raw/ipl_player_stats.csv")
auction = pd.read_csv("data/raw/ipl_auction_data.csv")

print("="*55)
print("📊 IPL Player Analytics — EDA & Feature Engineering")
print("="*55)
print(f"\nStats shape   : {stats.shape}")
print(f"Auction shape : {auction.shape}")
print("\n--- Stats Sample ---")
print(stats.head(3).to_string(index=False))

# ── Plot 1: Top Run-Scorers (career avg) ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("IPL Player Analytics — Exploratory Analysis", fontsize=16, fontweight='bold', y=1.01)

# 1a) Top 10 run scorers
run_agg = stats.groupby("player_name")["runs"].sum().nlargest(10).reset_index()
sns.barplot(data=run_agg, x="runs", y="player_name", palette="YlOrRd_r", ax=axes[0,0])
axes[0,0].set_title("Top 10 Run Scorers (2018-2023)", fontweight='bold')
axes[0,0].set_xlabel("Total Runs"); axes[0,0].set_ylabel("")
for i, v in enumerate(run_agg["runs"]):
    axes[0,0].text(v + 30, i, str(v), va='center', fontsize=9)

# 1b) Top 10 wicket takers
wkt_agg = stats[stats["wickets"] > 0].groupby("player_name")["wickets"].sum().nlargest(10).reset_index()
sns.barplot(data=wkt_agg, x="wickets", y="player_name", palette="Blues_r", ax=axes[0,1])
axes[0,1].set_title("Top 10 Wicket Takers (2018-2023)", fontweight='bold')
axes[0,1].set_xlabel("Total Wickets"); axes[0,1].set_ylabel("")
for i, v in enumerate(wkt_agg["wickets"]):
    axes[0,1].text(v + 0.3, i, str(v), va='center', fontsize=9)

# 1c) Avg strike rate by role
sr_role = stats.groupby("role")["strike_rate"].mean().reset_index().sort_values("strike_rate", ascending=False)
colors = {"Batsman":"#e74c3c","WK-Batsman":"#e67e22","All-Rounder":"#2ecc71","Bowler":"#3498db"}
bar_colors = [colors.get(r, "#95a5a6") for r in sr_role["role"]]
axes[1,0].bar(sr_role["role"], sr_role["strike_rate"], color=bar_colors, edgecolor='white', linewidth=1.5)
axes[1,0].set_title("Average Strike Rate by Role", fontweight='bold')
axes[1,0].set_xlabel("Role"); axes[1,0].set_ylabel("Strike Rate")
for i, (_, row) in enumerate(sr_role.iterrows()):
    axes[1,0].text(i, row["strike_rate"] + 0.5, f"{row['strike_rate']:.1f}", ha='center', fontsize=9)

# 1d) Season-wise avg runs trend
season_trend = stats.groupby("season")["runs"].mean().reset_index()
axes[1,1].plot(season_trend["season"], season_trend["runs"], marker='o', linewidth=2.5,
               color="#e74c3c", markersize=8, markerfacecolor='white', markeredgewidth=2)
axes[1,1].fill_between(season_trend["season"], season_trend["runs"],
                         alpha=0.15, color="#e74c3c")
axes[1,1].set_title("Season-wise Average Runs per Player", fontweight='bold')
axes[1,1].set_xlabel("Season"); axes[1,1].set_ylabel("Avg Runs")
axes[1,1].set_xticks(season_trend["season"])

plt.tight_layout()
plt.savefig("outputs/plots/01_eda_overview.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Plot saved: outputs/plots/01_eda_overview.png")

# ── Plot 2: Auction Price Analysis ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("IPL Auction Price Analysis", fontsize=14, fontweight='bold')

# 2a) Price distribution by role
for role, grp in auction.groupby("role"):
    axes[0].hist(grp["sold_price_cr"], bins=8, alpha=0.6, label=role,
                 color=colors.get(role, "#95a5a6"), edgecolor='white')
axes[0].set_title("Price Distribution by Role")
axes[0].set_xlabel("Price (Cr)"); axes[0].set_ylabel("Count")
axes[0].legend()

# 2b) Age vs Price scatter
role_palette = {r: colors[r] for r in colors}
for role, grp in auction.groupby("role"):
    axes[1].scatter(grp["age"], grp["sold_price_cr"], label=role,
                    color=colors.get(role, "#95a5a6"), alpha=0.7, s=80, edgecolors='white')
axes[1].set_title("Age vs Auction Price")
axes[1].set_xlabel("Age"); axes[1].set_ylabel("Sold Price (Cr)")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/plots/02_auction_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot saved: outputs/plots/02_auction_analysis.png")

# ── Feature Engineering ───────────────────────────────────────────────────────
print("\n⚙️  Engineering Features...")

# 1) Aggregate last 3 seasons per player
latest = stats[stats["season"] >= 2021].copy()

agg = latest.groupby("player_name").agg(
    matches     = ("matches",      "sum"),
    total_runs  = ("runs",         "sum"),
    avg_sr      = ("strike_rate",  "mean"),
    avg_bat_avg = ("batting_avg",  "mean"),
    total_50s   = ("fifties",      "sum"),
    total_100s  = ("hundreds",     "sum"),
    total_wkts  = ("wickets",      "sum"),
    avg_eco     = ("economy_rate", "mean"),
    avg_bowl_avg= ("bowling_avg",  "mean"),
    seasons_played = ("season",    "count"),
).reset_index()

# 2) Form Index — weighted recent avg (2023=3, 2022=2, 2021=1)
weights = {2021: 1, 2022: 2, 2023: 3}
stats_recent = stats[stats["season"] >= 2021].copy()
stats_recent["weight"] = stats_recent["season"].map(weights)
stats_recent["w_runs"] = stats_recent["runs"] * stats_recent["weight"]
stats_recent["w_wkts"] = stats_recent["wickets"] * stats_recent["weight"]
form = stats_recent.groupby("player_name").apply(
    lambda x: pd.Series({
        "form_runs":    x["w_runs"].sum() / x["weight"].sum(),
        "form_wickets": x["w_wkts"].sum() / x["weight"].sum(),
    })
).reset_index()

agg = agg.merge(form, on="player_name", how="left")

# 3) Custom Impact Score
# Batsmen: runs + SR bonus + milestone bonus
# Bowlers: wickets + economy bonus
# All-rounders: blend

def impact_score(row, role):
    if role in ["Batsman", "WK-Batsman"]:
        score = (row["total_runs"] / 10) + (max(0, row["avg_sr"] - 120) * 0.5) + \
                (row["total_50s"] * 5) + (row["total_100s"] * 15)
    elif role == "Bowler":
        eco_bonus = max(0, (10 - row["avg_eco"]) * 8) if row["avg_eco"] > 0 else 0
        score = (row["total_wkts"] * 8) + eco_bonus
    else:  # All-Rounder
        bat = (row["total_runs"] / 15) + (row["total_50s"] * 4)
        bowl = (row["total_wkts"] * 6)
        score = bat + bowl
    return round(score, 2)

# Merge role
player_roles = stats[["player_name","role"]].drop_duplicates()
agg = agg.merge(player_roles, on="player_name", how="left")
agg["impact_score"] = agg.apply(lambda r: impact_score(r, r["role"]), axis=1)

# 4) Normalize impact score 0-100
_min, _max = agg["impact_score"].min(), agg["impact_score"].max()
agg["impact_score_norm"] = ((agg["impact_score"] - _min) / (_max - _min) * 100).round(2)

# 5) Merge with auction data
final = auction.merge(agg, on="player_name", how="left")
final["role"] = final["role_x"].combine_first(final["role_y"])
final.drop(columns=["role_x","role_y"], inplace=True, errors="ignore")

# Fill NaN for players with no recent stats
for col in ["total_runs","avg_sr","avg_bat_avg","total_wkts","avg_eco","impact_score_norm"]:
    if col in final.columns:
        final[col] = final[col].fillna(final[col].median())

final.to_csv("data/processed/ipl_features.csv", index=False)
print(f"✅ Processed dataset: {final.shape[0]} rows × {final.shape[1]} cols")
print(f"\n📌 Top 5 by Impact Score:")
print(final.nlargest(5,"impact_score_norm")[["player_name","role","impact_score_norm","sold_price_cr"]].to_string(index=False))

# ── Plot 3: Impact Score vs Auction Price ────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
for role, grp in final.groupby("role"):
    ax.scatter(grp["impact_score_norm"], grp["sold_price_cr"],
               label=role, color=colors.get(role,"#95a5a6"),
               s=100, alpha=0.8, edgecolors='white', linewidth=0.8)
    for _, row in grp.nlargest(2,"sold_price_cr").iterrows():
        ax.annotate(row["player_name"],
                    (row["impact_score_norm"], row["sold_price_cr"]),
                    fontsize=7.5, alpha=0.85,
                    xytext=(5, 5), textcoords="offset points")

ax.set_title("Impact Score vs Auction Price — IPL Players", fontsize=14, fontweight='bold')
ax.set_xlabel("Impact Score (0-100)", fontsize=11)
ax.set_ylabel("Sold Price (Cr)", fontsize=11)
ax.legend(title="Role")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("outputs/plots/03_impact_vs_price.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot saved: outputs/plots/03_impact_vs_price.png")
print("\n✅ EDA & Feature Engineering complete!")
