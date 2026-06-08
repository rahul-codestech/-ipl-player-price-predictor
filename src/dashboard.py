"""
Step 4: Final Summary Dashboard
Creates a single publication-ready figure combining all key insights
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder

df    = pd.read_csv("data/processed/ipl_features.csv")
stats = pd.read_csv("data/raw/ipl_player_stats.csv")

with open("outputs/models/gb_auction_predictor.pkl", "rb") as f:
    saved = pickle.load(f)
model      = saved["model"]
FEATURES   = saved["features"]
le_role    = saved["le_role"]
le_country = saved["le_country"]

df["role_enc"]    = le_role.transform(df["role"].fillna("Batsman"))
df["country_enc"] = le_country.transform(df["country"].fillna("India"))
for feat in FEATURES:
    if feat in df.columns:
        df[feat] = df[feat].fillna(df[feat].median())

df["predicted_price"] = model.predict(df[FEATURES])
df["value_gap"]       = df["sold_price_cr"] - df["predicted_price"]

COLORS = {"Batsman":"#e74c3c","WK-Batsman":"#e67e22",
          "All-Rounder":"#2ecc71","Bowler":"#3498db"}

# ── Big Dashboard Figure ───────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("#0f1117")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

title_kw = dict(fontsize=11, fontweight='bold', color='white', pad=8)
label_kw = dict(color='#aaaaaa', fontsize=8.5)

def style_ax(ax):
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    ax.spines[:].set_color("#333344")
    ax.xaxis.label.set_color('#aaaaaa')
    ax.yaxis.label.set_color('#aaaaaa')

# ── 1) Top Run Scorers ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
run_agg = stats.groupby("player_name")["runs"].sum().nlargest(8).reset_index()
bars = ax1.barh(run_agg["player_name"], run_agg["runs"],
                color="#e74c3c", edgecolor='#0f1117', linewidth=0.8)
ax1.set_title("🏏 Top Run Scorers", **title_kw)
ax1.invert_yaxis()
style_ax(ax1)
for bar, val in zip(bars, run_agg["runs"]):
    ax1.text(val+20, bar.get_y()+bar.get_height()/2, str(val),
             va='center', fontsize=7.5, color='white')

# ── 2) Top Wicket Takers ────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
wkt_agg = stats[stats["wickets"]>0].groupby("player_name")["wickets"].sum().nlargest(8).reset_index()
ax2.barh(wkt_agg["player_name"], wkt_agg["wickets"],
         color="#3498db", edgecolor='#0f1117', linewidth=0.8)
ax2.set_title("🎯 Top Wicket Takers", **title_kw)
ax2.invert_yaxis(); style_ax(ax2)

# ── 3) Impact Score Leaderboard ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
top_impact = df.nlargest(8, "impact_score_norm")[["player_name","impact_score_norm","role"]]
bar_colors = [COLORS.get(r, "#95a5a6") for r in top_impact["role"]]
ax3.barh(top_impact["player_name"], top_impact["impact_score_norm"],
         color=bar_colors, edgecolor='#0f1117', linewidth=0.8)
ax3.set_title("⚡ Impact Score (0–100)", **title_kw)
ax3.invert_yaxis(); style_ax(ax3)

# ── 4) Actual vs Predicted ──────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0:2])
for role, grp in df.groupby("role"):
    ax4.scatter(grp["sold_price_cr"], grp["predicted_price"],
                label=role, color=COLORS.get(role,"#95a5a6"),
                s=70, alpha=0.85, edgecolors='#0f1117', linewidth=0.5)
lim = [0, df["sold_price_cr"].max() + 30]
ax4.plot(lim, lim, '--', color='#555566', linewidth=1.5, label="Perfect Prediction")
ax4.set_title("🤖 Actual vs Predicted Auction Price  |  R² = 0.91", **title_kw)
ax4.set_xlabel("Actual Price (Cr)", **label_kw)
ax4.set_ylabel("Predicted Price (Cr)", **label_kw)
ax4.legend(fontsize=8, facecolor="#1a1d27", labelcolor='white',
           edgecolor='#333344', loc='upper left')
style_ax(ax4)
for _, row in df[df["is_star"]==1].iterrows():
    ax4.annotate(row["player_name"],
                 (row["sold_price_cr"], row["predicted_price"]),
                 fontsize=6.5, color='#dddddd',
                 xytext=(4, 4), textcoords='offset points')

# ── 5) Feature Importance ───────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
fi = pd.DataFrame({"Feature": FEATURES,
                   "Importance": model.feature_importances_}).sort_values("Importance")
fi_colors = ["#f39c12" if v > 0.1 else "#3498db" for v in fi["Importance"]]
ax5.barh(fi["Feature"], fi["Importance"], color=fi_colors, edgecolor='#0f1117', linewidth=0.5)
ax5.set_title("📊 Feature Importance", **title_kw)
style_ax(ax5)

# ── 6) Overvalued ───────────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0])
ov = df.nlargest(6, "value_gap")[["player_name","value_gap"]]
ax6.barh(ov["player_name"], ov["value_gap"], color="#e74c3c", edgecolor='#0f1117')
ax6.set_title("💸 Most Overvalued", **title_kw)
ax6.invert_yaxis(); style_ax(ax6)
ax6.set_xlabel("Overpaid by (Cr)", **label_kw)

# ── 7) Undervalued ──────────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 1])
uv = df.nsmallest(6, "value_gap")[["player_name","value_gap"]].copy()
uv["value_gap"] = uv["value_gap"].abs()
ax7.barh(uv["player_name"], uv["value_gap"], color="#2ecc71", edgecolor='#0f1117')
ax7.set_title("💎 Most Undervalued", **title_kw)
ax7.invert_yaxis(); style_ax(ax7)
ax7.set_xlabel("Underpaid by (Cr)", **label_kw)

# ── 8) Season Trend ─────────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
trend = stats.groupby("season")["runs"].mean().reset_index()
ax8.plot(trend["season"], trend["runs"], marker='o', linewidth=2,
         color="#f39c12", markersize=7, markerfacecolor='white', markeredgewidth=2)
ax8.fill_between(trend["season"], trend["runs"], alpha=0.15, color="#f39c12")
ax8.set_title("📈 Season Avg Runs Trend", **title_kw)
ax8.set_xticks(trend["season"]); style_ax(ax8)

# ── Main Title ───────────────────────────────────────────────────────────────
fig.text(0.5, 0.98,
         "IPL Player Performance & Auction Value Predictor",
         ha='center', va='top', fontsize=17, fontweight='bold', color='white')
fig.text(0.5, 0.955,
         "End-to-End Data Analytics Project  |  Python · Scikit-learn · Gradient Boosting  |  R² = 0.91",
         ha='center', va='top', fontsize=10, color='#aaaaaa')

plt.savefig("outputs/plots/00_DASHBOARD.png", dpi=160, bbox_inches='tight',
            facecolor="#0f1117")
plt.close()
print("✅ Final dashboard saved: outputs/plots/00_DASHBOARD.png")
