"""
Step 3: ML Model — IPL Auction Price Predictor
- Random Forest + Gradient Boosting
- Cross-validation, R², RMSE
- Feature Importance visualization
- Saves trained model
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model     import LinearRegression, Ridge
from sklearn.model_selection  import cross_val_score, train_test_split
from sklearn.preprocessing    import LabelEncoder
from sklearn.metrics          import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

# ── Load processed data ────────────────────────────────────────────────────
df = pd.read_csv("data/processed/ipl_features.csv")
print("="*55)
print("🤖  IPL Auction Price — ML Model Training")
print("="*55)
print(f"\nDataset shape: {df.shape}")

# ── Encode Categorical ─────────────────────────────────────────────────────
le_role    = LabelEncoder()
le_country = LabelEncoder()
df["role_enc"]    = le_role.fit_transform(df["role"].fillna("Batsman"))
df["country_enc"] = le_country.fit_transform(df["country"].fillna("India"))

FEATURES = [
    "age", "role_enc", "country_enc",
    "base_price_lakhs", "is_star",
    "total_runs", "avg_sr", "avg_bat_avg",
    "total_50s", "total_100s",
    "total_wkts", "avg_eco",
    "form_runs", "form_wickets",
    "impact_score_norm", "seasons_played",
]
TARGET = "sold_price_cr"

# Fill any remaining NaN
for f in FEATURES:
    if f in df.columns:
        df[f] = df[f].fillna(df[f].median())

X = df[FEATURES]
y = df[TARGET]

# ── Train / Test Split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ── Models ──────────────────────────────────────────────────────────────────
models = {
    "Linear Regression"   : LinearRegression(),
    "Ridge Regression"    : Ridge(alpha=10),
    "Random Forest"       : RandomForestRegressor(n_estimators=200, max_depth=6,
                                                   min_samples_leaf=2, random_state=42),
    "Gradient Boosting"   : GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                        max_depth=4, random_state=42),
}

results = []
print("\n--- Cross-Validation Results (5-Fold) ---")
for name, model in models.items():
    cv_r2   = cross_val_score(model, X, y, cv=5, scoring="r2")
    cv_rmse = cross_val_score(model, X, y, cv=5,
                              scoring="neg_root_mean_squared_error")
    results.append({
        "Model"   : name,
        "CV R²"   : round(cv_r2.mean(), 4),
        "CV R² SD": round(cv_r2.std(), 4),
        "CV RMSE" : round(-cv_rmse.mean(), 2),
    })
    print(f"  {name:<22}  R²={cv_r2.mean():.3f} ± {cv_r2.std():.3f}  |  RMSE={-cv_rmse.mean():.1f} Cr")

# ── Best Model: Gradient Boosting ───────────────────────────────────────────
best_model = models["Gradient Boosting"]
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\n--- Test Set Performance (Gradient Boosting) ---")
print(f"  R²   : {r2_score(y_test, y_pred):.4f}")
print(f"  RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} Cr")
print(f"  MAE  : {mean_absolute_error(y_test, y_pred):.2f} Cr")

# Save model
with open("outputs/models/gb_auction_predictor.pkl", "wb") as f:
    pickle.dump({"model": best_model, "features": FEATURES,
                 "le_role": le_role, "le_country": le_country}, f)
print("\n✅ Model saved: outputs/models/gb_auction_predictor.pkl")

# ── Plot 1: Model Comparison ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("ML Model Performance", fontsize=14, fontweight='bold')

res_df = pd.DataFrame(results)
colors_bar = ["#3498db","#27ae60","#e74c3c","#f39c12"]
bars = axes[0].bar(res_df["Model"], res_df["CV R²"], color=colors_bar,
                    edgecolor='white', linewidth=1.5)
axes[0].set_title("Cross-Validation R² Score")
axes[0].set_ylabel("R² Score"); axes[0].set_ylim(0, 1)
axes[0].set_xticklabels(res_df["Model"], rotation=15, ha='right')
for bar, val in zip(bars, res_df["CV R²"]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha='center', fontsize=9)

# Actual vs Predicted
axes[1].scatter(y_test, y_pred, color="#e74c3c", alpha=0.8, s=80, edgecolors='white')
lims = [min(y_test.min(), y_pred.min()) - 10, max(y_test.max(), y_pred.max()) + 10]
axes[1].plot(lims, lims, 'k--', linewidth=1.5, alpha=0.7, label="Perfect Prediction")
axes[1].set_title("Actual vs Predicted Auction Price")
axes[1].set_xlabel("Actual Price (Cr)"); axes[1].set_ylabel("Predicted Price (Cr)")
axes[1].legend()
for i, (actual, pred) in enumerate(zip(y_test, y_pred)):
    name = df.loc[X_test.index[i], "player_name"]
    if abs(actual - pred) > 60:
        axes[1].annotate(name, (actual, pred), fontsize=7, xytext=(3,3),
                         textcoords='offset points')

plt.tight_layout()
plt.savefig("outputs/plots/04_model_performance.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot saved: outputs/plots/04_model_performance.png")

# ── Plot 2: Feature Importance ───────────────────────────────────────────────
feat_imp = pd.DataFrame({
    "Feature"   : FEATURES,
    "Importance": best_model.feature_importances_
}).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors_feat = ["#e74c3c" if v > 0.08 else "#3498db" for v in feat_imp["Importance"]]
bars = ax.barh(feat_imp["Feature"], feat_imp["Importance"],
               color=colors_feat, edgecolor='white', linewidth=0.8)
ax.set_title("Feature Importance — Gradient Boosting Model",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Importance Score")
for bar, val in zip(bars, feat_imp["Importance"]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig("outputs/plots/05_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot saved: outputs/plots/05_feature_importance.png")

# ── Plot 3: Top Overvalued / Undervalued Players ─────────────────────────────
df_full = df.copy()
X_all = df_full[FEATURES].fillna(df_full[FEATURES].median())
df_full["predicted_price"] = best_model.predict(X_all)
df_full["value_gap"] = df_full["sold_price_cr"] - df_full["predicted_price"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("IPL Auction — Overvalued vs Undervalued Players", fontsize=13, fontweight='bold')

top_over = df_full.nlargest(8, "value_gap")[["player_name","sold_price_cr","predicted_price","value_gap"]]
top_under= df_full.nsmallest(8, "value_gap")[["player_name","sold_price_cr","predicted_price","value_gap"]]

axes[0].barh(top_over["player_name"], top_over["value_gap"], color="#e74c3c", edgecolor='white')
axes[0].set_title("Most Overvalued (Paid > Model Value)", fontweight='bold')
axes[0].set_xlabel("Overpriced by (Cr)"); axes[0].invert_yaxis()

axes[1].barh(top_under["player_name"], top_under["value_gap"].abs(), color="#27ae60", edgecolor='white')
axes[1].set_title("Most Undervalued (Paid < Model Value)", fontweight='bold')
axes[1].set_xlabel("Underpriced by (Cr)"); axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("outputs/plots/06_over_undervalued.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot saved: outputs/plots/06_over_undervalued.png")

print("\n✅ All ML steps complete!")
print("\n📌 Overvalued Players (top 5):")
print(top_over.head(5)[["player_name","sold_price_cr","predicted_price","value_gap"]].to_string(index=False))
print("\n📌 Undervalued Players (top 5):")
tmp = top_under.head(5)[["player_name","sold_price_cr","predicted_price","value_gap"]].copy()
tmp["value_gap"] = tmp["value_gap"].abs()
print(tmp.to_string(index=False))
