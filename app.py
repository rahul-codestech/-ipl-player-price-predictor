import streamlit as st
import pickle
import numpy as np
import os

# ── Load Model & Encoders ───────────────────────────────
base = os.path.dirname(__file__)

with open(os.path.join(base, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(base, 'encoders.pkl'), 'rb') as f:
    enc = pickle.load(f)

le_role    = enc['le_role']
le_country = enc['le_country']

# ── Page Config ─────────────────────────────────────────
st.set_page_config(page_title="IPL Price Predictor", page_icon="🏏", layout="centered")

st.title("🏏 IPL Player Price Predictor")
st.markdown("Fill in the player details to predict **auction price in Crores**")
st.divider()

# ── Input Form ──────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    age             = st.slider("Age", 18, 40, 25)
    role            = st.selectbox("Role", le_role.classes_)
    country         = st.selectbox("Country", le_country.classes_)
    base_price      = st.number_input("Base Price (Lakhs)", 20, 200, 50)
    is_star         = st.selectbox("Star Player?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    seasons_played  = st.slider("Seasons Played", 1, 15, 3)

with col2:
    total_runs      = st.number_input("Total Runs", 0, 10000, 500)
    avg_bat_avg     = st.number_input("Batting Average", 0.0, 100.0, 25.0)
    avg_sr          = st.number_input("Strike Rate", 0.0, 250.0, 120.0)
    total_50s       = st.number_input("Total 50s", 0, 100, 5)
    total_100s      = st.number_input("Total 100s", 0, 50, 1)
    total_wkts      = st.number_input("Total Wickets", 0, 500, 20)
    avg_eco         = st.number_input("Economy Rate", 0.0, 20.0, 8.0)
    form_runs       = st.number_input("Form Runs (Last Season)", 0, 1000, 200)
    form_wickets    = st.number_input("Form Wickets (Last Season)", 0, 50, 5)
    impact_score    = st.number_input("Impact Score (0-1)", 0.0, 1.0, 0.5)

st.divider()

# ── Predict ─────────────────────────────────────────────
if st.button("🚀 Predict Price", use_container_width=True):
    role_enc    = le_role.transform([role])[0]
    country_enc = le_country.transform([country])[0]

    features = np.array([[age, role_enc, country_enc, base_price, is_star,
                          total_runs, avg_sr, avg_bat_avg, total_50s, total_100s,
                          total_wkts, avg_eco, form_runs, form_wickets,
                          impact_score, seasons_played]])

    price = model.predict(features)[0]

    st.success(f"💰 Predicted Auction Price: **₹ {price:.2f} Crore**")

    if price > 10:
        st.balloons()
        st.markdown("🌟 **Star Player Territory!**")
    elif price > 5:
        st.markdown("💪 **Solid Pick!**")
    else:
        st.markdown("🎯 **Budget Friendly Player**")
        