# 🏏 IPL Player Auction Price Predictor

An end-to-end Machine Learning web app that predicts IPL player auction prices based on performance stats.

## 🌐 Live Demo
👉 [Try the App on Hugging Face](https://huggingface.co/spaces/rahul2719/ipl-player-price-predictor)

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| R² Score | 0.88 |
| RMSE | 39.41 Cr |
| Cross-Validation R² | 0.91 ± 0.08 |

## 🔧 Tech Stack
- **Language:** Python
- **ML Model:** Gradient Boosting Regressor
- **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **Web App:** Streamlit
- **Deployment:** Hugging Face Spaces

## ⚙️ Features
- Real-time player price prediction
- Interactive UI with sliders & dropdowns
- Supports all player roles (Batsman, Bowler, All-Rounder, WK-Batsman)
- Predicts based on 16 features including batting avg, strike rate, wickets, form

## 🚀 Run Locally
```bash
git clone https://github.com/rahul-codestech/-ipl-player-price-predictor.git
cd ipl-player-price-predictor
pip install -r requirements.txt
streamlit run app.py
```

## 👤 Author
**Rahul** | [GitHub](https://github.com/rahul-codestech) | [Hugging Face](https://huggingface.co/rahul2719)
