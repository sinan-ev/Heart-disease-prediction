# ❤️ HeartGuard AI

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> **A production-ready, interactive heart disease risk assessment web app powered by ensemble machine learning models.**

---

## 📖 Description

**HeartGuard AI** is a Streamlit web application that predicts the likelihood of heart disease based on 13 clinical indicators from the UCI Heart Disease Dataset. It provides real-time risk scoring with an interactive gauge chart, color-coded risk factor summaries, and prediction history tracking — all wrapped in a visually polished dark-themed UI.

---

## ✨ Features

- 🎨 **Dark-themed UI** with a custom red-gradient header banner and card-style containers
- 🌳🚀⚡ **3 ML Models** — Random Forest, Gradient Boosting, and XGBoost — selectable at runtime
- 📊 **Interactive Plotly Gauge Chart** showing risk score 0–100 with green / yellow / red zones
- 🔎 **Risk Factor Summary Table** with color-coded indicators (normal vs. risky)
- 📈 **Dataset Statistics Tab** with descriptive stats, age distribution histogram, and target bar chart
- 🕑 **Prediction History** stored in session state and shown in the sidebar (last 5)
- 📋 **Medical Disclaimer**, **How It Works** explanation, and **Dataset Stats** in bottom tabs
- ✅ **Graceful error handling** — try/except around all model loading and prediction code
- 🚀 **Deployment-ready** with Procfile, setup.sh, and GitHub Actions CI workflow

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| ML Models | scikit-learn, XGBoost |
| Data Processing | NumPy, Pandas, Joblib |
| Visualisation | Plotly |
| Dataset | UCI Heart Disease (Cleveland) |
| Language | Python 3.11 |
| Deployment | Heroku / Railway / Streamlit Cloud |

---

## 🚀 Setup & Running

### 1. Clone the repository
```bash
git clone https://github.com/your-username/heartguard-ai.git
cd heartguard-ai
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 🧠 Model Information

| Model | Type | Key Strength |
|-------|------|-------------|
| Random Forest 🌳 | Bagging ensemble | Robust, low variance, feature importance |
| Gradient Boosting 🚀 | Sequential boosting | High accuracy, handles complex patterns |
| XGBoost ⚡ | Optimised gradient boosting | Fast, regularised, state-of-the-art |

All three models were trained on the UCI Cleveland Heart Disease Dataset and evaluated with cross-validation. Pre-trained `.pkl` files are bundled with the repository.

---

## 📂 Dataset

- **Source:** [UCI Machine Learning Repository — Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- **Records:** 303 patient records (Cleveland database)
- **Features:** 13 clinical attributes
- **Target:** Binary — 0 (No Disease) / 1 (Disease)

### Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| age | Numeric | Age in years |
| sex | Binary | 1 = Male, 0 = Female |
| cp | Categorical | Chest pain type (0–3) |
| trestbps | Numeric | Resting blood pressure (mmHg) |
| chol | Numeric | Serum cholesterol (mg/dL) |
| fbs | Binary | Fasting blood sugar > 120 mg/dL |
| restecg | Categorical | Resting ECG results (0–2) |
| thalach | Numeric | Maximum heart rate achieved |
| exang | Binary | Exercise-induced angina |
| oldpeak | Numeric | ST depression induced by exercise |
| slope | Categorical | Slope of peak exercise ST segment |
| ca | Numeric | Number of major vessels (0–4) |
| thal | Categorical | Thalassemia type (0–3) |

---

## 📸 Screenshots

> *Screenshots will be added after deployment.*

| Main View | Results Panel | Dataset Stats |
|-----------|--------------|---------------|
| *(coming soon)* | *(coming soon)* | *(coming soon)* |

---

## 🌐 Deployment

### Streamlit Community Cloud
1. Push the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set **main file** to `app.py` and deploy.

### Heroku
```bash
heroku create heartguard-ai
git push heroku main
```

### Railway
Connect your GitHub repository in the [Railway dashboard](https://railway.app) — Railway auto-detects the Procfile.

---

## ⚠️ Medical Disclaimer

This application is for **educational and research purposes only**. It is **not a medical device** and should not be used as a substitute for professional medical advice. Always consult a qualified healthcare professional for medical decisions.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

Built with ❤️ as part of a machine learning portfolio project.

- **Year:** 2026
- **Stack:** Python · Streamlit · scikit-learn · XGBoost · Plotly
