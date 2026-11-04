import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HeartGuard AI",
    page_icon="heart",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global font & background ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Header banner ── */
.header-banner {
    background: linear-gradient(135deg, #1a3a5c 0%, #1e5f8e 60%, #2980b9 100%);
    padding: 2rem 2.5rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 24px rgba(41, 128, 185, 0.25);
}
.header-banner h1 {
    color: #ffffff;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: 0.5px;
}
.header-banner p {
    color: rgba(255, 255, 255, 0.82);
    font-size: 1.05rem;
    margin: 0.4rem 0 0 0;
}

/* ── Section headings ── */
.section-header {
    font-size: 1rem;
    font-weight: 700;
    color: #2980b9;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid #dde3ea;
    padding-bottom: 0.3rem;
}

/* ── Result boxes ── */
.result-safe {
    background: #eaf7ee;
    border: 2px solid #27ae60;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    color: #1a7a40;
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 1rem;
    box-shadow: 0 3px 14px rgba(39, 174, 96, 0.15);
}
.result-risk {
    background: #fef0ee;
    border: 2px solid #e74c3c;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    color: #c0392b;
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 1rem;
    box-shadow: 0 3px 14px rgba(231, 76, 60, 0.15);
}

/* ── Badges ── */
.badge-safe {
    display: inline-block;
    background: #d5f5e3;
    color: #1a7a40;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.82rem;
    font-weight: 600;
}
.badge-risk {
    display: inline-block;
    background: #fce8e6;
    color: #c0392b;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.82rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "Random Forest":       os.path.join(BASE_DIR, "random_forest_model.pkl"),
    "Gradient Boosting":   os.path.join(BASE_DIR, "gradient_boosting_model.pkl"),
    "XGBoost":             os.path.join(BASE_DIR, "xgboost_model.pkl"),
}
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATA_PATH   = os.path.join(BASE_DIR, "heart_disease_uci.csv")

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []

# ── Helper: load model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

@st.cache_resource
def load_scaler(path: str):
    return joblib.load(path)

@st.cache_data
def load_data(path: str):
    return pd.read_csv(path)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")
    st.markdown("---")

    model_choice = st.selectbox(
        "Select Prediction Model",
        list(MODEL_PATHS.keys()),
        index=0,
    )

    st.markdown("---")

    try:
        model  = load_model(MODEL_PATHS[model_choice])
        scaler = load_scaler(SCALER_PATH)
        st.success("Model loaded successfully")
    except Exception as exc:
        st.error(f"Model load error: {exc}")
        model  = None
        scaler = None

    st.markdown("---")

    with st.expander("About"):
        st.markdown("""
**HeartGuard AI** uses machine learning to assess heart disease risk based on clinical indicators.

Three ensemble models are available:
- **Random Forest** — bagging of decision trees
- **Gradient Boosting** — sequential boosting
- **XGBoost** — optimized gradient boosting

*This tool is for educational purposes only.*
        """)

    st.markdown("---")
    st.markdown("### Prediction History")
    if st.session_state["history"]:
        hist_df = pd.DataFrame(st.session_state["history"][-5:])
        st.dataframe(hist_df[["model", "result", "risk_pct"]].rename(
            columns={"model": "Model", "result": "Result", "risk_pct": "Risk %"}
        ), use_container_width=True)
    else:
        st.caption("No predictions yet.")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <h1>HeartGuard AI</h1>
  <p>Clinical Heart Disease Risk Assessment powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
col_form, col_results = st.columns([1.6, 1], gap="large")

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — Input Form
# ═══════════════════════════════════════════════════════════════════════════════
with col_form:
    st.markdown("### Patient Information")

    with st.form("input_form"):

        # ── Demographics ───────────────────────────────────────────────────────
        with st.expander("Patient Demographics", expanded=True):
            age = st.slider("Age", min_value=1, max_value=100, value=45,
                            help="Patient age in years")
            sex_label = st.selectbox("Sex", ["Male", "Female"],
                                     help="Biological sex of the patient")
            sex = 1 if sex_label == "Male" else 0

        # ── Cardiac Indicators ─────────────────────────────────────────────────
        with st.expander("Cardiac Indicators", expanded=True):
            cp = st.selectbox(
                "Chest Pain Type",
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "0 – Typical Angina",
                    1: "1 – Atypical Angina",
                    2: "2 – Non-anginal Pain",
                    3: "3 – Asymptomatic",
                }[x],
            )
            trestbps = st.slider("Resting Blood Pressure (mmHg)",
                                 min_value=80, max_value=200, value=120)
            chol = st.slider("Cholesterol (mg/dL)",
                             min_value=100, max_value=600, value=200)
            fbs = st.checkbox("Fasting Blood Sugar > 120 mg/dL")
            restecg = st.selectbox(
                "Resting ECG Results",
                options=[0, 1, 2],
                format_func=lambda x: {
                    0: "0 – Normal",
                    1: "1 – ST-T Wave Abnormality",
                    2: "2 – Left Ventricular Hypertrophy",
                }[x],
            )

        # ── Exercise Data ──────────────────────────────────────────────────────
        with st.expander("Exercise Data", expanded=True):
            thalach = st.slider("Max Heart Rate Achieved",
                                min_value=60, max_value=250, value=150)
            exang = st.checkbox("Exercise Induced Angina")
            oldpeak = st.slider("ST Depression (Oldpeak)",
                                min_value=0.0, max_value=6.2,
                                step=0.1, value=1.0)
            slope = st.selectbox(
                "Slope of Peak Exercise ST Segment",
                options=[0, 1, 2],
                format_func=lambda x: {
                    0: "0 – Upsloping",
                    1: "1 – Flat",
                    2: "2 – Downsloping",
                }[x],
            )
            ca = st.selectbox("Number of Major Vessels (0–4)",
                              options=[0, 1, 2, 3, 4])
            thal = st.selectbox(
                "Thalassemia",
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "0 – Normal",
                    1: "1 – Fixed Defect",
                    2: "2 – Reversible Defect",
                    3: "3 – Unknown",
                }[x],
            )

        submitted = st.form_submit_button(
            "Analyze Heart Health",
            use_container_width=True,
            type="primary",
        )

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Results
# ═══════════════════════════════════════════════════════════════════════════════
with col_results:
    st.markdown("### Risk Assessment")

    if not submitted:
        st.info("Fill in the patient details on the left and click **Analyze Heart Health** to see results.")
    else:
        if model is None or scaler is None:
            st.error("Model or scaler not loaded. Check the sidebar for errors.")
        else:
            # Build feature array
            features = np.array([[
                age, sex, cp, trestbps, chol,
                int(fbs), restecg, thalach, int(exang),
                oldpeak, slope, ca, thal,
            ]], dtype=float)

            try:
                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)[0]

                # Probability
                if hasattr(model, "predict_proba"):
                    proba    = model.predict_proba(features_scaled)[0]
                    risk_pct = float(proba[1]) * 100
                else:
                    risk_pct = 100.0 if int(prediction) == 1 else 0.0

                # ── Result banner ──────────────────────────────────────────
                if int(prediction) == 0:
                    st.markdown(
                        '<div class="result-safe">LOW RISK<br>'
                        '<small>No significant heart disease indicators detected</small></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="result-risk">HIGH RISK<br>'
                        '<small>Heart disease indicators detected — consult a physician</small></div>',
                        unsafe_allow_html=True,
                    )

                # ── Probability bar ────────────────────────────────────────
                st.markdown(f"**Risk Probability: {risk_pct:.1f}%**")
                st.progress(min(risk_pct / 100.0, 1.0))

                # ── Gauge chart ────────────────────────────────────────────
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_pct,
                    delta={"reference": 50, "valueformat": ".1f"},
                    number={"suffix": "%", "valueformat": ".1f"},
                    title={"text": "Heart Disease Risk Score", "font": {"size": 16, "color": "#2c3e50"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1,
                                 "tickcolor": "#7f8c8d"},
                        "bar": {"color": "#2980b9", "thickness": 0.25},
                        "bgcolor": "#f9fbfd",
                        "borderwidth": 1,
                        "bordercolor": "#dde3ea",
                        "steps": [
                            {"range": [0,  40], "color": "#d5f5e3"},
                            {"range": [40, 70], "color": "#fef9e7"},
                            {"range": [70, 100], "color": "#fce8e6"},
                        ],
                        "threshold": {
                            "line": {"color": "#2c3e50", "width": 3},
                            "thickness": 0.75,
                            "value": risk_pct,
                        },
                    },
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="#ffffff",
                    font_color="#2c3e50",
                    height=300,
                    margin=dict(l=20, r=20, t=60, b=20),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # ── Risk factor summary ────────────────────────────────────
                st.markdown("#### Risk Factor Summary")

                risk_factors = {
                    "Age":             (age,         age > 60,       f"{age} yrs",       "> 60 yrs"),
                    "Cholesterol":     (chol,        chol > 240,     f"{chol} mg/dL",    "> 240 mg/dL"),
                    "Resting BP":      (trestbps,    trestbps > 140, f"{trestbps} mmHg", "> 140 mmHg"),
                    "Max Heart Rate":  (thalach,     thalach < 100,  f"{thalach} bpm",   "< 100 bpm"),
                    "ST Depression":   (oldpeak,     oldpeak > 2.0,  f"{oldpeak}",       "> 2.0"),
                    "Exercise Angina": (int(exang),  exang,          "Yes" if exang else "No", "Yes"),
                }

                rows = []
                for factor, (val, is_risky, display, threshold) in risk_factors.items():
                    status = "Risky" if is_risky else "Normal"
                    rows.append({
                        "Factor":    factor,
                        "Value":     display,
                        "Threshold": threshold,
                        "Status":    status,
                    })

                risk_df = pd.DataFrame(rows)

                def style_status(val):
                    if val == "Risky":
                        return "background-color: #fce8e6; color: #c0392b; font-weight: bold;"
                    return "background-color: #d5f5e3; color: #1a7a40; font-weight: bold;"

                st.dataframe(
                    risk_df.style.map(style_status, subset=["Status"]),
                    use_container_width=True,
                    hide_index=True,
                )

                # ── Save to history ────────────────────────────────────────
                st.session_state["history"].append({
                    "model":    model_choice,
                    "result":   "High Risk" if int(prediction) == 1 else "Low Risk",
                    "risk_pct": f"{risk_pct:.1f}%",
                    "age":      age,
                    "chol":     chol,
                })

            except Exception as exc:
                st.error(f"Prediction error: {exc}")

# ═══════════════════════════════════════════════════════════════════════════════
# BOTTOM — Info Tabs
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
tab_data, tab_how, tab_disclaimer = st.tabs(
    ["Dataset Stats", "How It Works", "Medical Disclaimer"]
)

with tab_data:
    try:
        df = load_data(DATA_PATH)
        st.markdown(f"**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown("**Descriptive Statistics**")
            st.dataframe(df.describe().round(2), use_container_width=True)

        with col_d2:
            age_col = None
            for c in ["age", "Age"]:
                if c in df.columns:
                    age_col = c
                    break

            if age_col:
                fig_age = px.histogram(
                    df, x=age_col, nbins=20,
                    title="Age Distribution",
                    color_discrete_sequence=["#2980b9"],
                    template="simple_white",
                )
                fig_age.update_layout(
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=320,
                )
                st.plotly_chart(fig_age, use_container_width=True)

        target_col = None
        for c in ["target", "num", "condition"]:
            if c in df.columns:
                target_col = c
                break

        if target_col:
            target_counts = df[target_col].value_counts().reset_index()
            target_counts.columns = ["Target", "Count"]
            target_counts["Label"] = target_counts["Target"].apply(
                lambda x: "Heart Disease" if x >= 1 else "No Disease"
            )
            fig_target = px.bar(
                target_counts, x="Label", y="Count",
                title="Target Variable Distribution",
                color="Label",
                color_discrete_map={"Heart Disease": "#e74c3c", "No Disease": "#27ae60"},
                template="simple_white",
            )
            fig_target.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10),
                height=320,
            )
            st.plotly_chart(fig_target, use_container_width=True)

    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")

with tab_how:
    st.markdown("""
### How HeartGuard AI Works

HeartGuard AI uses three powerful ensemble machine-learning models trained on the **UCI Heart Disease Dataset** (Cleveland), which contains 303 patient records with 13 clinical features.

---

#### Random Forest
A **bagging** ensemble of hundreds of decision trees. Each tree is trained on a random bootstrap sample of the data with a random subset of features. Predictions are made by majority vote. Random Forest is robust to overfitting and provides stable probability estimates.

---

#### Gradient Boosting
A **sequential boosting** method that builds trees one at a time, where each new tree corrects the errors of all previous trees. It minimises a differentiable loss function using gradient descent in function space, making it highly accurate on structured/tabular data.

---

#### XGBoost (Extreme Gradient Boosting)
An **optimised implementation** of gradient boosting with regularisation (L1 & L2), parallel tree construction, and hardware optimisations. XGBoost consistently achieves state-of-the-art results on tabular datasets and is the method of choice in many Kaggle competitions.

---

#### Pipeline
1. Raw input features are collected from the form.
2. Features are **standardised** using a pre-fitted `StandardScaler` (zero mean, unit variance).
3. The scaled features are passed to the selected model.
4. The model returns a **binary prediction** (0 = no disease, 1 = disease) and a **probability score**.
5. The probability score drives the gauge chart and risk assessment.

---

#### Features Used
| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | 1 = Male, 0 = Female |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mmHg) |
| chol | Serum cholesterol (mg/dL) |
| fbs | Fasting blood sugar > 120 mg/dL |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels (0–4) |
| thal | Thalassemia type |
    """)

with tab_disclaimer:
    st.markdown("""
### Medical Disclaimer

> **IMPORTANT: This application is for educational and research purposes ONLY.**

---

**HeartGuard AI is NOT a medical device and should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.**

#### Key Points:

- **Not a Diagnostic Tool** — This application uses machine learning models trained on a limited dataset and **cannot replace** clinical diagnosis by a qualified healthcare professional.

- **Dataset Limitations** — The models were trained on the UCI Cleveland Heart Disease Dataset (303 records). Performance may vary significantly on real-world populations and may not generalise to all demographic groups.

- **No Medical Validation** — These models have not undergone clinical trials or regulatory approval (FDA, CE, etc.) required for medical devices.

- **Consult a Doctor** — If you are concerned about heart disease or any other medical condition, **please consult a licensed physician** or cardiologist immediately.

- **Data Privacy** — Do not enter real patient data into this application without ensuring appropriate consent and data governance compliance.

- **Liability** — The developers of this application assume no liability for decisions made based on its output.

---

*By using this application, you acknowledge that you have read and understood this disclaimer.*

**Emergency?** If you are experiencing symptoms of a heart attack (chest pain, shortness of breath, arm/jaw pain), call emergency services immediately: **911 (US) | 999 (UK) | 112 (EU)**
    """)
