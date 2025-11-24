# app.py
import streamlit as st
import pandas as pd
import io
import datetime

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# --------------------------------
# CLEAN CSS (Glassmorphism + titles bold only)
# --------------------------------
base_css = """
<style>
/* Background gradient */
body {
    background: linear-gradient(135deg, rgba(31,58,147,1), rgba(110,43,168,1), rgba(245,69,145,1));
    background-attachment: fixed;
}

/* Sidebar stays white */
section[data-testid="stSidebar"] {
    background-color: rgba(255,255,255,0.97) !important;
    color:#111 !important;
}

/* Glass Panels */
.glass-panel, .glass-panel-2 {
    backdrop-filter: blur(10px);
    background: rgba(255,255,255,0.65);
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 20px;
    border: 1px solid rgba(0,0,0,0.12);
    color:#111 !important;
}

/* NORMAL TEXT */
html, body, div, p, span, label, input, textarea, select {
    color:#111 !important;
    font-weight:400 !important;
}

/* ONLY TITLES BOLD */
h1, h2, h3, h4, h5 {
    color:#111 !important;
    font-weight:800 !important;
}

/* Inputs */
input, textarea, select {
    background: rgba(255,255,255,0.9) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(0,0,0,0.25) !important;
}

/* Buttons */
.stButton>button {
    background: rgba(255,255,255,0.9) !important;
    color:#111 !important;
    border-radius: 10px !important;
    border: 1px solid rgba(0,0,0,0.25);
    font-weight:600 !important;
}

.stButton>button:hover {
    background: rgba(240,240,240,1) !important;
    transform: translateY(-2px);
}

/* Tables */
table, th, td {
    color:#111 !important;
    font-weight:400 !important;
}
</style>
"""

# Apply CSS
st.markdown(base_css, unsafe_allow_html=True)


# -------------------------
# SINGLE SIDEBAR (NO THEME MODE + NO PREDICTOR)
# -------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    page = st.radio(
        "📄 Navigate",
        ["Drug Information", "Bulk Prediction", "Monitoring", "About"],
        key="nav_select"
    )

    st.markdown("---")
    st.write("Use sidebar to navigate.")


# -------------------------
# Header
# -------------------------
st.markdown("""
<div class="glass-panel" style="text-align:center;">
    <h1>💊 Drug Prescription Classifier</h1>
    <p style="color:#111; font-weight:600; margin-top:6px; font-size:18px;">
        Predict drugs, analyze data, generate reports, and monitor usage.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------
# DRUG DETAILS
# --------------------------------
drug_images = {"drugA": "💊","drugB": "🩺","drugC": "⚗️","drugX": "🧬","drugY": "🩸"}

drug_details = {
    "drugA": {"name":"Drug A","use":"Used for normal BP and cholesterol.","mechanism":"Supports circulatory health.","side_effects":["Headache","Dry mouth"],"precautions":"Avoid alcohol.","dosage":"1 tablet daily."},
    "drugB": {"name":"Drug B","use":"For high blood pressure.","mechanism":"Relaxes blood vessels.","side_effects":["Low BP","Fatigue"],"precautions":"Not for pregnancy.","dosage":"1 tablet/day."},
    "drugC": {"name":"Drug C","use":"Balances electrolyte levels.","mechanism":"Balances Na/K.","side_effects":["Nausea"],"precautions":"Monitor levels.","dosage":"1–2 per day."},
    "drugX": {"name":"Drug X","use":"High cholesterol.","mechanism":"Reduces cholesterol.","side_effects":["Muscle pain"],"precautions":"Avoid high-fat foods.","dosage":"Evening dose."},
    "drugY": {"name":"Drug Y","use":"High BP + cholesterol.","mechanism":"Lowers BP & cholesterol.","side_effects":["Dizziness"],"precautions":"Regular BP checks.","dosage":"1 daily."}
}


# --------------------------------
# MODEL TRAINING FUNCTION
# --------------------------------
def load_sample_df():
    return pd.DataFrame([
        [23,'F','HIGH','HIGH',0.792535,0.031258,'drugY'],
        [47,'M','LOW','HIGH',0.739309,0.056468,'drugC'],
        [47,'M','LOW','HIGH',0.697269,0.068944,'drugC'],
        [28,'F','NORMAL','HIGH',0.563682,0.072289,'drugX'],
        [61,'F','LOW','HIGH',0.559294,0.030998,'drugY'],
        [45,'M','NORMAL','NORMAL',0.7,0.05,'drugA']
    ], columns=['Age','Sex','BP','Cholesterol','Na','K','Drug'])


@st.cache_resource
def train_model(df):
    X = df[['Age','Sex','BP','Cholesterol','Na','K']]
    y = df['Drug']

    numeric = ['Age','Na','K']
    categorical = ['Sex','BP','Cholesterol']

    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
    ])

    model = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, multi_class='multinomial'))
    ])

    model.fit(X,y)
    return model



# SESSION LOGS
if "logs" not in st.session_state:
    st.session_state["logs"] = []

def append_log(entry: dict):
    st.session_state["logs"].append(entry)



# --------------------------------
# PAGE 1 — DRUG INFORMATION
# --------------------------------
if page == "Drug Information":
    st.markdown('<div class="glass-panel"><h2>Drug Information</h2></div>', unsafe_allow_html=True)

    choice = st.selectbox("Select Drug", list(drug_details.keys()))
    info = drug_details[choice]
    icon = drug_images[choice]

    st.markdown(f"""
    <div class="glass-panel">
        <h2>{icon} {info['name']}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Use")
    st.write(info["use"])

    st.markdown("### Mechanism")
    st.write(info["mechanism"])

    st.markdown("### Side Effects")
    for s in info["side_effects"]:
        st.write("- " + s)

    st.markdown("### Precautions")
    st.write(info["precautions"])

    st.markdown("### Dosage")
    st.write(info["dosage"])



# --------------------------------
# PAGE 2 — BULK PREDICTION
# --------------------------------
if page == "Bulk Prediction":

    st.markdown('<div class="glass-panel"><h2>Bulk Prediction</h2></div>', unsafe_allow_html=True)

    csv_file = st.file_uploader("Upload CSV for bulk prediction (Age,Sex,BP,Cholesterol,Na,K)", type=["csv"])

    if csv_file:
        df_bulk = pd.read_csv(csv_file)
        st.dataframe(df_bulk.head())

        required = ['Age','Sex','BP','Cholesterol','Na','K']
        missing = [c for c in required if c not in df_bulk.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            model = train_model(load_sample_df())  # using sample training data
            df_bulk["Predicted_Drug"] = model.predict(df_bulk[required])
            st.success("Bulk prediction completed!")

            # Save logs
            for _, r in df_bulk.iterrows():
                append_log({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "Age": r['Age'],
                    "Sex": r['Sex'],
                    "BP": r['BP'],
                    "Cholesterol": r['Cholesterol'],
                    "Na": r['Na'],
                    "K": r['K'],
                    "prediction": r["Predicted_Drug"]
                })

            st.dataframe(df_bulk)

            st.download_button(
                "📥 Download Results CSV",
                df_bulk.to_csv(index=False).encode("utf-8"),
                file_name="bulk_predictions.csv",
                mime="text/csv"
            )

    else:
        st.info("Upload a CSV file to begin.")



# --------------------------------
# PAGE 3 — MONITORING
# --------------------------------
if page == "Monitoring":
    st.markdown('<div class="glass-panel"><h2>Monitoring</h2></div>', unsafe_allow_html=True)

    logs = st.session_state["logs"]

    if not logs:
        st.info("No predictions yet this session.")
    else:
        df_logs = pd.DataFrame(logs)

        st.write("### Prediction Counts")
        st.bar_chart(df_logs["prediction"].value_counts())

        st.write("### Timeline")
        times = pd.to_datetime(df_logs["timestamp"]).dt.floor("min").value_counts().sort_index()
        st.line_chart(times)

        st.write("### Logs Table")
        st.dataframe(df_logs)

        st.download_button(
            "📥 Download Logs",
            df_logs.to_csv(index=False).encode("utf-8"),
            "prediction_logs.csv",
            "text/csv"
        )



# --------------------------------
# PAGE 4 — ABOUT
# --------------------------------
if page == "About":
    st.markdown('<div class="glass-panel"><h2>About</h2></div>', unsafe_allow_html=True)

    st.write("""
    ### Drug Prescription Classifier  
    This application predicts drug types from patient data using machine learning.

    **Features:**
    - Drug information  
    - Bulk CSV predictions  
    - Usage monitoring  
    - Glass UI theme  
    """)

st.caption("Built with ❤️ — Drug Predictor App")
