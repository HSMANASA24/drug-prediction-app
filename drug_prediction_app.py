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

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Drug Prescription Classifier", page_icon="💊", layout="centered")

# -------------------------
# CSS: Light + Dark themes (no f-strings)
# -------------------------
base_css = """
<style>
/* Background gradient for Light Mode */
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

# -------------------------
# Dark theme CSS (Modern Gray Mode)
# -------------------------

dark_css = """
<style>
/* Dark background */
body {
  background-color: #1c1c1c !important;
  color: #ffffff !important;
}

/* Sidebar remains light */
section[data-testid="stSidebar"] {
  background-color: rgba(255,255,255,0.97) !important;
  color: #111 !important;
}

/* Dark glass panels */
.glass-panel, .glass-panel-2 {
  backdrop-filter: blur(12px);
  background: rgba(40,40,40,0.55) !important;
  border-radius: 16px;
  padding: 20px;
  border: 1px solid rgba(255,255,255,0.08);
  color:#ffffff !important;
}

/* Normal text */
html, body, div, p, span, label {
  color:#ffffff !important;
  font-weight:400 !important;
}

/* Titles bold */
h1, h2, h3, h4 {
  font-weight:800 !important;
  color:#ffffff !important;
}

/* Inputs */
input, select, textarea {
  background: rgba(255,255,255,0.12) !important;
  border-radius: 10px !important;
  border: 1px solid rgba(255,255,255,0.25) !important;
  color:#ffffff !important;
}
input::placeholder { color: #ddd !important; }

/* Buttons */
.stButton>button {
  background: rgba(255,255,255,0.15) !important;
  color:#ffffff !important;
  border-radius: 10px !important;
  border: 1px solid rgba(255,255,255,0.35);
  font-weight:600 !important;
}
.stButton>button:hover { background: rgba(255,255,255,0.25) !important; transform: translateY(-2px); }

/* Tables readable */
table, th, td { color:#ffffff !important; }
</style>
"""

# -------------------------
# Sidebar (includes theme toggle + navigation)
# -------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "🌗 Theme Mode",
        ["Light Mode", "Dark Mode"],
        key="theme_switch"
    )

    page = st.radio(
        "📄 Navigate",
        ["Predictor", "Drug Information", "Bulk Prediction", "Monitoring", "About"],
        key="nav_select"
    )

    st.markdown("---")
    st.write("Use sidebar to navigate.")

# Apply the selected theme
st.markdown(base_css, unsafe_allow_html=True)
if mode == "Dark Mode":
    st.markdown(dark_css, unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.markdown("""
<div class="glass-panel" style="text-align:center;">
    <h1>💊 Drug Prescription Classifier</h1>
    <p style="color:#111; font-weight:600; margin-top:6px; font-size:18px;">
        Predict drugs, explain why, generate reports, and monitor usage.
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Drug info
# -------------------------
drug_images = {"drugA": "💊", "drugB": "🩺", "drugC": "⚗️", "drugX": "🧬", "drugY": "🩸"}

drug_details = {
    "drugA": {"name":"Drug A","use":"Used for normal BP and cholesterol.","mechanism":"Supports circulatory health.","side_effects":["Headache","Dry mouth"],"precautions":"Avoid alcohol.","dosage":"1 tablet daily."},
    "drugB": {"name":"Drug B","use":"For high blood pressure.","mechanism":"Relaxes blood vessels.","side_effects":["Low BP","Fatigue"],"precautions":"Not for pregnancy.","dosage":"1 tablet/day."},
    "drugC": {"name":"Drug C","use":"Balances electrolyte levels.","mechanism":"Balances Na/K.","side_effects":["Nausea"],"precautions":"Monitor levels.","dosage":"1–2 per day."},
    "drugX": {"name":"Drug X","use":"High cholesterol.","mechanism":"Reduces cholesterol.","side_effects":["Muscle pain"],"precautions":"Avoid high-fat foods.","dosage":"Evening dose."},
    "drugY": {"name":"Drug Y","use":"High BP + cholesterol.","mechanism":"Lowers BP & cholesterol.","side_effects":["Dizziness"],"precautions":"Regular BP checks.","dosage":"1 daily."}
}

# -------------------------
# Sample data + model training
# -------------------------
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
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, multi_class='multinomial'))])
    pipe.fit(X,y)
    return pipe

# -------------------------
# Session logs
# -------------------------
if 'logs' not in st.session_state:
    st.session_state['logs'] = []

def append_log(entry: dict):
    st.session_state['logs'].append(entry)

# -------------------------
# Explanation helper (simple rule-based)
# -------------------------
def explain_prediction(df_train, input_df, predicted):
    try:
        stats = df_train.groupby('Drug').agg({'Age':'mean','Na':'mean','K':'mean'})
        cat_modes = df_train.groupby('Drug').agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)[['Sex','BP','Cholesterol']]
    except Exception:
        return "No explanation available."

    reasons = []
    for col in ['Age','Na','K']:
        inp = float(input_df.iloc[0][col])
        mean_for_class = float(stats.loc[predicted][col])
        if inp >= mean_for_class:
            reasons.append(f"{col} ({inp}) is >= avg for {predicted} ({mean_for_class:.2f}) — supports prediction.")
        else:
            reasons.append(f"{col} ({inp}) is < avg for {predicted} ({mean_for_class:.2f}) — less supportive.")

    for col in ['Sex','BP','Cholesterol']:
        inp = input_df.iloc[0][col]
        mode_val = cat_modes.loc[predicted][col]
        if inp == mode_val:
            reasons.append(f"{col} = {inp} matches typical {predicted} patients.")
        else:
            reasons.append(f"{col} = {inp} differs from typical {predicted} patients (usually {mode_val}).")

    return "
".join(reasons)

# -------------------------
# PDF report generator
# -------------------------
def create_pdf_report(patient_info: dict, prediction: str, explanation: str, drug_info: dict):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin

    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y, "Drug Prescription Report")
    y -= 30

    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 25

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Patient Information:")
    y -= 18
    c.setFont("Helvetica", 11)
    for k, v in patient_info.items():
        c.drawString(margin+10, y, f"{k}: {v}")
        y -= 15

    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Prediction:")
    y -= 18
    c.setFont("Helvetica", 11)
    c.drawString(margin+10, y, f"Predicted Drug: {prediction}")
    y -= 18

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Explanation:")
    y -= 16
    c.setFont("Helvetica", 10)
    for line in explanation.split("
"):
        c.drawString(margin+10, y, line)
        y -= 14
        if y < 100:
            c.showPage()
            y = height - margin

    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Drug Details:")
    y -= 16
    c.setFont("Helvetica", 11)
    c.drawString(margin+10, y, f"Name: {drug_info.get('name','')}")
    y -= 14
    c.drawString(margin+10, y, f"Use: {drug_info.get('use','')}")
    y -= 14
    c.drawString(margin+10, y, f"Mechanism: {drug_info.get('mechanism','')}")
    y -= 14
    c.drawString(margin+10, y, "Side effects: " + ", ".join(drug_info.get('side_effects',[])))
    y -= 18
    c.drawString(margin+10, y, f"Precautions: {drug_info.get('precautions','')}")
    y -= 18
    c.save()
    buffer.seek(0)
    return buffer

# -------------------------
# PAGES
# -------------------------

# Predictor page
if page == "Predictor":
    st.markdown('<div class="glass-panel"><h2>Single Prediction</h2></div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-panel-2">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV (optional) for custom training", type=["csv"], key="predict_upload")
    st.markdown('</div>', unsafe_allow_html=True)

    df_train = pd.read_csv(uploaded) if uploaded else load_sample_df()
    model = train_model(df_train)

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("Enter Patient Details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", ["F", "M"])
        bp = st.selectbox("Blood Pressure", ["LOW", "NORMAL", "HIGH"]) 
    with col2:
        cholesterol = st.selectbox("Cholesterol", ["HIGH", "NORMAL"])
        na = st.number_input("Sodium (Na)", format="%.4f", value=0.70)
        k = st.number_input("Potassium (K)", format="%.4f", value=0.05)

    if st.button("🔍 Predict Drug Type"):
        input_df = pd.DataFrame([[age, sex, bp, cholesterol, na, k]], columns=['Age','Sex','BP','Cholesterol','Na','K'])
        pred = model.predict(input_df)[0]

        explanation = explain_prediction(df_train, input_df, pred)
        st.success(f"💊 Predicted Drug: **{pred}**")
        st.markdown("**Why this prediction?**")
        st.write(explanation)

        append_log({
            "timestamp": datetime.datetime.now().isoformat(),
            "Age": age, "Sex": sex, "BP": bp, "Cholesterol": cholesterol, "Na": na, "K": k,
            "prediction": pred
        })

        if st.button("📄 Download PDF Report"):
            patient_info = {"Age": age, "Sex": sex, "BP": bp, "Cholesterol": cholesterol, "Na": na, "K": k}
            pdf_buf = create_pdf_report(patient_info, pred, explanation, drug_details.get(pred, {}))
            st.download_button("📥 Download PDF", data=pdf_buf, file_name=f"report_{pred}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
    st.markdown('</div>', unsafe_allow_html=True)

# Drug Information page
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

# Bulk Prediction page
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
            model = train_model(load_sample_df())
            df_bulk["Predicted_Drug"] = model.predict(df_bulk[required])
            st.success("Bulk prediction completed!")

            for _, r in df_bulk.iterrows():
                append_log({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "Age": r['Age'], "Sex": r['Sex'], "BP": r['BP'], "Cholesterol": r['Cholesterol'],
                    "Na": r['Na'], "K": r['K'], "prediction": r['Predicted_Drug']
                })

            st.dataframe(df_bulk)
            st.download_button("📥 Download Results CSV", df_bulk.to_csv(index=False).encode("utf-8"), file_name="bulk_predictions.csv", mime="text/csv")
    else:
        st.info("Upload a CSV file to begin.")

# Monitoring page
if page == "Monitoring":
    st.markdown('<div class="glass-panel"><h2>Monitoring</h2></div>', unsafe_allow_html=True)
    logs = st.session_state.get('logs', [])
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
        st.download_button("📥 Download Logs", df_logs.to_csv(index=False).encode("utf-8"), "prediction_logs.csv", "text/csv")

# About page
if page == "About":
    st.markdown('<div class="glass-panel"><h2>About</h2></div>', unsafe_allow_html=True)
    st.write("""
    ### Drug Prescription Classifier
    This application predicts drug types from patient data using machine learning.

    **Features:**
    - Single prediction with explanation
    - Bulk CSV predictions
    - Usage monitoring
    - PDF medical reports
    - Glass UI theme with Light/Dark mode
    """)

st.caption("Built with ❤️ — Drug Predictor App")
