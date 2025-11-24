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

# ----------------------
# Page config & sidebar
# ----------------------
st.set_page_config(page_title="Drug Prescription Classifier", page_icon="💊", layout="centered")
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("🌗 Theme Mode", ["Light Mode", "Dark Mode"], key="theme_switch")
    st.markdown("---")
    page = st.radio("📄 Navigate", ["Predictor", "Drug Information", "Bulk Prediction", "Monitoring", "About"], key="nav_page")
    st.markdown("---")
    st.write("Use sidebar to switch pages or theme.")

# ----------------------
# Minimal CSS (glass + readability)
# ----------------------
base_css = """
<style>

/* Background gradient */
body {
  background: linear-gradient(135deg, rgba(31,58,147,1) 0%, rgba(110,43,168,1) 40%, rgba(245,69,145,1) 100%);
  background-attachment: fixed;
}

/* Sidebar (light) */
section[data-testid="stSidebar"] {
  background-color: rgba(255,255,255,0.97) !important;
  color: #111 !important;
}

/* Glass Panels */
.glass-panel, .glass-panel-2 {
  backdrop-filter: blur(10px) saturate(160%);
  background: rgba(255,255,255,0.65);
  border-radius: 18px;
  padding: 18px;
  margin-bottom: 20px;
  border: 1px solid rgba(0,0,0,0.12);
  color: #111 !important;   /* Dark text */
}

/* TEXT — make everything dark & bold */
html, body, div, p, span, label, input, select, textarea {
  color: #111 !important;
  font-weight: 700 !important;
}

/* Headings */
h1, h2, h3, h4, h5 {
  color: #111 !important;
  font-weight: 800 !important;
}

/* Input fields */
input, textarea, select {
  background: rgba(255,255,255,0.9) !important;
  color: #111 !important;
  border: 1px solid rgba(0,0,0,0.25) !important;
  border-radius: 10px !important;
}

/* File uploader */
div[data-testid="stFileUploader"] {
  background: rgba(255,255,255,0.85) !important;
  border-radius: 10px !important;
  border: 1px solid rgba(0,0,0,0.25) !important;
}

/* Buttons */
.stButton>button {
  background: rgba(255,255,255,0.9) !important;
  color: #111 !important;
  font-weight: 700 !important;
  border-radius: 10px;
  border: 1px solid rgba(0,0,0,0.25);
}
.stButton>button:hover {
  background: rgba(240,240,240,1) !important;
  transform: translateY(-2px);
}

/* Table text */
table, th, td {
  color: #111 !important;
  font-weight: 700 !important;
}

</style>
"""

# ----------------------
# Header
# ----------------------
st.markdown("""
<div class="glass-panel" style="text-align:center;">
    <h1>💊 Drug Prescription Classifier</h1>
    <p style="color:#111; font-weight:800; margin-top:6px; font-size:18px;">
        Predict drugs, explain why, generate reports, and monitor usage.
    </p>
</div>
""", unsafe_allow_html=True)


# ----------------------
# Emoji icons & drug info
# ----------------------
drug_images = {"drugA": "💊","drugB": "🩺","drugC": "⚗️","drugX": "🧬","drugY": "🩸"}
drug_details = {
    "drugA": {"name":"Drug A","use":"Used for normal BP and cholesterol.","mechanism":"Supports circulatory health.","side_effects":["Headache","Dry mouth"],"precautions":"Avoid alcohol.","dosage":"1 tablet daily."},
    "drugB": {"name":"Drug B","use":"For high blood pressure.","mechanism":"Relaxes blood vessels.","side_effects":["Low BP","Fatigue"],"precautions":"Not for pregnancy.","dosage":"1 tablet/day."},
    "drugC": {"name":"Drug C","use":"Electrolyte balance.","mechanism":"Balances Na/K.","side_effects":["Nausea"],"precautions":"Monitor levels.","dosage":"1–2 per day."},
    "drugX": {"name":"Drug X","use":"High cholesterol.","mechanism":"Reduces cholesterol.","side_effects":["Muscle pain"],"precautions":"Avoid high-fat foods.","dosage":"Evening dose."},
    "drugY": {"name":"Drug Y","use":"High BP + cholesterol.","mechanism":"Lowers BP & cholesterol.","side_effects":["Dizziness"],"precautions":"Regular BP checks.","dosage":"1 daily."}
}

# ----------------------
# Sample dataset + model utils
# ----------------------
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
    numeric = ['Age','Na','K']; categorical = ['Sex','BP','Cholesterol']
    pre = ColumnTransformer([("num", StandardScaler(), numeric),("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)])
    pipe = Pipeline([("pre", pre),("clf", LogisticRegression(max_iter=2000, multi_class='multinomial'))])
    pipe.fit(X,y)
    return pipe

# ----------------------
# Initialize session logs (in-memory)
# ----------------------
if 'logs' not in st.session_state:
    st.session_state['logs'] = []  # each entry: dict with timestamp, input, prediction, confidence, explanation

def append_log(entry: dict):
    st.session_state['logs'].append(entry)

# ----------------------
# Explain prediction: rule-based + class stats
# ----------------------
def explain_prediction(model, df_train, input_df, predicted):
    # model: trained pipeline; df_train: training dataframe; input_df: single-row df; predicted: predicted class label
    proba = None
    try:
        proba = model.predict_proba(input_df)[0]
        class_index = list(model.classes_).index(predicted)
        confidence = float(proba[class_index])
    except Exception:
        confidence = None

    # compute simple rule-based influences using class means and category modes
    stats = df_train.groupby('Drug').agg({'Age':'mean','Na':'mean','K':'mean'})
    cat_modes = df_train.groupby('Drug').agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)[['Sex','BP','Cholesterol']]
    reasons = []

    # numeric comparisons
    for col in ['Age','Na','K']:
        inp = float(input_df.iloc[0][col])
        mean_for_class = float(stats.loc[predicted][col])
        if inp >= mean_for_class:
            reasons.append(f"{col} ({inp}) is higher than average for {predicted} ({mean_for_class:.2f}) — increases chance.")
        else:
            reasons.append(f"{col} ({inp}) is lower than average for {predicted} ({mean_for_class:.2f}) — may lower chance.")

    # categorical matches
    for col in ['Sex','BP','Cholesterol']:
        inp = input_df.iloc[0][col]
        mode_val = cat_modes.loc[predicted][col]
        if inp == mode_val:
            reasons.append(f"{col} = {inp} matches typical {predicted} patients — supportive.")
        else:
            reasons.append(f"{col} = {inp} differs from typical {predicted} patients (usually {mode_val}) — neutral/contradictory.")

    # Compose explanation
    explanation = "\n".join(reasons)
    return confidence, explanation

# ----------------------
# PDF report generator (ReportLab)
# ----------------------
def create_pdf_report(patient_info: dict, prediction: str, confidence: float, explanation: str, drug_info: dict):
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
    conf_text = f"{confidence*100:.1f}%" if confidence is not None else "N/A"
    c.drawString(margin+10, y, f"Predicted Drug: {prediction} (Confidence: {conf_text})")
    y -= 18

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Explanation:")
    y -= 16
    c.setFont("Helvetica", 10)
    for line in explanation.split("\n"):
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

# ----------------------
# PAGES
# ----------------------

# Predict single patient
if page == "Predictor":
    st.markdown('<div class="glass-panel"><h3>Single Prediction</h3></div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel-2">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV (optional) - used for retraining (must have header row)", type=["csv"], key="up_single")
    st.markdown('</div>', unsafe_allow_html=True)

    df_train = pd.read_csv(uploaded) if uploaded else load_sample_df()
    model = train_model(df_train)

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("Enter Patient Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", ["F","M"])
        bp = st.selectbox("Blood Pressure", ["LOW","NORMAL","HIGH"])
    with col2:
        cholesterol = st.selectbox("Cholesterol", ["HIGH","NORMAL"])
        na = st.number_input("Sodium (Na)", value=0.7, format="%.4f")
        k = st.number_input("Potassium (K)", value=0.05, format="%.4f")

    if st.button("🔍 Predict"):
        input_df = pd.DataFrame([[age, sex, bp, cholesterol, na, k]], columns=['Age','Sex','BP','Cholesterol','Na','K'])
        pred = model.predict(input_df)[0]
        try:
            confidence, explanation = explain_prediction(model, df_train, input_df, pred)
        except Exception:
            confidence, explanation = None, "No detailed explanation available."

        st.success(f"💊 Predicted Drug: **{pred}**")
        if confidence is not None:
            st.info(f"Model confidence: **{confidence*100:.1f}%**")
        st.markdown("**Why this prediction?**")
        st.write(explanation)

        # append to session logs
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "Age": age, "Sex": sex, "BP": bp, "Cholesterol": cholesterol, "Na": na, "K": k,
            "prediction": pred, "confidence": confidence if confidence is not None else None,
            "explanation": explanation
        }
        append_log(log_entry)

        # PDF Report download
        if st.button("📄 Generate PDF Medical Report"):
            patient_info = {"Age": age, "Sex": sex, "BP": bp, "Cholesterol": cholesterol, "Na": na, "K": k}
            pdf_buf = create_pdf_report(patient_info, pred, confidence if confidence is not None else 0.0, explanation, drug_details.get(pred, {}))
            st.download_button("📥 Download PDF Report", data=pdf_buf, file_name=f"report_{pred}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
    st.markdown('</div>', unsafe_allow_html=True)

# Drug information page (unchanged)
if page == "Drug Information":
    st.markdown('<div class="glass-panel"><h3>Drug Information</h3></div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel-2">', unsafe_allow_html=True)
    choice = st.selectbox("Select Drug", list(drug_details.keys()))
    st.markdown('</div>', unsafe_allow_html=True)
    info = drug_details[choice]
    icon = drug_images[choice]
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown(f"<h2>{icon} {info['name']}</h2>", unsafe_allow_html=True)
    st.markdown("### Use"); st.write(info["use"])
    st.markdown("### Mechanism"); st.write(info["mechanism"])
    st.markdown("### Side Effects")
    for s in info["side_effects"]:
        st.markdown(f"- {s}")
    st.markdown("### Precautions"); st.write(info["precautions"])
    st.markdown("### Dosage"); st.write(info["dosage"])
    st.markdown('</div>', unsafe_allow_html=True)

# Bulk prediction (unchanged except logs)
if page == "Bulk Prediction":
    st.markdown('<div class="glass-panel"><h3>Bulk Prediction</h3></div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel-2">', unsafe_allow_html=True)
    csv_file = st.file_uploader("Upload CSV for bulk prediction (columns: Age,Sex,BP,Cholesterol,Na,K)", type=["csv"], key="bulk")
    st.markdown('</div>', unsafe_allow_html=True)
    if csv_file:
        bulk_df = pd.read_csv(csv_file)
        st.dataframe(bulk_df.head())
        required = ['Age','Sex','BP','Cholesterol','Na','K']; missing = [c for c in required if c not in bulk_df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            model = train_model(load_sample_df())
            bulk_df["Predicted_Drug"] = model.predict(bulk_df[required])
            st.success("Predictions complete")
            st.dataframe(bulk_df.head())
            # append logs for each row
            for _, row in bulk_df.iterrows():
                entry = {"timestamp": datetime.datetime.now().isoformat(),
                         "Age": int(row['Age']), "Sex": row['Sex'], "BP": row['BP'], "Cholesterol": row['Cholesterol'],
                         "Na": float(row['Na']), "K": float(row['K']), "prediction": row['Predicted_Drug']}
                append_log(entry)
            st.download_button("📥 Download predictions CSV", data=bulk_df.to_csv(index=False).encode("utf-8"),
                               file_name="bulk_predictions.csv", mime="text/csv")
    else:
        st.info("Upload CSV to begin bulk predictions.")

# Monitoring page: show logs, counts, timeline
if page == "Monitoring":
    st.markdown('<div class="glass-panel"><h3>Monitoring & Logs</h3></div>', unsafe_allow_html=True)
    logs = st.session_state.get('logs', [])
    if not logs:
        st.info("No predictions made yet this session.")
    else:
        logs_df = pd.DataFrame(logs)
        # quick stats
        st.markdown('<div class="glass-panel-2">', unsafe_allow_html=True)
        st.write("### Quick Stats")
        counts = logs_df['prediction'].value_counts()
        st.write("Most predicted drugs:")
        st.bar_chart(counts)
        st.write("Total predictions this session:", len(logs_df))
        st.markdown('</div>', unsafe_allow_html=True)

        # timeline
        st.write("### Prediction timeline")
        timeline = pd.to_datetime(logs_df['timestamp']).dt.floor('min').value_counts().sort_index()
        st.line_chart(timeline)

        # show logs table
        st.write("### Logs (most recent first)")
        st.dataframe(logs_df.sort_values(by='timestamp', ascending=False).reset_index(drop=True))

        # download logs
        st.download_button("📥 Download Logs CSV", data=logs_df.to_csv(index=False).encode("utf-8"),
                           file_name="session_prediction_logs.csv", mime="text/csv")

# About page
if page == "About":
    st.markdown('<div class="glass-panel"><h3>About</h3></div>', unsafe_allow_html=True)
    st.markdown("""
    ### Drug Prescription Classifier
    Predicts drugs using simple ML. This demo includes:
    - Single prediction w/ explanation
    - Bulk CSV predictions
    - Session monitoring (charts)
    - PDF medical report generation
    """)
    st.markdown("### Note on persistence")
    st.write("Logs are session-based (in-memory). For production, connect a DB or cloud storage for persistent monitoring.")

# Footer
st.caption("Built with ❤️ — enhanced with monitoring, explanations, and PDF reports.")
