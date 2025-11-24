import streamlit as st
import pandas as pd
import io
import datetime
import hashlib
import secrets
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# -------------------------
# App configuration
# -------------------------
st.set_page_config(page_title="Drug Prescription Classifier", page_icon="💊", layout="centered")

# -------------------------
# Simple secure password utilities
# -------------------------
# Use a server-side secret salt (in production, keep this secret and out of source)
_SALT = "a9f5b3c7"  # change this to a secure random value for production

def hash_password(password: str) -> str:
    """Return a salted SHA-256 hex digest of the password."""
    pw = (_SALT + password).encode('utf-8')
    return hashlib.sha256(pw).hexdigest()

# Default admin user (username: admin). Password: admin123 (hashed)
USERS = {
    "admin": hash_password("admin123")
}

# -------------------------
# CSS: Light + Dark themes (titles bold only)
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

# Dark CSS (Modern Gray Mode)
dark_css = """
<style>
body { background-color: #1c1c1c !important; color: #ffffff !important; }
section[data-testid="stSidebar"] { background-color: rgba(255,255,255,0.97) !important; color: #111 !important; }
.glass-panel, .glass-panel-2 { backdrop-filter: blur(12px); background: rgba(40,40,40,0.55) !important; border-radius: 16px; padding: 20px; border: 1px solid rgba(255,255,255,0.08); color:#ffffff !important; }
html, body, div, p, span, label { color:#ffffff !important; font-weight:400 !important; }
h1, h2, h3, h4 { font-weight:800 !important; color:#ffffff !important; }
input, select, textarea { background: rgba(255,255,255,0.12) !important; border-radius: 10px !important; border: 1px solid rgba(255,255,255,0.25) !important; color:#ffffff !important; }
input::placeholder { color: #ddd !important; }
.stButton>button { background: rgba(255,255,255,0.15) !important; color:#ffffff !important; border-radius: 10px !important; border: 1px solid rgba(255,255,255,0.35); font-weight:600 !important; }
.stButton>button:hover { background: rgba(255,255,255,0.25) !important; transform: translateY(-2px); }
table, th, td { color:#ffffff !important; }
</style>
"""

# -------------------------
# Helper: sample data, model training, pdf, explanation
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
def train_model(df, model_name='Logistic Regression'):
    X = df[['Age','Sex','BP','Cholesterol','Na','K']]
    y = df['Drug']

    numeric = ['Age','Na','K']
    categorical = ['Sex','BP','Cholesterol']

    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
    ])

    if model_name == 'Logistic Regression':
        clf = LogisticRegression(max_iter=2000, multi_class='multinomial')
    elif model_name == 'KNN':
        clf = KNeighborsClassifier()
    elif model_name == 'Decision Tree':
        clf = DecisionTreeClassifier()
    elif model_name == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=100)
    elif model_name == 'SVM':
        clf = SVC(probability=True)
    else:
        clf = LogisticRegression(max_iter=2000, multi_class='multinomial')

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)
    return pipe

# Explanation (rule-based)
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

    return "\n".join(reasons)

# PDF report generator
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
    if explanation:
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

# -------------------------
# Drug info dataset
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
# Authentication & session handling
# -------------------------
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
    st.session_state['username'] = None


def login_user(username: str, password: str) -> bool:
    """Validate username + password against USERS (hashed)."""
    if username in USERS:
        hashed = USERS[username]
        return secrets.compare_digest(hashed, hash_password(password))
    return False


def require_login():
    """Show login screen and block access until authenticated."""
    st.markdown('<div class="glass-panel" style="max-width:600px; margin:auto;">', unsafe_allow_html=True)
    st.title("🔒 Admin Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Login"):
            if login_user(user, pwd):
                st.session_state['authenticated'] = True
                st.session_state['username'] = user
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")
    with col2:
        if st.button("Forgot Password"):
            st.info("If you forgot the admin password, update the USERS dict in the code or reset on the server.")

    st.markdown('</div>', unsafe_allow_html=True)

# If not authenticated, show login and stop
if not st.session_state['authenticated']:
    st.markdown(base_css, unsafe_allow_html=True)
    st.markdown(dark_css, unsafe_allow_html=True)  # ensure login readable in dark
    require_login()
    st.stop()

# -------------------------
# Main app (authenticated users only)
# -------------------------
# Apply theme (light by default; allow user to toggle if desired)
with st.sidebar:
    st.header(f"Welcome, {st.session_state.get('username')}")
    theme_choice = st.radio("🌗 Theme Mode", ["Light Mode", "Dark Mode"], index=0, key="theme_choice")
    page = st.radio("📄 Navigate", ["Predictor", "Drug Information", "Bulk Prediction", "Monitoring", "Admin", "About"], key="nav_main")
    st.markdown("---")
    if st.button("Logout"):
        st.session_state['authenticated'] = False
        st.session_state['username'] = None
        st.experimental_rerun()

# Apply selected theme
st.markdown(base_css, unsafe_allow_html=True)
if theme_choice == 'Dark Mode':
    st.markdown(dark_css, unsafe_allow_html=True)

# Header
st.markdown("""
<div class="glass-panel" style="text-align:center;">
    <h1>💊 Drug Prescription Classifier</h1>
    <p style="color:#111; font-weight:600; margin-top:6px; font-size:18px;">Predict drugs, explain why, generate reports, and monitor usage.</p>
</div>
""", unsafe_allow_html=True)

# Session logs init
if 'logs' not in st.session_state:
    st.session_state['logs'] = []

# Predictor page
if page == "Predictor":
    st.markdown('<div class="glass-panel"><h2>Single Prediction</h2></div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-panel-2">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV (optional) for custom training", type=["csv"], key="predict_upload")
    st.markdown('</div>', unsafe_allow_html=True)

    df_train = pd.read_csv(uploaded) if uploaded else load_sample_df()

    # Model selection dropdown
    model_name = st.selectbox("Select model", ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"], index=0)
    model = train_model(df_train, model_name=model_name)

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
        for line in explanation.split("\n"):
            st.write("- ", line)

        append_log({
            "timestamp": datetime.datetime.now().isoformat(),
            "Age": age, "Sex": sex, "BP": bp, "Cholesterol": cholesterol, "Na": na, "K": k,
            "prediction": pred, "model": model_name, "user": st.session_state.get('username')
        })

        if st.button("📄 Download PDF Report"):
            patient_info = {"Age": age, "Sex": sex, "BP": bp, "Cholesterol": cholesterol, "Na": na, "K": k}
            pdf_buf = create_pdf_report(patient_info, pred, explanation, drug_details.get(pred, {}))
            st.download_button("📥 Download PDF", data=pdf_buf, file_name=f"report_{pred}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
    st.markdown('</div>', unsafe_allow_html=True)

# Drug Information
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

# Bulk Prediction
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

# Monitoring
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

# Admin page
if page == "Admin":
    st.markdown('<div class="glass-panel"><h2>Admin Panel</h2></div>', unsafe_allow_html=True)
    st.write("Manage users and logs")

    st.markdown('### Create new user (username + password)')
    new_user = st.text_input("New username")
    new_pass = st.text_input("New password", type="password")
    if st.button("Create user"):
        if new_user in USERS:
            st.error("User already exists")
        elif not new_user or not new_pass:
            st.error("Provide username and password")
        else:
            USERS[new_user] = hash_password(new_pass)
            st.success(f"User {new_user} created")

    st.markdown('### View & clear logs')
    logs = st.session_state.get('logs', [])
    if logs:
        st.dataframe(pd.DataFrame(logs))
        if st.button("Clear logs"):
            st.session_state['logs'] = []
            st.success("Logs cleared")
    else:
        st.info("No logs yet")

# About
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
