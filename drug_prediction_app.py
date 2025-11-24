# app.py - Smart Drug Shield (AI Healthcare Dashboard - Soft Medical Glow)
import streamlit as st
import pandas as pd
import numpy as np
import os
import secrets
from datetime import datetime

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Optional boosters (safe import)
HAS_XGB = True
HAS_LGB = True
try:
    from xgboost import XGBClassifier
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
except Exception:
    HAS_LGB = False

# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="🛡 Smart Drug Shield", page_icon="💊", layout="centered")

# Header (large, visible)
st.markdown("""
    <div style="width:100%; text-align:center; padding-top:8px; padding-bottom:4px;">
      <h1 style="margin:0; color:#003366; font-size:42px; font-weight:900;">
        🛡 Smart Drug Shield
      </h1>
      <p style="margin:0; color:#0066cc; font-weight:600;">AI-powered drug prescription classifier — Medical theme</p>
    </div>
""", unsafe_allow_html=True)

# -------------------------
# Medical theme CSS (Soft Medical Glow)
# -------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"]  { font-family: Inter, sans-serif; background: #F7F9FC; }
.glass { background: rgba(255,255,255,0.85); border-radius:14px; padding:18px; box-shadow: 0 6px 20px rgba(0,77,153,0.08); border:1px solid rgba(0,77,153,0.06); }
.header-gradient {
  background: linear-gradient(90deg, rgba(0,123,255,0.12), rgba(40,199,111,0.08));
  padding:12px; border-radius:12px;
}
.stButton>button { background: linear-gradient(90deg,#007BFF,#28C76F); color:white; border:none; padding:10px 18px; font-weight:700; border-radius:10px; box-shadow: 0 6px 18px rgba(34,139,230,0.12); }
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 10px 24px rgba(34,139,230,0.18); }
input, textarea, select { border-radius:10px; padding:8px; border:1px solid rgba(0,0,0,0.08); }
h2 { color:#003366; font-weight:800; }
.small-muted { color:#56677a; font-size:13px; }
.metric-card { border-radius:12px; padding:12px; background: white; box-shadow: 0 6px 18px rgba(2, 62, 138, 0.04); border:1px solid rgba(0,0,0,0.04); }
.tag { display:inline-block; padding:6px 10px; border-radius:999px; font-weight:700; color:#fff; }
.tag-blue { background: #007BFF; }
.tag-green { background: #28C76F; }
.conf-bar { height:12px; border-radius:8px; background: linear-gradient(90deg,#28C76F,#007BFF); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -------------------------
# Simple plain-text USERS (no hashing)
# -------------------------
USERS = {
    "admin": "Admin@123",
    "manasa": "Manasa@2005",
    "doctor": "Doctor@123",
    "student": "Student@123"
}

# -------------------------
# Session defaults
# -------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None

# -------------------------
# LOGIN PAGE (plain-text)
# -------------------------
def login_page():
    st.markdown('<div class="glass" style="max-width:720px; margin:auto;">', unsafe_allow_html=True)
    st.subheader("🔒 Admin Login")
    st.write("Enter your username and password to continue.")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    c1, c2 = st.columns([1,1])
    with c1:
        btn_login = st.button("Login")
    with c2:
        btn_clear = st.button("Clear")

    if btn_clear:
        # rerun to clear inputs
        st.rerun()

    if btn_login:
        if user in USERS and USERS[user] == pwd:
            st.session_state["authenticated"] = True
            st.session_state["username"] = user
            st.success(f"Welcome, {user} — logging you in...")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# require login
if not st.session_state["authenticated"]:
    login_page()

# -------------------------
# Sidebar (no dark mode)
# -------------------------
with st.sidebar:
    st.markdown('<div class="header-gradient">', unsafe_allow_html=True)
    st.header(f"Welcome, {st.session_state.get('username')}")
    st.markdown('</div>', unsafe_allow_html=True)

    page = st.radio("📄 Navigate", ["Predictor", "Drug Information", "Admin", "About"], index=0)
    st.markdown("---")
    st.markdown("**Data source:** local `/mnt/data/Drug.csv` (fallback to GitHub RAW if not found)")
    st.markdown("**Theme:** Medical (Blue + Green + White)")
    st.markdown("---")
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.rerun()

# -------------------------
# Data loading (local path primary)
# -------------------------
LOCAL_PATH = "/mnt/data/Drug.csv"   # <- local path used in this session (will be used as file URL if present)
GITHUB_RAW = "https://raw.githubusercontent.com/HSMANASA24/drug-prediction-app/c476f30acf26ddc14b6b4a7eb796786c23a23edd/Drug.csv"

@st.cache_data
def load_dataset():
    if os.path.exists(LOCAL_PATH):
        df = pd.read_csv(LOCAL_PATH)
        source = LOCAL_PATH
    else:
        df = pd.read_csv(GITHUB_RAW)
        source = GITHUB_RAW
    df.columns = [c.strip() for c in df.columns]
    mapping = {
        "drugA": "Amlodipine",
        "drugB": "Atenolol",
        "drugC": "ORS-K",
        "drugX": "Atorvastatin",
        "drugY": "Losartan"
    }
    if "Drug" in df.columns:
        df["Drug"] = df["Drug"].map(mapping).fillna(df["Drug"])
    return df, source

try:
    df_full, DATA_SOURCE = load_dataset()
except Exception as e:
    st.error("Failed to load dataset: " + str(e))
    st.stop()

st.sidebar.markdown(f"Rows: **{df_full.shape[0]}**  |  Columns: **{df_full.shape[1]}**")
st.sidebar.markdown(f"Source: `{DATA_SOURCE}`")

# -------------------------
# Drug information dictionary
# -------------------------
drug_details = {
    "Amlodipine": {
        "use": "Lowers blood pressure by relaxing blood vessels (calcium channel blocker).",
        "mechanism": "Calcium channel blocker that dilates peripheral arteries.",
        "side_effects": ["Dizziness", "Edema", "Flushing"],
        "precautions": "Monitor BP; caution with severe hypotension.",
        "dosage": "5–10 mg once daily (typical adult)."
    },
    "Atenolol": {
        "use": "Used for blood pressure control and heart rate reduction (beta-blocker).",
        "mechanism": "Selective β1-blocker reducing heart rate and cardiac output.",
        "side_effects": ["Fatigue", "Bradycardia", "Cold extremities"],
        "precautions": "Avoid in asthma; monitor heart rate.",
        "dosage": "50 mg once daily (adjust per clinical guidance)."
    },
    "ORS-K": {
        "use": "Oral rehydration / electrolyte replacement for sodium–potassium balance.",
        "mechanism": "Replenishes Na+ and K+ and maintains hydration.",
        "side_effects": ["Nausea", "Bloating"],
        "precautions": "Monitor electrolytes in severe cases.",
        "dosage": "As required during dehydration or imbalance."
    },
    "Atorvastatin": {
        "use": "Lowers LDL cholesterol and cardiovascular risk.",
        "mechanism": "HMG-CoA reductase inhibitor (statin).",
        "side_effects": ["Muscle pain", "Liver enzyme elevation"],
        "precautions": "Check liver enzymes; avoid during pregnancy.",
        "dosage": "10–20 mg in the evening (typical start)."
    },
    "Losartan": {
        "use": "Used to treat high blood pressure (angiotensin receptor blocker).",
        "mechanism": "Blocks angiotensin II receptors causing vasodilation.",
        "side_effects": ["Dizziness", "Increased potassium"],
        "precautions": "Avoid during pregnancy; monitor potassium.",
        "dosage": "25–50 mg once daily (adjust per clinical guidance)."
    }
}

# -------------------------
# OneHotEncoder compatibility helper
# -------------------------
def onehot_factory():
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown='ignore')

# -------------------------
# Train models helper
# -------------------------
@st.cache_resource
def build_and_train(model_name: str, df: pd.DataFrame):
    dfc = df.dropna().copy()
    X = dfc[['Age','Sex','BP','Cholesterol','Na','K']]
    y = dfc['Drug']
    pre = ColumnTransformer([
        ("num", StandardScaler(), ['Age','Na','K']),
        ("cat", onehot_factory(), ['Sex','BP','Cholesterol'])
    ])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True)
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    if HAS_LGB:
        models["LightGBM"] = LGBMClassifier()

    if model_name not in models:
        raise ValueError("Unknown model.")
    pipe = Pipeline([("pre", pre), ("clf", models[model_name])])
    pipe.fit(X, y)
    return pipe

# -------------------------
# Predictor page
# -------------------------
if page == "Predictor":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Single Prediction")
    st.markdown('<div class="small-muted">Enter patient details to predict the most suitable drug. Models trained on the dataset loaded above.</div>', unsafe_allow_html=True)

    models_list = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"]
    if HAS_XGB: models_list.append("XGBoost")
    if HAS_LGB: models_list.append("LightGBM")
    model_choice = st.selectbox("Select model", models_list, index=0)

    with st.spinner("Training model..."):
        try:
            model = build_and_train(model_choice, df_full)
        except Exception as e:
            st.error("Training failed: " + str(e))
            st.stop()

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 1, 120, 45)
        sex = st.selectbox("Sex", ["F","M"])
        bp = st.selectbox("Blood Pressure (BP)", ["LOW","NORMAL","HIGH"])
    with c2:
        chol = st.selectbox("Cholesterol", ["HIGH","NORMAL"])
        na = st.number_input("Sodium (Na)", format="%.6f", value=0.700000)
        k = st.number_input("Potassium (K)", format="%.6f", value=0.050000)

    if st.button("Predict"):
        input_df = pd.DataFrame([[age, sex, bp, chol, na, k]], columns=['Age','Sex','BP','Cholesterol','Na','K'])
        try:
            pred = model.predict(input_df)[0]
        except Exception as e:
            st.error("Prediction error: " + str(e))
            pred = None

        proba = None
        try:
            proba = model.predict_proba(input_df)[0]
        except Exception:
            proba = None

        if pred is not None:
            if proba is not None:
                sorted_idx = np.argsort(proba)[::-1]
                top_idx = int(sorted_idx[0])
                top_label = model.classes_[top_idx]
                top_conf = float(proba[top_idx]) * 100.0

                st.success(f"Predicted Drug: {top_label}  —  {top_conf:.2f}% confidence")
                # Confidence bar
                st.markdown("<div style='margin-top:8px; margin-bottom:8px;'><div style='width:100%; background:#eceff5; border-radius:8px; height:14px;'><div style='width:{}%; background:linear-gradient(90deg,#28C76F,#007BFF); height:100%; border-radius:8px;'></div></div></div>".format(min(max(top_conf,0),100)), unsafe_allow_html=True)

                st.write("Top predictions:")
                for i in range(min(3, len(sorted_idx))):
                    idx = int(sorted_idx[i])
                    label = model.classes_[idx]
                    pval = proba[idx] * 100.0
                    st.write(f"{i+1}. **{label}** — {pval:.2f}%")
                # confidence message
                if top_conf >= 80:
                    st.info("Confidence: High ✅")
                elif top_conf >= 60:
                    st.info("Confidence: Moderate ⚠️")
                elif top_conf >= 40:
                    st.warning("Confidence: Low ⚠️ Consider review")
                else:
                    st.error("Confidence: Very Low ❌ Seek further checks")
            else:
                st.success("Predicted Drug: " + str(pred))
                st.info("Selected algorithm does not support probabilities.")

            # short explanation
            explanation = (
                f"The model predicted {pred} based on the following input:\n"
                f"- Age: {age}\n- Sex: {sex}\n- BP: {bp}\n- Cholesterol: {chol}\n- Sodium (Na): {na}\n- Potassium (K): {k}"
            )
            st.markdown("**Why this prediction?**")
            st.info(explanation)

            # show drug info
            if pred in drug_details:
                dd = drug_details[pred]
                st.markdown("---")
                st.markdown(f"<div class='metric-card'><h3 style='margin:0;color:#003366'>{pred}</h3><p class='small-muted' style='margin:0'>{dd['use']}</p></div>", unsafe_allow_html=True)
                st.markdown(f"**Mechanism:** {dd['mechanism']}")
                st.markdown(f"**Side Effects:** {', '.join(dd['side_effects'])}")
                st.markdown(f"**Precautions:** {dd['precautions']}")
                st.markdown(f"**Dosage:** {dd['dosage']}")
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Drug Information page
# -------------------------
if page == "Drug Information":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("💊 Drug Information")
    st.write("Detailed descriptions of drugs available in the model.")
    for name, info in drug_details.items():
        with st.expander(f"📌 {name}"):
            st.markdown(f"**Use:** {info['use']}")
            st.markdown(f"**Mechanism:** {info['mechanism']}")
            st.markdown(f"**Side Effects:** {', '.join(info['side_effects'])}")
            st.markdown(f"**Precautions:** {info['precautions']}")
            st.markdown(f"**Dosage:** {info['dosage']}")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Admin page (in-memory users)
# -------------------------
if page == "Admin":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("👤 Admin — User Management (in-memory)")
    st.write("Current users:")
    for u in USERS.keys():
        st.write("•", u)
    st.markdown("---")
    st.write("### Add new user (in-memory)")
    new_user = st.text_input("Username", key="add_user")
    new_pass = st.text_input("Password", type="password", key="add_pass")
    if st.button("Add User"):
        if not new_user or not new_pass:
            st.error("Provide username and password.")
        elif new_user in USERS:
            st.error("User already exists.")
        else:
            USERS[new_user] = new_pass
            st.success("User added (in-memory).")
            st.rerun()
    st.markdown("---")
    st.write("### Remove user")
    removable = [u for u in USERS.keys() if u != "admin"]
    if removable:
        remove_user = st.selectbox("Select user to remove", removable, key="remove_user")
        if st.button("Remove User"):
            USERS.pop(remove_user, None)
            st.success("User removed.")
            st.rerun()
    else:
        st.info("No removable users.")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# About page
# -------------------------
if page == "About":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("ℹ️ About Smart Drug Shield")
    st.markdown(f"""
    **Smart Drug Shield** is an educational demo: an AI-driven drug prescription classifier built for learning and demonstration.
    - Dataset loaded from: `{DATA_SOURCE}`
    - Models available: Logistic Regression, KNN, Decision Tree, Random Forest, SVM {(' + XGBoost' if HAS_XGB else '')}{(' + LightGBM' if HAS_LGB else '')}
    - Designed with a medical AI theme (blue + green + white).
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# --- End of app.py ---
# -------------------------
