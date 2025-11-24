# app.py - Smart Drug Shield (FINAL)
# Multi-user hashed login (no OTP), loads dataset from local /mnt/data/Drug.csv if present,
# otherwise falls back to the GitHub RAW URL you provided earlier.
# Includes multiple ML models, confidence, top-3, and drug information.

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import secrets
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Optional boosters (will be used only if installed)
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

# -------------------------------------------------
# App config & title
# -------------------------------------------------
st.set_page_config(page_title="🛡 Smart Drug Shield", page_icon="💊", layout="centered")
st.markdown("""
    <h1 style='text-align:center; font-size:40px; font-weight:900; margin-top:-10px;'>
        🛡 Smart Drug Shield
    </h1>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Simple CSS for glass effect and theme
# -------------------------------------------------
BASE_CSS = """
<style>
.glass-panel { backdrop-filter: blur(6px); background: rgba(255,255,255,0.92); border-radius:12px; padding:14px; margin-bottom:16px; }
h1,h2,h3,h4 { font-weight:800; }
.stButton>button { border-radius:8px !important; padding:8px 12px !important; }
</style>
"""
DARK_CSS = """
<style>
.glass-panel { background: rgba(20,20,25,0.6) !important; color: #fff !important; }
h1,h2,h3,h4 { color: #eaf2ff !important; }
.stButton>button { color:#fff !important; }
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# -------------------------------------------------
# Users with hashed passwords (SHA-256 + salt)
# Provided user list (confirmed by you):
# admin, Admin@123
# manasa, Manasa@2005
# doctor, Doctor@123
# student, Student@123
# -------------------------------------------------
_SALT = "a9f5b3c7"
def hash_password(password: str) -> str:
    return hashlib.sha256((_SALT + password).encode()).hexdigest()

USERS = {
    "admin": "2c6e86244d7f669c447d4353bdca3fab2d1cc73f5c51a406fb0e6266b0f85e63",
    "manasa": "3fb682bc3163bfdf80909311dfc86ad848f1e8ab76587c7b8082fbbe6d41ff3c",
    "doctor": "56860e5d0f26d1f79ce911557299dc8ba719a3f0a5f7f08ce73825063ea0f29e",
    "student": "02d6b184749eea90c563d6c9286c99c2a12c2fbbab549f7ee4df25fcbaf71c86"
}

# -------------------------------------------------
# Authentication (simple username + password)
# -------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None

def login_page():
    st.markdown('<div class="glass-panel" style="max-width:720px; margin:auto;">', unsafe_allow_html=True)
    st.subheader("🔒 Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    col1, col2 = st.columns([1,1])
    with col1:
        login_btn = st.button("Login")
    with col2:
        reset_btn = st.button("Clear")

    if reset_btn:
        st.experimental_rerun()

    if login_btn:
        if user in USERS and secrets.compare_digest(USERS[user], hash_password(pwd)):
            st.session_state["authenticated"] = True
            st.session_state["username"] = user
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

if not st.session_state["authenticated"]:
    login_page()

# -------------------------------------------------
# Sidebar: navigation & theme
# -------------------------------------------------
with st.sidebar:
    st.header(f"Welcome, {st.session_state.get('username')}")
    theme_choice = st.radio("🌗 Theme Mode", ["Light Mode", "Dark Mode"], index=0)
    page = st.radio("📄 Navigate", ["Predictor", "Drug Information", "Admin", "About"], index=0)
    st.markdown("---")
    st.markdown("Data source: either local `/mnt/data/Drug.csv` (if present) or GitHub RAW")
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.experimental_rerun()

if theme_choice == "Dark Mode":
    st.markdown(DARK_CSS, unsafe_allow_html=True)

# -------------------------------------------------
# Dataset loading: try local path first, else GitHub RAW
# (Developer note: local path seen in conversation: /mnt/data/Drug.csv)
# Raw GitHub URL (user provided earlier)
# -------------------------------------------------
LOCAL_PATH = "/mnt/data/Drug.csv"
GITHUB_RAW = "https://raw.githubusercontent.com/HSMANASA24/drug-prediction-app/c476f30acf26ddc14b6b4a7eb796786c23a23edd/Drug.csv"

@st.cache_data
def load_dataset():
    # prefer uploaded local file if exists ( Streamlit Cloud might not have it )
    if os.path.exists(LOCAL_PATH):
        df = pd.read_csv(LOCAL_PATH)
    else:
        df = pd.read_csv(GITHUB_RAW)
    # Ensure consistent columns
    df.columns = [c.strip() for c in df.columns]
    # Map dataset codes to user-chosen real names (Option B)
    mapping = {
        "drugA": "Amlodipine",
        "drugB": "Atenolol",
        "drugC": "ORS-K",
        "drugX": "Atorvastatin",
        "drugY": "Losartan"
    }
    if "Drug" in df.columns:
        df["Drug"] = df["Drug"].map(mapping).fillna(df["Drug"])
    return df

try:
    df_full = load_dataset()
except Exception as e:
    st.error("Failed to load dataset: " + str(e))
    st.stop()

st.sidebar.markdown(f"Rows: **{df_full.shape[0]}**  |  Columns: **{df_full.shape[1]}**")

# -------------------------------------------------
# Drug details (as requested)
# -------------------------------------------------
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

# -------------------------------------------------
# ML training helper
# -------------------------------------------------
# handle OneHotEncoder param compatibility
def onehot_encoder_factory():
    # sklearn >=1.2 uses sparse_output, earlier versions use sparse
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown='ignore')

@st.cache_resource
def build_and_train(model_name: str, df: pd.DataFrame):
    df = df.dropna().copy()
    X = df[['Age','Sex','BP','Cholesterol','Na','K']]
    y = df['Drug']

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), ['Age','Na','K']),
        ("cat", onehot_encoder_factory(), ['Sex','BP','Cholesterol'])
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
        raise ValueError("Unknown model: " + str(model_name))

    pipe = Pipeline([("pre", preprocessor), ("clf", models[model_name])])
    pipe.fit(X, y)
    return pipe

# -------------------------------------------------
# Predictor page
# -------------------------------------------------
if page == "Predictor":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("Single Prediction")

    available_models = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"]
    if HAS_XGB:
        available_models.append("XGBoost")
    if HAS_LGB:
        available_models.append("LightGBM")

    model_choice = st.selectbox("Select Model", available_models, index=0)

    with st.spinner("Training model on dataset..."):
        try:
            model = build_and_train(model_choice, df_full)
        except Exception as e:
            st.error("Training failed: " + str(e))
            st.stop()

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", ["F","M"])
        bp = st.selectbox("Blood Pressure (BP)", ["LOW","NORMAL","HIGH"])
    with col2:
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
                top1_idx = int(sorted_idx[0])
                top1_label = model.classes_[top1_idx]
                top1_conf = float(proba[top1_idx]) * 100.0
                st.success(f"Predicted Drug: {top1_label} ({top1_conf:.2f}% confidence)")

                st.write("Top predictions:")
                for i in range(min(3, len(sorted_idx))):
                    idx = int(sorted_idx[i])
                    label = model.classes_[idx]
                    prob_pct = proba[idx] * 100.0
                    st.write(f"{i+1}. {label} — {prob_pct:.2f}%")

                if top1_conf >= 80:
                    st.info("Confidence: High ✅")
                elif top1_conf >= 60:
                    st.info("Confidence: Moderate ⚠️")
                elif top1_conf >= 40:
                    st.warning("Confidence: Low ⚠️ Consider review")
                else:
                    st.error("Confidence: Very Low ❌ Seek further checks")
            else:
                st.success("Predicted Drug: " + str(pred))
                st.info("Selected algorithm does not provide probabilities.")

            explanation = (
                "The model predicted " + str(pred) + " because:\n"
                "- Age: " + str(age) + "\n"
                "- Sex: " + str(sex) + "\n"
                "- BP: " + str(bp) + "\n"
                "- Cholesterol: " + str(chol) + "\n"
                "- Sodium (Na): " + str(na) + "\n"
                "- Potassium (K): " + str(k)
            )
            st.info(explanation)

            if pred in drug_details:
                dd = drug_details[pred]
                st.markdown("---")
                st.markdown(f"### About {pred}")
                st.markdown(f"**Use:** {dd['use']}")
                st.markdown(f"**Mechanism:** {dd['mechanism']}")
                st.markdown(f"**Side Effects:** {', '.join(dd['side_effects'])}")
                st.markdown(f"**Precautions:** {dd['precautions']}")
                st.markdown(f"**Dosage:** {dd['dosage']}")
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# Drug Information page
# -------------------------------------------------
if page == "Drug Information":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("💊 Drug Information")
    for name, info in drug_details.items():
        with st.expander(f"📌 {name}"):
            st.markdown(f"**Use:** {info['use']}")
            st.markdown(f"**Mechanism:** {info['mechanism']}")
            st.markdown(f"**Side Effects:** {', '.join(info['side_effects'])}")
            st.markdown(f"**Precautions:** {info['precautions']}")
            st.markdown(f"**Dosage:** {info['dosage']}")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# Admin page
# -------------------------------------------------
if page == "Admin":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("👤 Admin — User Management (in-memory)")
    st.write("Current users:")
    for u in USERS.keys():
        st.write("•", u)
    st.markdown("---")
    st.write("### Add New User (in-memory)")
    new_user = st.text_input("Username", key="add_user")
    new_pass = st.text_input("Password", type="password", key="add_pass")
    if st.button("Add User"):
        if not new_user or not new_pass:
            st.error("Provide username and password.")
        elif new_user in USERS:
            st.error("User already exists.")
        else:
            USERS[new_user] = hash_password(new_pass)
            st.success("User added (in-memory).")
            st.experimental_rerun()
    st.markdown("---")
    st.write("### Remove User")
    remove_user = st.selectbox("Select user to remove", list(USERS.keys()), key="remove_user_select")
    if st.button("Remove User"):
        if remove_user == "admin":
            st.error("Cannot remove main admin.")
        else:
            USERS.pop(remove_user, None)
            st.success("User removed.")
            st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# About page
# -------------------------------------------------
if page == "About":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("ℹ️ About Smart Drug Shield")
    st.markdown("""
    Smart Drug Shield is an educational demo that trains classification models to predict an appropriate drug
    given patient features (Age, Sex, BP, Cholesterol, Na, K). The dataset is loaded from your repository
    (GitHub RAW) or from local `/mnt/data/Drug.csv` if present.

    **Notes**
    - Drug labels from CSV are mapped to clinical names.
    - This is for demonstration / learning and not a clinical decision tool.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
