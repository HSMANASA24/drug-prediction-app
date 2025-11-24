# app.py - Smart Drug Shield (RAW GitHub + Email OTP integrated)

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import secrets
import smtplib
import ssl
import random
import time
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Optional libraries
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

# ---------------------------
# App config and title
# ---------------------------
st.set_page_config(page_title="🛡 Smart Drug Shield", page_icon="💊", layout="centered")
st.markdown("""
    <h1 style='text-align:center; font-size:40px; font-weight:900; margin-top:-10px;'>
        🛡 Smart Drug Shield
    </h1>
""", unsafe_allow_html=True)

# ---------------------------
# CSS
# ---------------------------
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

# ---------------------------
# Simple admin users (in-memory)
# ---------------------------
_SALT = "a9f5b3c7"
def hash_password(password: str) -> str:
    return hashlib.sha256((_SALT + password).encode()).hexdigest()

# Default admin user (can be extended by Admin page)
USERS = {
    "admin": hash_password("admin123")
}

# ---------------------------
# Email OTP config & helpers
# ---------------------------
# RAW GitHub URL for your dataset
RAW_DATA_URL = "https://raw.githubusercontent.com/HSMANASA24/drug-prediction-app/c476f30acf26ddc14b6b4a7eb796786c23a23edd/Drug.csv"

def get_email_credentials():
    # Prefer Streamlit secrets
    try:
        email_addr = st.secrets["EMAIL_ADDRESS"]
        email_app_pass = st.secrets["EMAIL_APP_PASSWORD"]
    except Exception:
        email_addr = os.environ.get("EMAIL_ADDRESS")
        email_app_pass = os.environ.get("EMAIL_APP_PASSWORD")
    return email_addr, email_app_pass

def send_otp_via_gmail(to_email: str, otp_code: str):
    sender, app_pass = get_email_credentials()
    if not sender or not app_pass:
        raise RuntimeError("Email credentials not found. Set EMAIL_ADDRESS and EMAIL_APP_PASSWORD in Streamlit secrets.")
    message = f"""From: Smart Drug Shield <{sender}>
To: {to_email}
Subject: Your Smart Drug Shield OTP

Your one-time password (OTP) is: {otp_code}

This OTP is valid for 5 minutes.
"""
    context = ssl.create_default_context()
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.ehlo()
        server.starttls(context=context)
        server.login(sender, app_pass)
        server.sendmail(sender, [to_email], message)

def generate_otp():
    return f"{random.randint(100000, 999999)}"

# Initialize OTP session vars
if "otp_sent" not in st.session_state:
    st.session_state["otp_sent"] = False
if "otp_code" not in st.session_state:
    st.session_state["otp_code"] = None
if "otp_timestamp" not in st.session_state:
    st.session_state["otp_timestamp"] = None
if "auth_email" not in st.session_state:
    st.session_state["auth_email"] = None

# Authenticated flag and username
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None

def show_email_otp_login():
    st.markdown('<div class="glass-panel" style="max-width:700px; margin:auto;">', unsafe_allow_html=True)
    st.subheader("🔒 Login with Email + OTP")
    email = st.text_input("Enter your email address", value=st.session_state.get("auth_email", ""))
    col1, col2 = st.columns([1,1])
    with col1:
        send_btn = st.button("Send OTP")
    with col2:
        verify_btn = st.button("Verify OTP")
    if send_btn:
        if not email:
            st.error("Please enter an email address.")
        else:
            otp = generate_otp()
            try:
                send_otp_via_gmail(email, otp)
                st.session_state["otp_sent"] = True
                st.session_state["otp_code"] = otp
                st.session_state["otp_timestamp"] = time.time()
                st.session_state["auth_email"] = email
                st.success(f"OTP sent to {email}. (Check inbox / spam)")
            except Exception as e:
                st.error("Failed to send OTP: " + str(e))
    if st.session_state.get("otp_sent"):
        user_otp = st.text_input("Enter the OTP you received", key="user_otp")
        if verify_btn:
            if not user_otp:
                st.error("Enter the OTP first.")
            else:
                now = time.time()
                sent_at = st.session_state.get("otp_timestamp") or 0
                if now - sent_at > 300:
                    st.error("OTP expired. Request a new one.")
                    st.session_state["otp_sent"] = False
                    st.session_state["otp_code"] = None
                    st.session_state["otp_timestamp"] = None
                elif user_otp == st.session_state.get("otp_code"):
                    st.success("OTP verified — login successful.")
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = email.split("@")[0]
                    # clear OTP values
                    st.session_state["otp_sent"] = False
                    st.session_state["otp_code"] = None
                    st.session_state["otp_timestamp"] = None
                    st.experimental_rerun()
                else:
                    st.error("Incorrect OTP. Try again.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# If not authenticated, show login
if not st.session_state["authenticated"]:
    show_email_otp_login()

# ---------------------------
# Sidebar: navigation & theme
# ---------------------------
with st.sidebar:
    st.header(f"Welcome, {st.session_state.get('username','user')}")
    theme_choice = st.radio("🌗 Theme Mode", ["Light Mode", "Dark Mode"], index=0)
    page = st.radio("📄 Navigate", ["Predictor", "Drug Information", "Admin", "About"], index=0)
    st.markdown("---")
    st.markdown("📄 **Dataset Source:** GitHub RAW")
    st.markdown(f"`{RAW_DATA_URL}`")
    st.markdown("---")
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.experimental_rerun()

# Apply dark CSS if user selected dark mode
if theme_choice == "Dark Mode":
    st.markdown(DARK_CSS, unsafe_allow_html=True)

# ---------------------------
# Data loading
# ---------------------------
@st.cache_data
def load_and_prepare_github(url: str):
    df = pd.read_csv(url)
    # Map codes to Option B names
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
    df_full = load_and_prepare_github(RAW_DATA_URL)
except Exception as e:
    st.error("Failed to load dataset from GitHub RAW URL: " + str(e))
    st.stop()

# Quick dataset info
st.sidebar.markdown(f"Rows: **{df_full.shape[0]}**  |  Columns: **{df_full.shape[1]}**")

# ---------------------------
# Drug information dictionary
# ---------------------------
drug_details = {
    "Amlodipine": {
        "use": "Lowers BP by relaxing blood vessels (calcium channel blocker).",
        "mechanism": "Calcium channel blocker — vasodilation.",
        "side_effects": ["Dizziness", "Edema", "Flushing"],
        "precautions": "Monitor BP and avoid severe hypotension.",
        "dosage": "Typical: 5–10 mg once daily."
    },
    "Atenolol": {
        "use": "BP control and heart rate reduction (beta-blocker).",
        "mechanism": "Selective β1-blocker.",
        "side_effects": ["Fatigue", "Bradycardia"],
        "precautions": "Avoid in asthma; monitor HR.",
        "dosage": "Typical: 50 mg once daily."
    },
    "ORS-K": {
        "use": "Replenish electrolytes (Na/K).",
        "mechanism": "Restores sodium and potassium balance.",
        "side_effects": ["Nausea"],
        "precautions": "Monitor electrolytes in severe cases.",
        "dosage": "Per dehydration/electrolyte protocol."
    },
    "Atorvastatin": {
        "use": "Lowers LDL cholesterol.",
        "mechanism": "HMG-CoA reductase inhibitor (statin).",
        "side_effects": ["Muscle pain", "Liver enzyme rise"],
        "precautions": "Avoid in pregnancy; monitor LFTs.",
        "dosage": "Typical: 10–20 mg, usually in evening."
    },
    "Losartan": {
        "use": "Treat high blood pressure (ARB).",
        "mechanism": "Blocks angiotensin II receptors.",
        "side_effects": ["Dizziness", "Increased potassium"],
        "precautions": "Avoid in pregnancy; monitor potassium.",
        "dosage": "Typical: 25–50 mg once daily."
    }
}

# ---------------------------
# ML training helper
# ---------------------------
@st.cache_resource
def build_and_train(model_name: str, df: pd.DataFrame):
    df = df.dropna().copy()
    X = df[['Age','Sex','BP','Cholesterol','Na','K']]
    y = df['Drug']

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), ['Age','Na','K']),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['Sex','BP','Cholesterol'])
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

# ---------------------------
# Predictor page
# ---------------------------
if page == "Predictor":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("Single Prediction")

    # available model list
    available_models = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"]
    if HAS_XGB:
        available_models.append("XGBoost")
    if HAS_LGB:
        available_models.append("LightGBM")

    model_choice = st.selectbox("Select Model", available_models, index=0)

    with st.spinner("Training model on your dataset..."):
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

        # Try to get probabilities
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

            # Show drug info
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

# ---------------------------
# Drug Information page
# ---------------------------
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

# ---------------------------
# Admin page
# ---------------------------
if page == "Admin":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("👤 Admin — User Management")
    st.write("Current users (in-memory):")
    for u in USERS.keys():
        st.write("•", u)
    st.markdown("---")
    st.write("### Add New User")
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

# ---------------------------
# About page
# ---------------------------
if page == "About":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("ℹ️ About Smart Drug Shield")
    st.markdown("""
    Smart Drug Shield is an educational demo app that trains classification models on a patient dataset
    (Age, Sex, BP, Cholesterol, Na, K) to predict an appropriate drug.  
    Dataset source: **GitHub RAW** (the app loads the CSV from your repository).
    
    **Notes**
    - Drug labels from the CSV are mapped to clinical names (Amlodipine, Atenolol, ORS-K, Atorvastatin, Losartan).
    - Email+OTP login requires `EMAIL_ADDRESS` and `EMAIL_APP_PASSWORD` set in Streamlit Secrets.
    - This is a demonstration tool — not for real clinical decisions.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
