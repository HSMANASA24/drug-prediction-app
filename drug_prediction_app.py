import streamlit as st
import pandas as pd
import numpy as np
import os
import secrets

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# ====================================================
# APP CONFIG
# ====================================================
st.set_page_config(page_title="🛡 Smart Drug Shield", page_icon="💊", layout="centered")

st.markdown("""
    <h1 style='text-align:center; font-size:42px; font-weight:900; color:#0A3D62;'>
        🛡 Smart Drug Shield
    </h1>
    <p style='text-align:center; font-size:18px; color:#145A32; margin-top:-10px;'>
        AI-powered Drug Prescription Classifier — Medical Theme
    </p>
""", unsafe_allow_html=True)


# ====================================================
# MEDICAL THEME (BLUE + GREEN + WHITE)
# ====================================================
MEDICAL_CSS = """
<style>
body {
    background: linear-gradient(135deg, #e8f9ff, #d4fce5);
}
.glass-panel {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(8px);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 20px;
    border: 1px solid #bde4ff;
}
h1, h2, h3, h4 {
    color: #0A3D62 !important;
    font-weight: 800;
}
label, p, span, div {
    color: #0A3D62 !important;
}
.stButton>button {
    background-color: #0A3D62 !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
}
.stButton>button:hover {
    background-color: #145A32 !important;
}
</style>
"""
st.markdown(MEDICAL_CSS, unsafe_allow_html=True)


# ====================================================
# LOGIN SYSTEM (NO HASHING)
# ====================================================
USERS = {
    "admin": "Admin@123",
    "manasa": "Manasa@2005",
    "doctor": "Doctor@123",
    "student": "Student@123",
}

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "username" not in st.session_state:
    st.session_state["username"] = None


def login_page():
    st.markdown('<div class="glass-panel" style="max-width:700px; margin:auto;">', unsafe_allow_html=True)
    st.subheader("🔒 Smart Drug Shield Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user in USERS and USERS[user] == pwd:
            st.session_state["authenticated"] = True
            st.session_state["username"] = user
        else:
            st.error("❌ Invalid username or password")

    st.markdown("</div>", unsafe_allow_html=True)


# STOP HERE IF NOT LOGGED IN
if not st.session_state["authenticated"]:
    login_page()
    st.stop()


# ====================================================
# SIDEBAR
# ====================================================
with st.sidebar:
    st.header(f"Welcome, {st.session_state['username']}")
    page = st.radio("📄 Navigate", ["Predictor", "Drug Information", "Admin", "About"])

    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.experimental_rerun()


# ====================================================
# LOAD DATASET
# ====================================================
LOCAL_PATH = "/mnt/data/Drug.csv"
GITHUB_RAW = "https://raw.githubusercontent.com/HSMANASA24/drug-prediction-app/c476f30acf26ddc14b6b4a7eb796786c23a23edd/Drug.csv"

@st.cache_data
def load_dataset():
    if os.path.exists(LOCAL_PATH):
        df = pd.read_csv(LOCAL_PATH)
    else:
        df = pd.read_csv(GITHUB_RAW)

    df.columns = [c.strip() for c in df.columns]

    # Map dataset codes to real drug names
    mapping = {
        "drugA": "Amlodipine",
        "drugB": "Atenolol",
        "drugC": "ORS-K",
        "drugX": "Atorvastatin",
        "drugY": "Losartan",
    }
    df["Drug"] = df["Drug"].map(mapping).fillna(df["Drug"])
    return df


df_full = load_dataset()


# ====================================================
# DRUG DETAILS
# ====================================================
drug_details = {
    "Amlodipine": {
        "use": "Lowers blood pressure by relaxing blood vessels.",
        "mechanism": "Calcium channel blocker.",
        "side_effects": ["Dizziness", "Swelling", "Headache"],
        "precautions": "Monitor BP regularly.",
        "dosage": "5–10 mg daily"
    },
    "Atenolol": {
        "use": "Controls blood pressure and heart rate.",
        "mechanism": "Beta-blocker.",
        "side_effects": ["Cold hands", "Fatigue"],
        "precautions": "Avoid in asthma.",
        "dosage": "50 mg daily"
    },
    "ORS-K": {
        "use": "Electrolyte replenishment.",
        "mechanism": "Rehydrates body with Na+ and K+.",
        "side_effects": ["Nausea"],
        "precautions": "Monitor electrolytes.",
        "dosage": "As required"
    },
    "Atorvastatin": {
        "use": "Lowers cholesterol.",
        "mechanism": "Statin drug.",
        "side_effects": ["Muscle pain", "Liver effects"],
        "precautions": "Avoid alcohol.",
        "dosage": "10–20 mg daily"
    },
    "Losartan": {
        "use": "Treats hypertension.",
        "mechanism": "Angiotensin-II receptor blocker.",
        "side_effects": ["Dizziness", "High potassium"],
        "precautions": "Avoid pregnancy.",
        "dosage": "25–50 mg daily"
    }
}


# ====================================================
# ML MODEL TRAINING
# ====================================================
def build_model(df, model_name):
    X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na', 'K']]
    y = df['Drug']

    pre = ColumnTransformer([
        ("num", StandardScaler(), ['Age', 'Na', 'K']),
        ("cat", OneHotEncoder(sparse_output=False), ['Sex', 'BP', 'Cholesterol'])
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
    }

    pipe = Pipeline([("pre", pre), ("model", models[model_name])])
    pipe.fit(X, y)

    return pipe


# ====================================================
# PAGE: PREDICTOR
# ====================================================
if page == "Predictor":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("🔍 Single Prediction — AI Drug Classifier")

    model_name = st.selectbox("Choose ML Model", 
        ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"])

    with st.spinner("Training model..."):
        model = build_model(df_full, model_name)

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, 45)
        sex = st.selectbox("Sex", ["F", "M"])
        bp = st.selectbox("Blood Pressure", ["LOW", "NORMAL", "HIGH"])
    with col2:
        chol = st.selectbox("Cholesterol", ["NORMAL", "HIGH"])
        na = st.number_input("Sodium (Na)", 0.0, 2.0, 0.70)
        k = st.number_input("Potassium (K)", 0.0, 2.0, 0.05)

    if st.button("Predict Drug"):
        input_df = pd.DataFrame([[age, sex, bp, chol, na, k]],
                                columns=['Age', 'Sex', 'BP', 'Cholesterol', 'Na', 'K'])

        proba = model.predict_proba(input_df)[0]
        pred = model.predict(input_df)[0]

        top_idx = np.argsort(proba)[::-1]

        st.success(f"Recommended Drug: **{pred}** ({proba[top_idx[0]]*100:.2f}% confidence)")

        st.write("### Top 3 Predictions")
        for i in range(min(3, len(top_idx))):
            st.write(f"{i+1}. {model.classes_[top_idx[i]]}: {proba[top_idx[i]]*100:.2f}%")

        st.write("### Why this prediction?")
        st.info(
            f"• Age: {age}\n"
            f"• Sex: {sex}\n"
            f"• BP: {bp}\n"
            f"• Cholesterol: {chol}\n"
            f"• Na: {na}\n"
            f"• K: {k}"
        )

        if pred in drug_details:
            d = drug_details[pred]
            st.write("---")
            st.subheader(f"📌 About {pred}")
            st.write(f"**Use:** {d['use']}")
            st.write(f"**Mechanism:** {d['mechanism']}")
            st.write(f"**Side Effects:** {', '.join(d['side_effects'])}")
            st.write(f"**Precautions:** {d['precautions']}")
            st.write(f"**Dosage:** {d['dosage']}")

    st.markdown("</div>", unsafe_allow_html=True)


# ====================================================
# PAGE: DRUG INFORMATION
# ====================================================
if page == "Drug Information":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("💊 Drug Information — Full Details")

    for drug, info in drug_details.items():
        with st.expander(f"📌 {drug}"):
            st.write(f"**Use:** {info['use']}")
            st.write(f"**Mechanism:** {info['mechanism']}")
            st.write(f"**Side Effects:** {', '.join(info['side_effects'])}")
            st.write(f"**Precautions:** {info['precautions']}")
            st.write(f"**Dosage:** {info['dosage']}")

    st.markdown("</div>", unsafe_allow_html=True)


# ====================================================
# PAGE: ADMIN
# ====================================================
if page == "Admin":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("👤 User Management Panel")

    st.write("### Existing Users")
    for u in USERS:
        st.write("•", u)

    st.write("---")
    st.write("### Add User")
    new_user = st.text_input("New Username")
    new_pass = st.text_input("New Password", type="password")

    if st.button("Add User"):
        if new_user in USERS:
            st.error("User already exists!")
        else:
            USERS[new_user] = new_pass
            st.success("User added successfully!")

    st.write("---")
    st.write("### Remove User")
    remove_user = st.selectbox("Select User to Remove", list(USERS.keys()))

    if st.button("Delete User"):
        if remove_user == "admin":
            st.error("Cannot delete admin!")
        else:
            USERS.pop(remove_user)
            st.success("User removed successfully!")

    st.markdown("</div>", unsafe_allow_html=True)


# ====================================================
# PAGE: ABOUT
# ====================================================
if page == "About":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("ℹ️ About Smart Drug Shield")

    st.markdown("""
### **What is Smart Drug Shield?**
Smart Drug Shield is an intelligent drug recommendation system that uses **Machine Learning** to recommend the most suitable drug based on patient health features.

### **Features**
- ✔ Medical-themed UI (blue + green + white)
- ✔ Login system for multiple users
- ✔ Multiple ML Models (LogReg, KNN, DT, RF, SVM)
- ✔ Probability-based prediction with **top-3 drug recommendations**
- ✔ Detailed drug information page
- ✔ Admin panel to manage users

### **Dataset Columns**
- Age  
- Sex  
- BP  
- Cholesterol  
- Sodium (Na)  
- Potassium (K)  
- Drug (Label)

### **Important Note**
This system is created for:
- Academic projects  
- Machine learning demonstrations  
- Healthcare research simulations  

**⚠️ Not for real medical diagnosis.**
""")
    st.markdown("</div>", unsafe_allow_html=True)
