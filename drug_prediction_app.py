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

# =====================================================
# Streamlit App Configuration
# =====================================================
st.set_page_config(page_title="Drug Prescription Classifier", page_icon="💊", layout="centered")

# =====================================================
# Password Hashing Utilities
# =====================================================
_SALT = "a9f5b3c7"  # Change for production

def hash_password(password: str) -> str:
    return hashlib.sha256((_SALT + password).encode()).hexdigest()

USERS = {"admin": hash_password("admin123")}

# =====================================================
# CSS Themes
# =====================================================
base_css = """
<style>
body { background: linear-gradient(135deg, #1f3a93, #6e2ba8, #f54591); background-attachment: fixed; }
section[data-testid="stSidebar"] { background: #fff !important; color:#000 !important; }
.glass-panel { backdrop-filter: blur(10px); background: rgba(255,255,255,0.65); border-radius:16px; padding:18px; margin-bottom:20px; color:#000 !important; }
html, body, div, p, span, label { color:#000 !important; }
h1, h2, h3, h4 { color:#000 !important; font-weight:800; }
input, select { background: rgba(255,255,255,0.9)!important; color:#000!important; border-radius:10px!important; }
.stButton>button { background:white!important; color:#000!important; border-radius:10px!important; font-weight:600!important; }
</style>
"""

dark_css = """
<style>
body { background:#1c1c1c!important; color:white!important; }
section[data-testid="stSidebar"] { background:#fff!important; color:#000!important; }
.glass-panel { backdrop-filter: blur(12px); background: rgba(40,40,40,0.55)!important; border-radius:16px; padding:20px; color:white!important; }
input, select { background: rgba(255,255,255,0.15)!important; color:white!important; border-radius:10px!important; }
.stButton>button { background: rgba(255,255,255,0.2)!important; color:white!important; }
</style>
"""

# =====================================================
# Authentication
# =====================================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = None

def login_user(username, password):
    return username in USERS and secrets.compare_digest(USERS[username], hash_password(password))

def require_login():
    st.markdown('<div class="glass-panel" style="max-width:600px; margin:auto;">', unsafe_allow_html=True)
    st.title("🔒 Admin Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    col1, col2 = st.columns([1, 1])
    with col1:
        login_btn = st.button("Login")
    with col2:
        forgot_btn = st.button("Forgot Password")

    if login_btn:
        if login_user(user, pwd):
            st.session_state["authenticated"] = True
            st.session_state["username"] = user
            st.experimental_rerun()
        else:
            st.error("❌ Invalid username or password")
    if forgot_btn:
        st.info("Reset password manually in the USERS dictionary.")

    st.markdown('</div>', unsafe_allow_html=True)

# BLOCK APP UNTIL LOGIN
if not st.session_state["authenticated"]:
    st.markdown(base_css, unsafe_allow_html=True)
    require_login()
    st.stop()

# =====================================================
# Sidebar
# =====================================================
with st.sidebar:
    st.header(f"Welcome, {st.session_state['username']}")
    theme_choice = st.radio("🌗 Theme Mode", ["Light Mode", "Dark Mode"])
    page = st.radio("📄 Navigate", ["Predictor", "Drug Information", "Admin", "About"])
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.experimental_rerun()

# Apply theme
st.markdown(base_css, unsafe_allow_html=True)
if theme_choice == "Dark Mode":
    st.markdown(dark_css, unsafe_allow_html=True)

# =====================================================
# Machine Learning Helpers
# =====================================================
def load_sample_df():
    return pd.DataFrame([
        [23, 'F', 'HIGH', 'HIGH', 0.79, 0.03, 'drugY'],
        [47, 'M', 'LOW', 'HIGH', 0.73, 0.05, 'drugC'],
        [28, 'F', 'NORMAL', 'HIGH', 0.56, 0.07, 'drugX'],
        [61, 'F', 'LOW', 'HIGH', 0.55, 0.03, 'drugY'],
        [45, 'M', 'NORMAL', 'NORMAL', 0.70, 0.05, 'drugA'],
    ], columns=['Age','Sex','BP','Cholesterol','Na','K','Drug'])

@st.cache_resource
def train_model(df, model_name):
    X = df[['Age','Sex','BP','Cholesterol','Na','K']]
    y = df['Drug']

    pre = ColumnTransformer([
        ("num", StandardScaler(), ['Age','Na','K']),
        ("cat", OneHotEncoder(sparse_output=False), ['Sex','BP','Cholesterol'])
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True)
    }

    pipe = Pipeline([("pre", pre), ("clf", models[model_name])])
    pipe.fit(X, y)
    return pipe

# =====================================================
# Predictor Page
# =====================================================
if page == "Predictor":
    st.markdown('<div class="glass-panel"><h2>Single Prediction</h2></div>', unsafe_allow_html=True)

    df_train = load_sample_df()
    model_name = st.selectbox("Select Model", ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"])

    model = train_model(df_train, model_name)

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, 45)
        sex = st.selectbox("Sex", ["F", "M"])
        bp = st.selectbox("Blood Pressure", ["LOW", "NORMAL", "HIGH"])
    with col2:
        chol = st.selectbox("Cholesterol", ["HIGH", "NORMAL"])
        na = st.number_input("Sodium (Na)", format="%.3f", value=0.70)
        k = st.number_input("Potassium (K)", format="%.3f", value=0.05)

    if st.button("Predict"):
        input_df = pd.DataFrame([[age, sex, bp, chol, na, k]], columns=['Age','Sex','BP','Cholesterol','Na','K'])
        pred = model.predict(input_df)[0]
        st.success(f"Predicted Drug: {pred}")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Drug details
# ---------------------------
drug_details = {
    "drugA": {
        "name": "Drug A",
        "use": "Used for normal BP and normal cholesterol.",
        "mechanism": "Supports circulatory health.",
        "side_effects": ["Headache", "Dry mouth", "Dizziness"],
        "precautions": "Avoid alcohol; stay hydrated.",
        "dosage": "1 tablet daily."
    },
    "drugB": {
        "name": "Drug B",
        "use": "For high blood pressure.",
        "mechanism": "Relaxes blood vessels.",
        "side_effects": ["Low BP", "Fatigue"],
        "precautions": "Not suitable for pregnancy.",
        "dosage": "1 tablet/day."
    },
    "drugC": {
        "name": "Drug C",
        "use": "For electrolyte imbalance.",
        "mechanism": "Balances Na/K levels.",
        "side_effects": ["Nausea", "Stomach upset"],
        "precautions": "Monitor Na/K levels regularly.",
        "dosage": "1–2 tablets/day."
    },
    "drugX": {
        "name": "Drug X",
        "use": "Normal BP + high cholesterol.",
        "mechanism": "Reduces cholesterol synthesis.",
        "side_effects": ["Muscle pain", "Weakness"],
        "precautions": "Avoid high-fat diet.",
        "dosage": "Take in the evening."
    },
    "drugY": {
        "name": "Drug Y",
        "use": "High BP + high cholesterol.",
        "mechanism": "Lowers cholesterol synthesis and reduces BP.",
        "side_effects": ["Dizziness", "Muscle fatigue"],
        "precautions": "Check BP regularly.",
        "dosage": "1 tablet/day."
    }
}

# =====================================================
# Drug Information Page
# =====================================================
if page == "Drug Information":
    st.title("💊 Drug Information")
    st.write("Browse details of all available drugs.")

    for key, info in drug_details.items():
        with st.expander(f"📌 {info['name']}"):
            st.markdown(f"**Use:** {info['use']}")
            st.markdown(f"**Mechanism:** {info['mechanism']}")
            st.markdown(f"**Side Effects:** {', '.join(info['side_effects'])}")
            st.markdown(f"**Precautions:** {info['precautions']}")
            st.markdown(f"**Dosage:** {info['dosage']}")

# =====================================================
# (Bulk Prediction Removed)
# =====================================================

if page == "Bulk Prediction":
    st.title("Bulk Prediction")
    st.info("Upload CSV for multiple predictions.")

# =====================================================
# (Monitoring Removed)
# =====================================================

if page == "Monitoring":
    st.title("Prediction Monitoring")
    st.info("Graphical logs will appear here.")

# =====================================================
# Admin Page
# =====================================================
if page == "Admin":
    st.title("👤 User Management — Admin Panel")
    st.write("Add or remove application users.")

    # Display current users
    st.subheader("Current Users")
    for u in USERS.keys():
        st.markdown(f"• **{u}**")

    st.markdown("---")

    st.subheader("Add New User")
    new_user = st.text_input("Username", key="add_user_name")
    new_pass = st.text_input("Password", type="password", key="add_user_pass")

    if st.button("Add User", key="add_user_btn"):
        if new_user in USERS:
            st.error("User already exists!")
        else:
            USERS[new_user] = hash_password(new_pass)
            st.success(f"User '{new_user}' added successfully!")
            st.rerun()

    st.markdown("---")

    st.subheader("Remove User")
    user_to_delete = st.selectbox("Select User to Remove", list(USERS.keys()), key="remove_user_select")

    if st.button("Remove User", key="remove_user_btn"):
        if user_to_delete == "admin":
            st.error("Cannot remove the main admin user.")
        else:
            USERS.pop(user_to_delete, None)
            st.success(f"User '{user_to_delete}' removed successfully!")
            st.rerun()

# =====================================================
# About Page
# =====================================================
if page == "About":
    st.title("ℹ️ About This Application")
    st.markdown(
        """
        ## 💊 Drug Prescription Classifier
        This application is an intelligent drug prediction system built using **Machine Learning**.

        ### 🔍 What it does
        - Predicts the most suitable drug based on patient health indicators
        - Uses multiple ML models (Logistic Regression, KNN, Decision Tree, Random Forest, SVM)
        - Provides detailed drug information
        - Includes secure login with an Admin Panel
        - Allows administrators to manage user accounts

        ### 🧠 How predictions work
        The model analyzes:
        - **Age**
        - **Sex**
        - **Blood Pressure (BP)**
        - **Cholesterol Level**
        - **Sodium (Na)** and **Potassium (K)** levels

        Based on these features, the algorithm recommends one of the available drugs.

        ### 🎨 Application Features
        - Modern **glassmorphism UI**
        - **Light/Dark mode themes**
        - Clean and responsive layout
        - Secure password hashing system

        ### 👨‍⚕️ Intended Use
        This tool is designed for:
        - Students learning machine learning
        - Healthcare projects
        - Demonstration of predictive modeling
        - Academic research

        ### 🛠 Developer Notes
        - Built with **Python + Streamlit**
        - Machine learning powered by **scikit-learn**
        - Styling enhanced using **HTML/CSS** inside Streamlit

        ---
        **Made with ❤️ for educational and demonstration purposes.**
        """
    )
