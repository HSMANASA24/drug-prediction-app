# app.py (fixed full file)
import streamlit as st
import pandas as pd
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

from lightgbm import LGBMClassifier

# =====================================================
# Streamlit App Configuration
# =====================================================
st.set_page_config(page_title="🛡️ Smart Drug Shield", page_icon="💊", layout="centered")

# =====================================================
# Simple secure password hashing (for demo)
# =====================================================
_SALT = "a9f5b3c7"  # change for production

def hash_password(password: str) -> str:
    return hashlib.sha256((_SALT + password).encode()).hexdigest()

# Default users dictionary (passwords stored hashed)
USERS = {"admin": hash_password("admin123")}

# =====================================================
# CSS Themes (light default)
# =====================================================
base_css = """
<style>
/* Layout + glass panel */
body { background: linear-gradient(135deg, #f4f7fb, #e8eefc); }
section[data-testid="stSidebar"] { background: #ffffff !important; color:#000 !important; }
.glass-panel { backdrop-filter: blur(8px); background: rgba(255,255,255,0.85); border-radius:12px; padding:16px; margin-bottom:16px; color:#000 !important; }
h1,h2,h3,h4 { color:#0b2545 !important; font-weight:700; }
input, select, textarea { background: rgba(255,255,255,0.98)!important; color:#000!important; border-radius:8px!important; }
.stButton>button { background:#0b63b5!important; color:#fff!important; border-radius:8px!important; font-weight:600!important; }
</style>
"""

dark_css = """
<style>
body { background:#0f1724!important; color:#eef2ff!important; }
section[data-testid="stSidebar"] { background:#0b1220!important; color:#fff!important; }
.glass-panel { backdrop-filter: blur(6px); background: rgba(20,20,25,0.6)!important; border-radius:12px; padding:16px; margin-bottom:16px; color:#fff!important; }
h1,h2,h3,h4 { color:#a8d0ff !important; font-weight:700; }
input, select, textarea { background: rgba(255,255,255,0.03)!important; color:#fff!important; border-radius:8px!important; }
.stButton>button { background:#1f6feb!important; color:#fff!important; }
</style>
"""

# =====================================================
# Authentication utilities
# =====================================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = None

def login_user(username, password):
    """Check username and password against USERS dict (hashed)."""
    if not username or not password:
        return False
    return username in USERS and secrets.compare_digest(USERS[username], hash_password(password))

def require_login():
    """Show login box and block app until authenticated."""
    st.markdown('<div class="glass-panel" style="max-width:680px; margin:auto;">', unsafe_allow_html=True)
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
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

    if forgot_btn:
        st.info("If you forgot the admin password, update USERS in the code or add a new user in Admin panel.")

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# BLOCK until authenticated
if not st.session_state["authenticated"]:
    st.markdown(base_css, unsafe_allow_html=True)
    require_login()

# =====================================================
# Sidebar (after login)
# =====================================================
with st.sidebar:
    st.header(f"Welcome, {st.session_state.get('username', 'admin')}")
    theme_choice = st.radio("🌗 Theme Mode", ["Light Mode", "Dark Mode"], index=0, key="theme_choice")
    page = st.radio("📄 Navigate", ["Predictor", "Drug Information", "Admin", "About"], index=0, key="nav_main")
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.rerun()

# Apply chosen theme
st.markdown(base_css, unsafe_allow_html=True)
if theme_choice == "Dark Mode":
    st.markdown(dark_css, unsafe_allow_html=True)

# =====================================================
# Machine Learning helpers
# =====================================================
def load_sample_df():
    """Return a small sample dataset with updated drug names."""
    return pd.DataFrame([
        [23, 'F', 'HIGH', 'HIGH', 0.79, 0.03, 'Amlodipine-Atorvastatin'],
        [47, 'M', 'LOW', 'HIGH', 0.73, 0.05, 'Losartan'],
        [28, 'F', 'NORMAL', 'HIGH', 0.56, 0.07, 'Atorvastatin'],
        [61, 'F', 'LOW', 'HIGH', 0.55, 0.03, 'Amlodipine-Atorvastatin'],
        [45, 'M', 'NORMAL', 'NORMAL', 0.70, 0.05, 'Atenolol']
    ], columns=['Age','Sex','BP','Cholesterol','Na','K','Drug'])

@st.cache_resource
def train_model(df, model_name):
    """Train and return a pipeline for the requested model."""
    X = df[['Age','Sex','BP','Cholesterol','Na','K']]
    y = df['Drug']

    pre = ColumnTransformer([
        ("num", StandardScaler(), ['Age','Na','K']),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['Sex','BP','Cholesterol'])
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
       
        "LightGBM": LGBMClassifier()
    }

    if model_name not in models:
        raise ValueError("Unknown model: " + str(model_name))

    pipe = Pipeline([("pre", pre), ("clf", models[model_name])])
    pipe.fit(X, y)
    return pipe

# =====================================================
# Predictor Page
# =====================================================
if page == "Predictor":
    st.markdown('<div class="glass-panel"><h2>Single Prediction</h2></div>', unsafe_allow_html=True)

    df_train = load_sample_df()
    model_name = st.selectbox("Select Model", ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM", "LightGBM"])

    with st.spinner("Training model..."):
        model = train_model(df_train, model_name)

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", ["F", "M"])
        bp = st.selectbox("Blood Pressure", ["LOW", "NORMAL", "HIGH"])
    with col2:
        chol = st.selectbox("Cholesterol", ["HIGH", "NORMAL"])
        na = st.number_input("Sodium (Na)", format="%.3f", value=0.70)
        k = st.number_input("Potassium (K)", format="%.3f", value=0.05)

    if st.button("Predict"):
        input_df = pd.DataFrame([[age, sex, bp, chol, na, k]],
                                columns=['Age','Sex','BP','Cholesterol','Na','K'])

        # some classifiers support predict_proba; we assume our chosen ones do
        try:
            proba = model.predict_proba(input_df)[0]
        except Exception:
            proba = None

        try:
            pred = model.predict(input_df)[0]
        except Exception as e:
            st.error("Prediction failed: " + str(e))
            pred = None

        if pred is not None:
            if proba is not None:
                confidence = max(proba) * 100
                st.success("Predicted Drug: " + str(pred) + " (" + str(round(confidence, 2)) + "% confidence)")
            else:
                st.success("Predicted Drug: " + str(pred))

            # Safe explanation string (no unterminated f-strings)
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
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# Drug details (clean, single dictionary)
# =====================================================
drug_details = {
    "Atenolol": {
        "name": "Atenolol",
        "use": "Used for mild blood pressure control.",
        "mechanism": "Beta-blocker that reduces heart rate and BP.",
        "side_effects": ["Fatigue", "Cold extremities", "Dizziness"],
        "precautions": "Not recommended for asthma patients.",
        "dosage": "50 mg once daily."
    },

    "Losartan": {
        "name": "Losartan",
        "use": "Used for high blood pressure.",
        "mechanism": "ARB that relaxes blood vessels.",
        "side_effects": ["Low BP", "Increased potassium", "Fatigue"],
        "precautions": "Avoid in pregnancy.",
        "dosage": "25–50 mg per day."
    },

    "ORS-K": {
        "name": "ORS-K",
        "use": "Corrects sodium–potassium imbalance.",
        "mechanism": "Replenishes electrolytes and restores hydration.",
        "side_effects": ["Nausea", "Stomach upset"],
        "precautions": "Monitor Na/K levels.",
        "dosage": "As required during dehydration or imbalance."
    },

    "Atorvastatin": {
        "name": "Atorvastatin",
        "use": "Used for high cholesterol.",
        "mechanism": "Reduces cholesterol synthesis in the liver.",
        "side_effects": ["Muscle pain", "Weakness", "Liver enzyme changes"],
        "precautions": "Avoid high-fat diet; monitor liver function.",
        "dosage": "10–20 mg in the evening."
    },

    "Amlodipine-Atorvastatin": {
        "name": "Amlodipine-Atorvastatin",
        "use": "Used for high BP and high cholesterol together.",
        "mechanism": "Combines BP-lowering and cholesterol-lowering action.",
        "side_effects": ["Muscle fatigue", "Dizziness", "Edema"],
        "precautions": "Regular BP and cholesterol monitoring.",
        "dosage": "1 tablet daily."
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
            st.markdown("**Use:** " + info["use"])
            st.markdown("**Mechanism:** " + info["mechanism"])
            st.markdown("**Side Effects:** " + ", ".join(info["side_effects"]))
            st.markdown("**Precautions:** " + info["precautions"])
            st.markdown("**Dosage:** " + info["dosage"])

# =====================================================
# Admin Page - User management
# =====================================================
if page == "Admin":
    st.title("👤 User Management — Admin Panel")
    st.write("Add or remove application users.")

    # Display current users
    st.subheader("Current Users")
    for u in USERS.keys():
        st.markdown("• **" + u + "**")

    st.markdown("---")

    st.subheader("Add New User")
    new_user = st.text_input("Username", key="add_user_name")
    new_pass = st.text_input("Password", type="password", key="add_user_pass")

    if st.button("Add User", key="add_user_btn"):
        if not new_user or not new_pass:
            st.error("Please enter both username and password.")
        elif new_user in USERS:
            st.error("User already exists!")
        else:
            USERS[new_user] = hash_password(new_pass)
            st.success("User '" + new_user + "' added successfully!")
            st.rerun()

    st.markdown("---")

    st.subheader("Remove User")
    user_list = list(USERS.keys())
    user_to_delete = st.selectbox("Select User to Remove", user_list, key="remove_user_select")

    if st.button("Remove User", key="remove_user_btn"):
        if user_to_delete == "admin":
            st.error("Cannot remove the main admin user.")
        else:
            USERS.pop(user_to_delete, None)
            st.success("User '" + user_to_delete + "' removed successfully!")
            st.rerun()

# =====================================================
# About Page
# =====================================================
if page == "About":
    st.title("ℹ️ About This Application")
    st.markdown(
        """
        ## 🛡️ Smart Drug Shield — Drug Prescription Classifier

        This app demonstrates a small end-to-end ML pipeline to predict a drug based on Age, Sex, BP, Cholesterol, Sodium (Na), and Potassium (K).

        **Features**
        - Multiple ML algorithms (Logistic, KNN, Decision Tree, RandomForest, SVM, XGBoost, LightGBM)
        - Single-prediction UI with confidence score and plain-language explanation
        - Secure Admin login and user management
        - Drug information page with clinical-style descriptions
        - Light/Dark theme and a clean glass-style UI

        **Intended use**
        Educational/demo only — not for clinical decisions. Always validate with clinical expertise.

        **Developer notes**
        - Python + Streamlit + scikit-learn; XGBoost and LightGBM are used if installed.
        """
    )
