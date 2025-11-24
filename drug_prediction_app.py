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
            st.rerun()
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
    page = st.radio("📄 Navigate", ["Predictor", "Drug Information", "Bulk Prediction", "Monitoring", "Admin", "About"])
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()


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

# =====================================================
# Drug Information Page
# =====================================================
if page == "Drug Information":
    st.title("Drug Information")
    st.info("Detailed drug descriptions will appear here.")

# =====================================================
# Bulk Prediction Page
# =====================================================
if page == "Bulk Prediction":
    st.title("Bulk Prediction")
    st.info("Upload CSV for multiple predictions.")

# =====================================================
# Monitoring Page
# =====================================================
if page == "Monitoring":
    st.title("Prediction Monitoring")
    st.info("Graphical logs will appear here.")

# =====================================================
# Admin Page
# =====================================================
if page == "Admin":
    st.title("Admin Panel")
    st.info("Manage users and logs.")

# =====================================================
# About Page
# =====================================================
if page == "About":
    st.title("About")
    st.write("Drug Predictor App — with ML models, Dark Mode, and Login System.")
