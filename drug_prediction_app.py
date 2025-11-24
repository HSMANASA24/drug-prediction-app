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

# =====================================================
# Streamlit App Configuration
# =====================================================
st.set_page_config(page_title="Smart Drug Shield", page_icon="💊", layout="centered")

# ================= MAIN VISIBLE TITLE =================
st.markdown("""
    <h1 style='text-align:center; font-size:40px; font-weight:900; margin-top:-15px;'>
        🛡 Smart Drug Shield
    </h1>
""", unsafe_allow_html=True)

# =====================================================
# Password Hashing Utilities
# =====================================================
_SALT = "a9f5b3c7"

def hash_password(password: str) -> str:
    return hashlib.sha256((_SALT + password).encode()).hexdigest()

USERS = {"admin": hash_password("admin123")}

# =====================================================
# CSS THEMES
# =====================================================
base_css = """
<style>
body { background: #f4f7fb; }
section[data-testid="stSidebar"] { background:#ffffff !important; color:#000; }
.glass-panel { background:rgba(255,255,255,0.9); padding:20px; border-radius:12px; }
h1,h2,h3,h4 { color:#0b2545 !important; font-weight:900; }
</style>
"""

dark_css = """
<style>
body { background:#121212 !important; color:white !important; }
section[data-testid="stSidebar"] { background:#1e1e1e !important; color:white; }
.glass-panel { background:rgba(255,255,255,0.08); backdrop-filter:blur(6px); color:white !important; }
h1,h2,h3,h4 { color:#e5efff !important; }
</style>
"""

# =====================================================
# LOGIN SYSTEM
# =====================================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = None

def login_page():
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.subheader("🔒 Admin Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and secrets.compare_digest(USERS[username], hash_password(password)):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

    st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state["authenticated"]:
    st.markdown(base_css, unsafe_allow_html=True)
    login_page()
    st.stop()

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.header(f"Welcome, {st.session_state['username']}")
    theme_choice = st.radio("🌗 Theme Mode", ["Light Mode", "Dark Mode"])
    page = st.radio("📄 Navigate", ["Predictor", "Drug Information", "Admin", "About"])

    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

# APPLY THEME
st.markdown(base_css, unsafe_allow_html=True)
if theme_choice == "Dark Mode":
    st.markdown(dark_css, unsafe_allow_html=True)

# =====================================================
# LOAD SAMPLE DATA
# =====================================================
def load_sample():
    return pd.DataFrame([
        [23, 'F', 'HIGH', 'HIGH', 0.79, 0.03, 'Amlodipine-Atorvastatin'],
        [47, 'M', 'LOW', 'HIGH', 0.73, 0.05, 'Losartan'],
        [28, 'F', 'NORMAL', 'HIGH', 0.56, 0.07, 'Atorvastatin'],
        [61, 'F', 'LOW', 'HIGH', 0.55, 0.03, 'Amlodipine-Atorvastatin'],
        [45, 'M', 'NORMAL', 'NORMAL', 0.70, 0.05, 'Atenolol']
    ], columns=['Age', 'Sex', 'BP', 'Cholesterol', 'Na', 'K', 'Drug'])

# =====================================================
# MODEL TRAINING
# =====================================================
@st.cache_resource
def train_model(df, name):
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

    pipe = Pipeline([("pre", pre), ("clf", models[name])])
    pipe.fit(X, y)
    return pipe

# =====================================================
# PREDICTOR PAGE
# =====================================================
if page == "Predictor":
    df_train = load_sample()
    model_name = st.selectbox("Select Model", ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"])
    model = train_model(df_train, model_name)

    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.subheader("Single Prediction")

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
        input_df = pd.DataFrame([[age, sex, bp, chol, na, k]], 
                                columns=['Age','Sex','BP','Cholesterol','Na','K'])
        pred = model.predict(input_df)[0]
        confidence = max(model.predict_proba(input_df)[0]) * 100

        st.success(f"Predicted Drug: {pred} ({confidence:.2f}% confidence)")

        explanation = (
            f"The model predicted **{pred}** because:\n"
            f"- Age: {age}\n"
            f"- Sex: {sex}\n"
            f"- BP: {bp}\n"
            f"- Cholesterol: {chol}\n"
            f"- Sodium (Na): {na}\n"
            f"- Potassium (K): {k}"
        )

        st.info(explanation)

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# DRUG DETAILS
# =====================================================
drug_details = {
    "Atenolol": {
        "use": "Mild blood pressure control.",
        "mechanism": "Beta-blocker.",
        "side_effects": ["Fatigue", "Cold extremities", "Dizziness"],
        "precautions": "Avoid in asthma.",
        "dosage": "50 mg once daily."
    },
    "Losartan": {
        "use": "High BP treatment.",
        "mechanism": "ARB – relaxes blood vessels.",
        "side_effects": ["Low BP", "Fatigue"],
        "precautions": "Avoid in pregnancy.",
        "dosage": "25–50 mg/day."
    },
    "ORS-K": {
        "use": "Electrolyte imbalance correction.",
        "mechanism": "Restores Na/K balance.",
        "side_effects": ["Nausea", "Upset stomach"],
        "precautions": "Monitor electrolyte levels.",
        "dosage": "As needed."
    },
    "Atorvastatin": {
        "use": "High cholesterol.",
        "mechanism": "Reduces cholesterol synthesis.",
        "side_effects": ["Muscle pain", "Weakness"],
        "precautions": "Avoid high-fat diet.",
        "dosage": "10–20 mg/day."
    },
    "Amlodipine-Atorvastatin": {
        "use": "High BP + high cholesterol.",
        "mechanism": "Combined BP & cholesterol control.",
        "side_effects": ["Dizziness", "Fatigue"],
        "precautions": "Monitor BP regularly.",
        "dosage": "1 tablet daily."
    }
}

# =====================================================
# DRUG INFORMATION PAGE
# =====================================================
if page == "Drug Information":
    st.subheader("💊 Drug Information")
    for drug, info in drug_details.items():
        with st.expander(f"📌 {drug}"):
            st.write(f"**Use:** {info['use']}")
            st.write(f"**Mechanism:** {info['mechanism']}")
            st.write(f"**Side Effects:** {', '.join(info['side_effects'])}")
            st.write(f"**Precautions:** {info['precautions']}")
            st.write(f"**Dosage:** {info['dosage']}")

# =====================================================
# ADMIN PAGE
# =====================================================
if page == "Admin":
    st.subheader("👤 Admin Panel – User Management")

    st.write("**Current Users:**")
    for user in USERS:
        st.write("•", user)

    st.write("---")
    st.write("### Add User")
    new_user = st.text_input("Username", key="newuser")
    new_pass = st.text_input("Password", type="password", key="newpass")

    if st.button("Add User"):
        if new_user in USERS:
            st.error("User already exists.")
        else:
            USERS[new_user] = hash_password(new_pass)
            st.success("User added successfully.")

# =====================================================
# ABOUT PAGE
# =====================================================
if page == "About":
    st.subheader("ℹ️ About Smart Drug Shield")
    st.write("""
        Smart Drug Shield is an AI-powered system designed to predict the most suitable drug 
        based on patient clinical parameters.  
        
        ✔ Machine Learning–based  
        ✔ Multiple models supported  
        ✔ Secure admin login  
        ✔ Drug information database  
        ✔ Clean UI with light/dark themes  
    """)
