# app.py - Smart Drug Shield (updated to use your /mnt/data/Drug.csv and Option B drug names)

import streamlit as st
import pandas as pd
import hashlib
import secrets
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Try optional imports (XGBoost / LightGBM). If not installed, we skip them gracefully.
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

# =====================================================
# App config & visible title
# =====================================================
st.set_page_config(page_title="🛡 Smart Drug Shield", page_icon="💊", layout="centered")

st.markdown("""
    <h1 style='text-align:center; font-size:40px; font-weight:900; margin-top:-10px;'>
        🛡 Smart Drug Shield
    </h1>
""", unsafe_allow_html=True)

# =====================================================
# Simple password hashing & default admin user
# =====================================================
_SALT = "a9f5b3c7"
def hash_password(password: str) -> str:
    return hashlib.sha256((_SALT + password).encode()).hexdigest()

USERS = {"admin": hash_password("admin123")}  # default; change in production

# =====================================================
# CSS for light/dark themes (keeps title visible)
# =====================================================
BASE_CSS = """
<style>
/* Basic glass style */
.glass-panel { backdrop-filter: blur(6px); background: rgba(255,255,255,0.9); border-radius:12px; padding:14px; margin-bottom:16px; }
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

# =====================================================
# Authentication utilities
# =====================================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = None

def login_user(username, password):
    if not username or not password:
        return False
    return username in USERS and secrets.compare_digest(USERS[username], hash_password(password))

def show_login():
    st.markdown('<div class="glass-panel" style="max-width:700px; margin:auto;">', unsafe_allow_html=True)
    st.subheader("🔒 Admin Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login_user(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid username or password")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

if not st.session_state["authenticated"]:
    show_login()

# =====================================================
# Sidebar navigation and theme selection
# =====================================================
with st.sidebar:
    st.header(f"Welcome, {st.session_state.get('username','admin')}")
    theme_choice = st.radio("🌗 Theme Mode", ["Light Mode", "Dark Mode"], index=0)
    page = st.radio("📄 Navigate", ["Predictor", "Drug Information", "Admin", "About"])
    st.markdown("---")
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.rerun()

# Apply dark CSS if selected
if theme_choice == "Dark Mode":
    st.markdown(DARK_CSS, unsafe_allow_html=True)

# =====================================================
# Data loading and mapping (using your uploaded dataset)
# =====================================================
DATA_PATH = "/mnt/data/Drug.csv"   # <-- your dataset path (from your upload)

@st.cache_data
def load_and_prepare(path=DATA_PATH):
    df = pd.read_csv(path)
    # Clean column names if necessary
    df.columns = [c.strip() for c in df.columns]
    # Map dataset drug labels to real drug names (Option B)
    mapping = {
        "drugA": "Amlodipine",
        "drugB": "Atenolol",
        "drugC": "ORS-K",
        "drugX": "Atorvastatin",
        "drugY": "Losartan"
    }
    # If some rows have already real names, map will ignore them (we use .map then fillna)
    df["Drug"] = df["Drug"].map(mapping).fillna(df["Drug"])
    return df

# Load dataset
df_full = load_and_prepare()

# Show small dataset info
st.sidebar.markdown(f"**Dataset:** {DATA_PATH}")
st.sidebar.markdown(f"Rows: {df_full.shape[0]}, Columns: {df_full.shape[1]}")
st.sidebar.markdown("---")

# =====================================================
# Drug info dictionary (real drug entries)
# =====================================================
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
        "dosage": "As per dehydration/electrolyte protocol."
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

# =====================================================
# ML pipeline & training using the real dataset
# =====================================================
@st.cache_resource
def build_and_train(model_name, df):
    # Drop NA rows (if any) and ensure types
    df = df.dropna().copy()
    X = df[['Age','Sex','BP','Cholesterol','Na','K']]
    y = df['Drug']

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), ['Age','Na','K']),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['Sex','BP','Cholesterol'])
    ])

    # Build models dict dynamically depending on available packages
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
        raise ValueError("Model not available: " + str(model_name))

    pipe = Pipeline([("pre", preprocessor), ("clf", models[model_name])])
    pipe.fit(X, y)
    return pipe

# =====================================================
# Predictor Page: Train on full dataset and predict
# =====================================================
if page == "Predictor":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("Single Prediction")

    # Model selection list (only include XGBoost/LightGBM if installed)
    available_models = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"]
    if HAS_XGB:
        available_models.append("XGBoost")
    if HAS_LGB:
        available_models.append("LightGBM")

    model_choice = st.selectbox("Select Model", available_models, index=0)

    # Train model (cached)
    with st.spinner("Training model on full dataset..."):
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

        # Predict
        try:
            pred = model.predict(input_df)[0]
        except Exception as e:
            st.error("Prediction error: " + str(e))
            pred = None

        # Get probabilities if available
        proba = None
        try:
            prob_array = model.predict_proba(input_df)[0]
            proba = prob_array
        except Exception:
            proba = None

        # Show results
        if pred is not None:
            if proba is not None:
                # Top prediction and confidence
                top_idx = int(np.argmax(proba))
                top_label = model.classes_[top_idx]
                confidence = float(proba[top_idx]) * 100
                st.success("Predicted Drug: " + str(top_label) + f"  ({confidence:.2f}% confidence)")

                # Top 3 predictions
                sorted_idx = np.argsort(proba)[::-1]
                st.write("Top predictions:")
                for i in range(min(3, len(sorted_idx))):
                    idx = sorted_idx[i]
                    label = model.classes_[idx]
                    prob_pct = proba[idx] * 100
                    st.write(f"{i+1}. {label} — {prob_pct:.2f}%")
                # Confidence indicator (color block)
                if confidence >= 80:
                    st.info("Confidence: High ✅")
                elif confidence >= 60:
                    st.info("Confidence: Moderate ⚠️")
                elif confidence >= 40:
                    st.warning("Confidence: Low ⚠️ Consider review")
                else:
                    st.error("Confidence: Very Low ❌ Seek further checks")
            else:
                st.success("Predicted Drug: " + str(pred))
                st.info("Model does not provide probabilities for this algorithm.")

            # Plain-language explanation (safe string build)
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

            # Show drug details if available
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

# =====================================================
# Drug Information page (shows details for mapped drugs)
# =====================================================
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

# =====================================================
# Admin panel (user management)
# =====================================================
if page == "Admin":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("👤 Admin — User Management")

    st.write("**Current Users:**")
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
            st.success("User added.")
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

# =====================================================
# About page
# =====================================================
if page == "About":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("ℹ️ About Smart Drug Shield")
    st.markdown("""
    Smart Drug Shield is an educational demo application that trains classification models to predict which
    drug (mapped to medical names) may be appropriate given patient features (Age, Sex, BP, Cholesterol, Na, K).
    
    **Notes**
    - The dataset used is: `/mnt/data/Drug.csv` (200 rows).
    - Drug labels have been mapped to medical names for clarity (Option B).
    - This is for demonstration / educational purposes — NOT a clinical decision tool.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
