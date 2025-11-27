# app.py - Smart Drug Shield (Ensemble predictor + Chatbot using ONLY Random Forest)
import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ---------------------------
# App config & header
# ---------------------------
st.set_page_config(page_title="🛡 Smart Drug Shield", page_icon="💊", layout="centered")
st.markdown("""
    <h1 style='text-align:center; font-size:40px; font-weight:900; color:#0A3D62; margin-bottom:0.1rem'>
        🛡 Smart Drug Shield
    </h1>
    <p style='text-align:center; color:#145A32; margin-top:0; margin-bottom:1rem'>
        AI-powered drug prescription classifier — Medical theme
    </p>
""", unsafe_allow_html=True)

# ---------------------------
# Simple medical theme CSS
# ---------------------------
MEDICAL_CSS = """
<style>
body { background: linear-gradient(135deg,#e8f9ff,#d4fce5); }
.glass-panel { background: rgba(255,255,255,0.92); backdrop-filter: blur(6px); border-radius:12px; padding:16px; margin-bottom:16px; border:1px solid #bfe9ff; }
h1,h2,h3,h4 { color:#0A3D62 !important; font-weight:800; }
label, p, span, div { color:#0A3D62 !important; }
.stButton>button { background-color:#0A3D62 !important; color:white !important; border-radius:8px !important; padding:8px 14px !important; }
.stButton>button:hover { background-color:#145A32 !important; }
</style>
"""
st.markdown(MEDICAL_CSS, unsafe_allow_html=True)

# ---------------------------
# In-memory users (plain text as requested)
# ---------------------------
USERS = {
    "admin": "Admin@123",
    "manasa": "Manasa@2005",
    "doctor": "Doctor@123",
    "student": "Student@123"
}

# ---------------------------
# Session initialization
# ---------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "predictor_inputs" not in st.session_state:
    st.session_state["predictor_inputs"] = None
if "ensemble_models" not in st.session_state:
    st.session_state["ensemble_models"] = None
if "rf_model" not in st.session_state:
    st.session_state["rf_model"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ---------------------------
# Login page
# ---------------------------
def login_page():
    st.markdown('<div class="glass-panel" style="max-width:720px; margin:auto;">', unsafe_allow_html=True)
    st.subheader("🔒 Smart Drug Shield — Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    col1, col2 = st.columns([1,1])
    with col1:
        login_btn = st.button("Login")
    with col2:
        clear_btn = st.button("Clear")
    if clear_btn:
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.experimental_rerun()
    if login_btn:
        if user in USERS and USERS[user] == pwd:
            st.session_state["authenticated"] = True
            st.session_state["username"] = user
            st.success("Login successful 🎉")
        else:
            st.error("❌ Invalid username or password")
    st.markdown('</div>', unsafe_allow_html=True)

if not st.session_state["authenticated"]:
    login_page()
    st.stop()

# ---------------------------
# Sidebar navigation
# ---------------------------
with st.sidebar:
    st.header(f"Welcome, {st.session_state['username']}")
    page = st.radio("📄 Navigate", ["Predictor", "Drug Information", "Admin", "About"])
    st.markdown("---")
    st.markdown("Dataset: local `/mnt/data/Drug.csv` or GitHub RAW fallback.")
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.experimental_rerun()

# ---------------------------
# Load dataset (local or GitHub raw)
# ---------------------------
LOCAL_PATH = "/mnt/data/Drug.csv"
GITHUB_RAW = "https://raw.githubusercontent.com/HSMANASA24/drug-prediction-app/c476f30acf26ddc14b6b4a7eb796786c23a23edd/Drug.csv"

@st.cache_data
def load_dataset():
    if os.path.exists(LOCAL_PATH):
        df = pd.read_csv(LOCAL_PATH)
    else:
        df = pd.read_csv(GITHUB_RAW)
    df.columns = [c.strip() for c in df.columns]
    mapping = {
        "drugA":"Amlodipine",
        "drugB":"Atenolol",
        "drugC":"ORS-K",
        "drugX":"Atorvastatin",
        "drugY":"Losartan"
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

# ---------------------------
# Drug information dictionary
# ---------------------------
drug_details = {
    "Amlodipine": {"use":"Lowers BP by relaxing vessels.","mechanism":"Calcium channel blocker.","side_effects":["Dizziness","Edema","Headache"],"precautions":"Monitor BP.","dosage":"5–10 mg daily"},
    "Atenolol": {"use":"Controls BP and heart rate.","mechanism":"Beta-blocker.","side_effects":["Fatigue","Bradycardia"],"precautions":"Avoid in asthma.","dosage":"50 mg daily"},
    "ORS-K": {"use":"Electrolyte replacement (Na/K).","mechanism":"Replenishes Na+ & K+.","side_effects":["Nausea"],"precautions":"Monitor electrolytes.","dosage":"As required"},
    "Atorvastatin": {"use":"Lowers cholesterol.","mechanism":"Statin.","side_effects":["Muscle pain","Liver changes"],"precautions":"Check liver enzymes.","dosage":"10–20 mg evening"},
    "Losartan": {"use":"Treats hypertension.","mechanism":"ARB.","side_effects":["Dizziness","High potassium"],"precautions":"Avoid pregnancy.","dosage":"25–50 mg daily"}
}

# ---------------------------
# OneHot encoder compatibility helper
# ---------------------------
def onehot_factory():
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown='ignore')

# ---------------------------
# Train ensemble (multiple models) - cached
# ---------------------------
@st.cache_resource
def train_ensemble_models(df):
    df = df.dropna().copy()
    X = df[['Age','Sex','BP','Cholesterol','Na','K']]
    y = df['Drug']
    pre = ColumnTransformer([("num", StandardScaler(), ['Age','Na','K']), ("cat", onehot_factory(), ['Sex','BP','Cholesterol'])])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True)
    }
    trained = []
    for name, m in models.items():
        pipe = Pipeline([("pre", pre), ("clf", m)])
        pipe.fit(X, y)
        trained.append((name, pipe))
    return trained

# ---------------------------
# Train single Random Forest for chatbot (cached)
# ---------------------------
@st.cache_resource
def train_chatbot_rf(df):
    df = df.dropna().copy()
    X = df[['Age','Sex','BP','Cholesterol','Na','K']]
    y = df['Drug']
    pre = ColumnTransformer([("num", StandardScaler(), ['Age','Na','K']), ("cat", onehot_factory(), ['Sex','BP','Cholesterol'])])
    rf = Pipeline([("pre", pre), ("clf", RandomForestClassifier())])
    rf.fit(X, y)
    return rf

# ensure ensemble & rf are trained and stored in session
if st.session_state["ensemble_models"] is None:
    with st.spinner("Training ensemble models..."):
        st.session_state["ensemble_models"] = train_ensemble_models(df_full)

if st.session_state["rf_model"] is None:
    with st.spinner("Training chatbot Random Forest model..."):
        st.session_state["rf_model"] = train_chatbot_rf(df_full)

# ---------------------------
# Ensemble prediction helper (majority vote + aggregated probs)
# ---------------------------
def ensemble_predict(model_pipes, input_df):
    preds = []
    prob_list = []
    for name, pipe in model_pipes:
        try:
            p = pipe.predict(input_df)[0]
            preds.append(p)
        except Exception:
            continue
        try:
            proba = pipe.predict_proba(input_df)[0]
            prob_list.append((pipe.classes_, proba))
        except Exception:
            pass
    if len(preds) == 0:
        return None, None, []
    final_pred = max(set(preds), key=preds.count)
    # average prob for final_pred
    probs_for_final = []
    for classes, proba in prob_list:
        if final_pred in classes:
            idx = list(classes).index(final_pred)
            probs_for_final.append(proba[idx])
    confidence = float(np.mean(probs_for_final))*100.0 if len(probs_for_final)>0 else None
    # top3 aggregated
    agg = {}
    for classes, proba in prob_list:
        for cls, p in zip(classes, proba):
            agg.setdefault(cls, []).append(p)
    avg_probs = {cls: float(np.mean(ps)) for cls, ps in agg.items()}
    sorted_labels = sorted(avg_probs.items(), key=lambda x: x[1], reverse=True)
    top3 = sorted_labels[:3]
    return final_pred, confidence, top3

# ---------------------------
# Predictor page with split layout (left predictor, right chatbot)
# ---------------------------
if page == "Predictor":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("🔍 Predictor (Ensemble) — Chatbot uses Random Forest")

    left, right = st.columns([2,1])

    # LEFT: Predictor
    with left:
        st.write("### Enter patient details")
        age = st.number_input("Age", 1, 120, 45, key="age")
        sex = st.selectbox("Sex", ["F","M"], key="sex")
        bp = st.selectbox("Blood Pressure (BP)", ["LOW","NORMAL","HIGH"], key="bp")
        chol = st.selectbox("Cholesterol", ["NORMAL","HIGH"], key="chol")
        na = st.number_input("Sodium (Na)", format="%.6f", value=0.700000, key="na")
        k = st.number_input("Potassium (K)", format="%.6f", value=0.050000, key="k")

        st.session_state.predictor_inputs = {"Age":age,"Sex":sex,"BP":bp,"Cholesterol":chol,"Na":na,"K":k}

        show_models = st.checkbox("Show each model's prediction", value=False)

        if st.button("Predict (Ensemble)"):
            input_df = pd.DataFrame([[age,sex,bp,chol,na,k]], columns=['Age','Sex','BP','Cholesterol','Na','K'])
            final_pred, confidence, top3 = ensemble_predict(st.session_state["ensemble_models"], input_df)
            if final_pred is None:
                st.error("Prediction failed.")
            else:
                if confidence is not None:
                    st.success(f"Ensemble recommendation: **{final_pred}** ({confidence:.2f}% confidence)")
                else:
                    st.success(f"Ensemble recommendation: **{final_pred}** (confidence unavailable)")
                st.write("Top predictions (aggregated):")
                for i,(lab,p) in enumerate(top3, start=1):
                    st.write(f"{i}. {lab} — {p*100:.2f}%")
                st.write("Why? Features used:")
                st.info(f"Age: {age}\nSex: {sex}\nBP: {bp}\nCholesterol: {chol}\nNa: {na}\nK: {k}")
                if final_pred in drug_details:
                    d = drug_details[final_pred]
                    st.markdown("---")
                    st.subheader(f"About {final_pred}")
                    st.write(f"Use: {d['use']}")
                    st.write(f"Mechanism: {d['mechanism']}")
                    st.write(f"Side effects: {', '.join(d['side_effects'])}")
                    st.write(f"Precautions: {d['precautions']}")
                    st.write(f"Dosage: {d['dosage']}")

            if show_models:
                st.write("---")
                st.write("Model predictions and (prob for its predicted label):")
                for name, pipe in st.session_state["ensemble_models"]:
                    try:
                        p = pipe.predict(input_df)[0]
                    except Exception:
                        p = "error"
                    prob_str = "n/a"
                    try:
                        probs = pipe.predict_proba(input_df)[0]
                        if p in pipe.classes_:
                            idx = list(pipe.classes_).index(p)
                            prob_str = f"{probs[idx]*100:.2f}%"
                    except Exception:
                        prob_str = "n/a"
                    st.write(f"{name}: {p} — {prob_str}")

    # RIGHT: Chatbot (uses ONLY Random Forest model)
    with right:
        st.write("### 💬 Smart Drug Assistant (uses Random Forest)")
        q = st.text_input("Ask something (e.g. 'Which drug for high BP?')", key="chat_input")
        col1, col2 = st.columns([1,1])
        with col1:
            ask_btn = st.button("Send")
        with col2:
            predict_btn = st.button("Predict (Chatbot RF)")

        if ask_btn and q:
            txt = q.strip().lower()
            reply = ""
            if "bp" in txt or "blood pressure" in txt:
                reply = "High BP can be treated using Amlodipine, Losartan or Atenolol depending on patient details. Use the Predictor for a data-driven recommendation."
            elif "cholesterol" in txt:
                reply = "High cholesterol is commonly treated with statins such as Atorvastatin."
            elif "side effect" in txt or "side-effect" in txt:
                reply = "Ask the drug name (e.g., 'side effects of Losartan') or check the Drug Information page."
            elif "predict" in txt or "suggest" in txt or "which drug" in txt:
                # instruct user to use Predict button or use Predict (Chatbot RF) button
                reply = "I can run a Random Forest prediction using the current Predictor inputs if you click 'Predict (Chatbot RF)'."
            else:
                reply = "I can explain drugs and run a Random Forest prediction using current predictor inputs. Try 'predict' or ask a drug name."
            st.session_state.chat_history.append(("You", q))
            st.session_state.chat_history.append(("Bot", reply))

        if predict_btn:
            inputs = st.session_state.get("predictor_inputs")
            if not inputs:
                bot_reply = "No predictor inputs found — please fill the Predictor form first."
            else:
                input_df = pd.DataFrame([[inputs["Age"], inputs["Sex"], inputs["BP"], inputs["Cholesterol"], inputs["Na"], inputs["K"]]],
                                        columns=['Age','Sex','BP','Cholesterol','Na','K'])
                try:
                    rf = st.session_state["rf_model"]
                    pred = rf.predict(input_df)[0]
                    proba = None
                    try:
                        proba = rf.predict_proba(input_df)[0]
                    except Exception:
                        proba = None
                    if proba is not None:
                        if pred in rf.classes_:
                            idx = list(rf.classes_).index(pred)
                            conf = proba[idx]*100.0
                            bot_reply = f"Random Forest predicts **{pred}** with **{conf:.2f}%** confidence."
                        else:
                            bot_reply = f"Random Forest predicts **{pred}** (confidence unavailable)."
                    else:
                        bot_reply = f"Random Forest predicts **{pred}** (probabilities not available)."
                except Exception as e:
                    bot_reply = "Prediction error: " + str(e)
            st.session_state.chat_history.append(("You", "Predict (Chatbot RF)"))
            st.session_state.chat_history.append(("Bot", bot_reply))

        # show last chat messages
        if st.session_state.chat_history:
            st.markdown("---")
            for who, msg in st.session_state.chat_history[-8:]:
                if who == "You":
                    st.markdown(f"**You:** {msg}")
                else:
                    st.markdown(f"**Bot:** {msg}")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Drug Information page
# ---------------------------
if page == "Drug Information":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("💊 Drug Information")
    for name, info in drug_details.items():
        with st.expander(f"📌 {name}"):
            st.write(f"**Use:** {info['use']}")
            st.write(f"**Mechanism:** {info['mechanism']}")
            st.write(f"**Side effects:** {', '.join(info['side_effects'])}")
            st.write(f"**Precautions:** {info['precautions']}")
            st.write(f"**Dosage:** {info['dosage']}")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Admin page (in-memory)
# ---------------------------
if page == "Admin":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("👤 Admin — User Management")
    st.write("Current users (in-memory):")
    for u in USERS:
        st.write("•", u)
    st.write("---")
    st.write("Add new user")
    new_user = st.text_input("Username", key="new_user")
    new_pass = st.text_input("Password", key="new_pass")
    if st.button("Add User"):
        if not new_user or not new_pass:
            st.error("Provide both username and password.")
        elif new_user in USERS:
            st.error("User already exists.")
        else:
            USERS[new_user] = new_pass
            st.success(f"User '{new_user}' added (in-memory).")
    st.write("---")
    st.write("Remove user")
    rem = st.selectbox("Select user", list(USERS.keys()), key="rem_user")
    if st.button("Delete User"):
        if rem == "admin":
            st.error("Cannot remove admin.")
        else:
            USERS.pop(rem, None)
            st.success(f"User '{rem}' removed.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# About page
# ---------------------------
if page == "About":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("ℹ️ About Smart Drug Shield")
    st.write("""
    Smart Drug Shield predicts drugs from patient features (Age, Sex, BP, Cholesterol, Na, K).
    - Predictor uses an ensemble of models (majority vote) and aggregated probabilities.
    - Chatbot uses a single Random Forest model for predictions + confidence.
    - Drug information is provided for common drugs.
    - For demonstration / learning only — not a clinical tool.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
