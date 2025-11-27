# app.py - Smart Drug Shield (Final, Ensemble + Chatbot)
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

# ---------------------------
# App config + Title
# ---------------------------
st.set_page_config(page_title="🛡 Smart Drug Shield", page_icon="💊", layout="centered")
st.markdown("""
    <h1 style='text-align:center; font-size:40px; font-weight:900; color:#0A3D62; margin-bottom:0.2rem'>
        🛡 Smart Drug Shield
    </h1>
    <p style='text-align:center; color:#145A32; margin-top:0; margin-bottom:1rem'>
        AI-powered drug prescription classifier — Medical theme
    </p>
""", unsafe_allow_html=True)

# ---------------------------
# Medical theme CSS
# ---------------------------
MEDICAL_CSS = """
<style>
body {
    background: linear-gradient(135deg, #e8f9ff, #d4fce5);
}
.glass-panel {
    background: rgba(255,255,255,0.92);
    backdrop-filter: blur(6px);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 18px;
    border: 1px solid #bfe9ff;
}
h1,h2,h3,h4 { color:#0A3D62 !important; font-weight:800; }
label, p, span, div { color:#0A3D62 !important; }
.stButton>button {
    background-color:#0A3D62 !important;
    color:white !important;
    border-radius:8px !important;
    padding:8px 14px !important;
}
.stButton>button:hover { background-color:#145A32 !important; }
</style>
"""
st.markdown(MEDICAL_CSS, unsafe_allow_html=True)

# ---------------------------
# Simple user DB (in-memory)
# ---------------------------
USERS = {
    "admin": "Admin@123",
    "manasa": "Manasa@2005",
    "doctor": "Doctor@123",
    "student": "Student@123"
}

# ---------------------------
# Session init
# ---------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
# Store predictor input in session so chatbot can use them
if "predictor_inputs" not in st.session_state:
    st.session_state["predictor_inputs"] = None
# Store trained ensemble models
if "ensemble_models" not in st.session_state:
    st.session_state["ensemble_models"] = None

# ---------------------------
# Safe login UI
# ---------------------------
def login_page():
    st.markdown('<div class="glass-panel" style="max-width:720px; margin:auto;">', unsafe_allow_html=True)
    st.subheader("🔒 Smart Drug Shield — Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns([1, 1])
    with col1:
        login_btn = st.button("Login")
    with col2:
        clear_btn = st.button("Clear")
    if clear_btn:
        # clear fields by rerendering
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.experimental_rerun()
    if login_btn:
        if username in USERS and USERS[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
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
    st.markdown("Dataset: local `/mnt/data/Drug.csv` (preferred) or GitHub RAW fallback.")
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.experimental_rerun()

# ---------------------------
# Dataset loading (local then GitHub)
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
    # Map short codes to human-friendly drug names (if present)
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

# ---------------------------
# Drug information dictionary
# ---------------------------
drug_details = {
    "Amlodipine": {
        "use": "Lowers blood pressure by relaxing blood vessels.",
        "mechanism": "Calcium channel blocker.",
        "side_effects": ["Dizziness", "Swelling (edema)", "Headache"],
        "precautions": "Monitor BP; report severe dizziness or swelling.",
        "dosage": "5–10 mg once daily (typical adult)."
    },
    "Atenolol": {
        "use": "Controls blood pressure and heart rate.",
        "mechanism": "Selective β1-blocker reducing heart rate.",
        "side_effects": ["Fatigue", "Cold extremities", "Bradycardia"],
        "precautions": "Avoid in asthma; monitor heart rate.",
        "dosage": "50 mg once daily (adjust per clinical guidance)."
    },
    "ORS-K": {
        "use": "Corrects sodium–potassium imbalance (oral rehydration/electrolyte).",
        "mechanism": "Replenishes Na+ and K+ to restore balance.",
        "side_effects": ["Nausea", "Bloating"],
        "precautions": "Monitor electrolytes in severe cases.",
        "dosage": "As required per clinical context."
    },
    "Atorvastatin": {
        "use": "Lowers LDL cholesterol and cardiovascular risk.",
        "mechanism": "HMG-CoA reductase inhibitor (statin).",
        "side_effects": ["Muscle pain", "Liver enzyme elevation"],
        "precautions": "Check liver enzymes; avoid in pregnancy.",
        "dosage": "10–20 mg in the evening (typical start)."
    },
    "Losartan": {
        "use": "Treats high blood pressure (angiotensin receptor blocker).",
        "mechanism": "Blocks angiotensin II receptors to reduce BP.",
        "side_effects": ["Dizziness", "Increased potassium"],
        "precautions": "Avoid during pregnancy; monitor potassium.",
        "dosage": "25–50 mg once daily (adjust as needed)."
    }
}

# ---------------------------
# OneHotEncoder compatibility helper
# ---------------------------
def onehot_factory():
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown='ignore')

# ---------------------------
# Train all models and cache them
# ---------------------------
@st.cache_resource
def train_all_models(df: pd.DataFrame):
    df = df.dropna().copy()
    X = df[['Age','Sex','BP','Cholesterol','Na','K']]
    y = df['Drug']

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

    trained_pipes = []
    for name, m in models.items():
        pipe = Pipeline([("pre", pre), ("clf", m)])
        pipe.fit(X, y)
        trained_pipes.append((name, pipe))
    return trained_pipes

# Ensure models are trained and stored in session for reuse
if st.session_state["ensemble_models"] is None:
    with st.spinner("Training ensemble models on dataset..."):
        st.session_state["ensemble_models"] = train_all_models(df_full)

# Utility: ensemble predict (majority vote) and compute confidence
def ensemble_predict(model_pipes, input_df):
    # gather model predictions and probabilities
    preds = []
    prob_list = []
    classes_list = []

    for name, pipe in model_pipes:
        try:
            p = pipe.predict(input_df)[0]
            preds.append(p)
        except Exception:
            continue
        # try to get probabilities
        try:
            proba = pipe.predict_proba(input_df)[0]
            prob_list.append((pipe.classes_, proba))
            classes_list.append(pipe.classes_)
        except Exception:
            # if predict_proba unavailable, skip probability for that model
            pass

    if len(preds) == 0:
        return None, None, []

    # majority vote for final prediction
    final_pred = max(set(preds), key=preds.count)

    # compute average probability for final_pred across models that provided probabilities
    probs_for_final = []
    for classes, proba in prob_list:
        # classes is array of labels for that model
        if final_pred in classes:
            idx = list(classes).index(final_pred)
            probs_for_final.append(proba[idx])
    if len(probs_for_final) > 0:
        confidence = float(np.mean(probs_for_final)) * 100.0
    else:
        confidence = None

    # compute top-3 aggregated by averaging probabilities across models that support probabilities
    # Build a dict of label -> list of probs
    agg = {}
    for classes, proba in prob_list:
        for cls, p in zip(classes, proba):
            agg.setdefault(cls, []).append(p)
    avg_probs = {cls: float(np.mean(ps)) for cls, ps in agg.items()}
    # sort labels by avg prob desc
    sorted_labels = sorted(avg_probs.items(), key=lambda x: x[1], reverse=True)
    top3 = sorted_labels[:3]

    return final_pred, confidence, top3

# ---------------------------
# Predictor + Chatbot UI (split)
# ---------------------------
if page == "Predictor":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("🔍 Single Prediction — AI Drug Classifier")

    # left predictor / right chatbot split
    left_col, right_col = st.columns([2, 1])

    # LEFT: predictor controls (user can still choose to view single-model predictions if desired)
    with left_col:
        st.write("### Predictor")
        # Save default predictor inputs to session so chatbot can use them
        age = st.number_input("Age", 1, 120, 45, key="age_input")
        sex = st.selectbox("Sex", ["F", "M"], key="sex_input")
        bp = st.selectbox("Blood Pressure (BP)", ["LOW", "NORMAL", "HIGH"], key="bp_input")
        chol = st.selectbox("Cholesterol", ["NORMAL", "HIGH"], key="chol_input")
        na = st.number_input("Sodium (Na)", format="%.6f", value=0.700000, key="na_input")
        k = st.number_input("Potassium (K)", format="%.6f", value=0.050000, key="k_input")

        # store current predictor inputs in session for chatbot
        st.session_state.predictor_inputs = {
            "Age": age, "Sex": sex, "BP": bp, "Cholesterol": chol, "Na": na, "K": k
        }

        # Show model-wise predictions if user wants (optional)
        show_models_checkbox = st.checkbox("Show each model's prediction", value=False)

        if st.button("Predict (Ensemble)"):
            input_df = pd.DataFrame([[age, sex, bp, chol, na, k]],
                                    columns=['Age','Sex','BP','Cholesterol','Na','K'])
            final_pred, confidence, top3 = ensemble_predict(st.session_state["ensemble_models"], input_df)

            if final_pred is None:
                st.error("Prediction failed. Check model training / input.")
            else:
                if confidence is not None:
                    st.success(f"Ensemble Recommendation: **{final_pred}** ({confidence:.2f}% confidence)")
                else:
                    st.success(f"Ensemble Recommendation: **{final_pred}** (confidence unavailable)")

                st.write("### Top predictions (aggregated probabilities)")
                for i, (lab, p) in enumerate(top3, start=1):
                    st.write(f"{i}. {lab} — {p*100:.2f}%")

                st.write("### Why this prediction?")
                st.info(
                    f"• Age: {age}\n"
                    f"• Sex: {sex}\n"
                    f"• BP: {bp}\n"
                    f"• Cholesterol: {chol}\n"
                    f"• Sodium (Na): {na}\n"
                    f"• Potassium (K): {k}"
                )

                # show drug info if available
                if final_pred in drug_details:
                    st.write("---")
                    d = drug_details[final_pred]
                    st.subheader(f"📌 About {final_pred}")
                    st.write(f"**Use:** {d['use']}")
                    st.write(f"**Mechanism:** {d['mechanism']}")
                    st.write(f"**Side Effects:** {', '.join(d['side_effects'])}")
                    st.write(f"**Precautions:** {d['precautions']}")
                    st.write(f"**Dosage:** {d['dosage']}")

            # optionally show individual model predictions
            if show_models_checkbox:
                st.write("---")
                st.write("Model — Prediction — Probability for predicted label (if available)")
                input_df = pd.DataFrame([[age, sex, bp, chol, na, k]],
                                        columns=['Age','Sex','BP','Cholesterol','Na','K'])
                for name, pipe in st.session_state["ensemble_models"]:
                    try:
                        p = pipe.predict(input_df)[0]
                    except Exception:
                        p = "error"
                    prob_str = ""
                    try:
                        probs = pipe.predict_proba(input_df)[0]
                        if p in pipe.classes_:
                            idx = list(pipe.classes_).index(p)
                            prob_str = f"{probs[idx]*100:.2f}%"
                        else:
                            prob_str = "0.00%"
                    except Exception:
                        prob_str = "n/a"
                    st.write(f"{name} — {p} — {prob_str}")

    # RIGHT: chatbot using the SAME ensemble (no separate model selection)
    with right_col:
        st.write("### 💬 Smart Drug Assistant")
        st.write("Ask about drugs, or use current predictor inputs to get ensemble recommendation (same as left).")

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # free-text question
        user_q = st.text_input("Type a question (e.g. 'Which drug for high BP?')", key="chat_input")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            send_btn = st.button("Send")
        with col_b:
            use_predictor_btn = st.button("Predict using current inputs")

        if send_btn and user_q:
            q = user_q.strip().lower()
            reply = ""
            # simple rule-based responses + guidance
            if "bp" in q or "blood pressure" in q:
                reply = ("High BP is often treated with drugs such as Amlodipine, Losartan or Atenolol. "
                         "Use the Predictor (left) with patient values for a data-driven recommendation.")
            elif "cholesterol" in q:
                reply = "High cholesterol is commonly treated with statins such as Atorvastatin."
            elif "why" in q and ("drug" in q or "predict" in q):
                reply = "Prediction is based on Age, Sex, BP, Cholesterol, Sodium (Na) and Potassium (K) using an ensemble of ML models."
            elif "side effect" in q or "side-effect" in q or "sideeffects" in q:
                reply = "Ask me which drug and I can show common side effects from the Drug Information page."
            else:
                reply = "I can explain drug choices, side effects, or run the ensemble prediction using current Predictor inputs. Try 'predict' or 'bp' or ask about a drug name."

            st.session_state["chat_history"].append(("You", user_q))
            st.session_state["chat_history"].append(("Bot", reply))

        if use_predictor_btn:
            # use current predictor inputs stored in session_state.predictor_inputs
            inputs = st.session_state.get("predictor_inputs")
            if inputs is None:
                st.warning("No predictor inputs available. Fill the Predictor form first.")
            else:
                input_df = pd.DataFrame([[inputs["Age"], inputs["Sex"], inputs["BP"], inputs["Cholesterol"], inputs["Na"], inputs["K"]]],
                                        columns=['Age','Sex','BP','Cholesterol','Na','K'])
                final_pred, confidence, top3 = ensemble_predict(st.session_state["ensemble_models"], input_df)
                if final_pred is None:
                    bot_reply = "Prediction failed. Please try again."
                else:
                    if confidence is not None:
                        bot_reply = f"Ensemble suggests **{final_pred}** with **{confidence:.2f}%** confidence. Top choices: " + \
                                    ", ".join([f"{lab} ({p*100:.2f}%)" for lab, p in top3])
                    else:
                        bot_reply = f"Ensemble suggests **{final_pred}** (confidence not available)."
                st.session_state["chat_history"].append(("You", "Predict using current inputs"))
                st.session_state["chat_history"].append(("Bot", bot_reply))

        # show history
        if st.session_state["chat_history"]:
            st.write("---")
            for who, text in st.session_state["chat_history"][-8:]:
                if who == "You":
                    st.markdown(f"**You:** {text}")
                else:
                    st.markdown(f"**Bot:** {text}")

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
# Admin page (in-memory user management)
# ---------------------------
if page == "Admin":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("👤 Admin — User Management")
    st.write("Current users (in-memory):")
    for u in USERS:
        st.write("•", u)
    st.write("---")
    st.write("Add user")
    new_user = st.text_input("Username", key="admin_new_user")
    new_pass = st.text_input("Password", key="admin_new_pass")
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
    remove_user = st.selectbox("Select user", list(USERS.keys()), key="admin_remove")
    if st.button("Delete User"):
        if remove_user == "admin":
            st.error("Cannot remove main admin.")
        else:
            USERS.pop(remove_user, None)
            st.success(f"User '{remove_user}' removed (in-memory).")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# About page
# ---------------------------
if page == "About":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("ℹ️ About Smart Drug Shield")
    st.markdown("""
    **Smart Drug Shield** is an educational demo that predicts a suitable drug from patient features:
    Age, Sex, Blood Pressure, Cholesterol, Sodium (Na), Potassium (K).

    Key features:
    - Ensemble (majority vote) ensures the *same recommendation* across UI and chatbot.
    - Confidence derived from averaging probabilities from models that provide them.
    - Multi-user login (in-memory users), Admin panel, Drug information.
    - **Not a clinical tool** — for learning and demonstration only.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
