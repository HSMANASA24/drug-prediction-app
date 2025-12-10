# app.py - Smart Drug Shield (Updated as per requirements)
import streamlit as st
import pandas as pd
import numpy as np
import os
import secrets
import re
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="🛡 Smart Drug Shield", page_icon="💊", layout="centered")

st.markdown("""
    <div style='text-align:center; margin-top:-20px;'>
        <h1 style='font-size:40px; font-weight:900; color:#0b3d3d;'>🛡 Smart Drug Shield</h1>
        <h4 style='color:#0b6f6f; margin-top:-10px;'>
            <b>AI-powered drug prescription classifier — Medical Theme</b>
        </h4>
    </div>
""", unsafe_allow_html=True)

# ---------------------------
# Simple CSS (medical theme)
# ---------------------------
st.markdown("""
<style>
.glass { backdrop-filter: blur(6px); background: rgba(255,255,255,0.92); border-radius:12px; padding:14px; margin-bottom:12px; }
h1,h2,h3,h4 { font-weight:800; color:#034f4f; }
.stButton>button { border-radius:8px !important; padding:8px 12px !important; background-color:#028090 !important; color:white !important; }
.stTextInput>div>input { border-radius:8px !important; }
section[data-testid="stSidebar"] { background: #f7fdfd !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Dataset paths - CSV only (local first, else GitHub RAW)
# ---------------------------
LOCAL_PATH = "/mnt/data/Drug.csv"
GITHUB_RAW = "https://raw.githubusercontent.com/HSMANASA24/drug-prediction-app/c476f30acf26ddc14b6b4a7eb796786c23a23edd/Drug.csv"

@st.cache_data(ttl=3600)
def load_dataset():
    if os.path.exists(LOCAL_PATH):
        df = pd.read_csv(LOCAL_PATH)
    else:
        df = pd.read_csv(GITHUB_RAW)
    df.columns = [c.strip() for c in df.columns]
    # Map original labels to meaningful drug names
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
    st.error("Failed to load dataset. Check network or provide /mnt/data/Drug.csv. Error: " + str(e))
    st.stop()

st.sidebar.markdown(f"**Dataset:** {Path(LOCAL_PATH).name if os.path.exists(LOCAL_PATH) else 'GitHub RAW'}")
st.sidebar.markdown(f"Rows: **{df_full.shape[0]}** | Columns: **{df_full.shape[1]}**")
st.sidebar.markdown("---")

# ---------------------------
# drug_details (fully filled, simple lists)
# ---------------------------
drug_details = {
    "Atorvastatin": {
        "use": "Used for lowering LDL cholesterol and reducing cardiovascular risk.",
        "mechanism": "HMG-CoA reductase inhibitor (statin) that reduces hepatic cholesterol synthesis and lowers LDL.",
        "side_effects": ["Muscle aches (myalgia)", "Mild elevation of liver enzymes"],
        "precautions": "Avoid in active liver disease or pregnancy; monitor liver enzymes and counsel on reporting severe muscle pain.",
        "dosage": "Typical starting dose 10–20 mg once daily (evening); titrate as needed.",
        "foods_to_avoid": ["Grapefruit juice", "Excess alcohol", "Very high-fat meals"],
        "foods_to_eat": ["High-fiber foods", "Fatty fish (omega-3)", "Vegetables and fruits"],
        "drug_interactions": ["Macrolide antibiotics", "Azole antifungals", "HIV protease inhibitors", "Fibrates"],
        "adverse_reactions": ["Rhabdomyolysis (rare)", "Severe liver injury (rare)"],
        "hospital_risk": "Moderate — hospitalization may be required for severe muscle breakdown or acute liver injury."
    },
    "Losartan": {
        "use": "Used for high blood pressure and kidney protection in diabetics.",
        "mechanism": "Angiotensin-II receptor blocker (ARB) that causes vasodilation and reduces aldosterone-mediated sodium retention.",
        "side_effects": ["Dizziness", "Hyperkalemia (raised potassium)"],
        "precautions": "Avoid in pregnancy; monitor renal function and potassium.",
        "dosage": "Typical 25–50 mg once daily (adjust per response).",
        "foods_to_avoid": ["Potassium-rich salt substitutes", "Excess bananas/tomato products if hyperkalemic"],
        "foods_to_eat": ["Low-sodium foods", "Fresh fruits and vegetables"],
        "drug_interactions": ["Potassium supplements", "NSAIDs", "ACE inhibitors (cautious)"],
        "adverse_reactions": ["Significant hyperkalemia", "Acute kidney injury in susceptible patients"],
        "hospital_risk": "High if severe electrolyte disturbances or acute kidney injury occur."
    },
    "Atenolol": {
        "use": "Used for blood pressure control, angina and heart rate management.",
        "mechanism": "Selective β1-blocker reducing heart rate and cardiac output.",
        "side_effects": ["Fatigue", "Bradycardia (slow heart rate)", "Cold extremities"],
        "precautions": "Avoid in asthma/COPD; taper dose before stopping to avoid rebound tachycardia/angina.",
        "dosage": "Typical 50 mg once daily (range 25–100 mg as per indication).",
        "foods_to_avoid": ["Excess caffeine", "High-sodium processed foods"],
        "foods_to_eat": ["High-fiber foods", "Vegetables, fruits"],
        "drug_interactions": ["Calcium channel blockers (additive)", "Antiarrhythmics"],
        "adverse_reactions": ["Severe bradycardia", "Worsening heart failure in some patients"],
        "hospital_risk": "Moderate — severe bradycardia or bronchospasm needs urgent care."
    },
    "ORS-K": {
        "use": "Oral rehydration and electrolyte therapy to correct Na+/K+ balance in dehydration.",
        "mechanism": "Replenishes fluids and electrolytes to restore homeostasis.",
        "side_effects": ["Nausea", "Bloating"],
        "precautions": "Monitor serum electrolytes in renal impairment; avoid excess potassium if at risk.",
        "dosage": "As clinically indicated per severity of dehydration/electrolyte loss.",
        "foods_to_avoid": ["High-sugar drinks", "Very salty snacks"],
        "foods_to_eat": ["Coconut water (natural electrolytes)", "Bananas (potassium)", "Soups, broths"],
        "drug_interactions": ["Potassium-sparing diuretics", "ACE inhibitors/ARBs (hyperkalemia risk)"],
        "adverse_reactions": ["Hyperkalemia in susceptible patients"],
        "hospital_risk": "Low when used correctly; increases with renal impairment or severe dehydration."
    },
    "Amlodipine": {
        "use": "Used to lower blood pressure and treat angina.",
        "mechanism": "Calcium channel blocker that dilates peripheral arteries.",
        "side_effects": ["Peripheral edema (ankle swelling)", "Dizziness", "Flushing"],
        "precautions": "Lower doses in hepatic impairment; avoid rapid standing to prevent dizziness.",
        "dosage": "5–10 mg once daily (adjust per response).",
        "foods_to_avoid": ["Grapefruit (possible interactions)", "High-sodium foods"],
        "foods_to_eat": ["Fresh fruits and vegetables", "Whole grains", "Low-salt diet"],
        "drug_interactions": ["Simvastatin (co-administration caution)", "Other antihypertensives (additive)"],
        "adverse_reactions": ["Severe edema (rare)", "Marked hypotension (rare)"],
        "hospital_risk": "Low; severe edema/hypotension may require care."
    }
}

# ---------------------------
# Simple in-memory users (plain text for demo)
# ---------------------------
USERS = {
    "poorvika": "Poorvika@123",
    "manasa": "Manasa@2005",
    "priya": "Priya@123",
    "student": "Student@123"
}

# ---------------------------
# Session defaults
# ---------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # list of (role, text)

# Interactive chatbot patient state
if "patient_step" not in st.session_state:
    st.session_state["patient_step"] = None  # 'age','sex','bp','chol'
if "patient_form" not in st.session_state:
    st.session_state["patient_form"] = {}    # stores Age, Sex, BP, Cholesterol, Na, K

# ---------------------------
# Login page
# ---------------------------
def login_page():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🔒 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Login"):
            if u in USERS and USERS[u] == p:
                st.session_state["authenticated"] = True
                st.session_state["username"] = u
                st.rerun()
            else:
                st.error("Invalid username or password")
    with col2:
        if st.button("Clear"):
            st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

if not st.session_state["authenticated"]:
    login_page()

# ---------------------------
# Sidebar nav
# ---------------------------
with st.sidebar:
    st.header(f"Welcome, {st.session_state['username']}")
    page = st.radio("📄 Navigate", ["Chatbot", "Predictor", "Drug Information", "Admin", "About"])
    st.markdown("---")
    st.write("Dataset: uses /mnt/data/Drug.csv if present, otherwise GitHub RAW.")
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.session_state["chat_history"] = []
        st.session_state["patient_step"] = None
        st.session_state["patient_form"] = {}
        st.rerun()

# ---------------------------
# Helper: OneHotEncoder compatibility
# ---------------------------
def make_onehot():
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown='ignore')

# ---------------------------
# Model builder & trainer (cached)
# ---------------------------
@st.cache_resource
def build_pipeline(model_name="RandomForest"):
    df = df_full.dropna().copy()
    X = df[['Age','Sex','BP','Cholesterol','Na','K']]
    y = df['Drug']

    pre = ColumnTransformer([
        ("num", StandardScaler(), ['Age','Na','K']),
        ("cat", make_onehot(), ['Sex','BP','Cholesterol'])
    ])

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }

    if model_name not in models:
        model_name = "RandomForest"

    pipe = Pipeline([("pre", pre), ("clf", models[model_name])])
    pipe.fit(X, y)
    return pipe

# Build DecisionTree for chatbot and as primary model
with st.spinner("Training DecisionTree model for predictions..."):
    try:
        dt_model = build_pipeline("DecisionTree")
    except Exception as e:
        st.error("Model training error: " + str(e))
        st.stop()


# ---------------------------
# Utility: parse structured patient data from text
# ---------------------------
def parse_patient_text(text):
    # simple regex rules — best-effort
    out = {}
    # Age
    m = re.search(r'\bage[:\s]*([0-9]{1,3})\b', text, re.IGNORECASE)
    if not m:
        m = re.search(r'\b([0-9]{1,3})\s*(?:years|yrs|y)\b', text, re.IGNORECASE)
    if m:
        out['Age'] = int(m.group(1))
    # Sex
    if re.search(r'\b(male|m)\b', text, re.IGNORECASE):
        out['Sex'] = 'M'
    elif re.search(r'\b(female|f)\b', text, re.IGNORECASE):
        out['Sex'] = 'F'
    # BP
    if re.search(r'\b(high blood pressure|high bp|bp high|hypertension)\b', text, re.IGNORECASE):
        out['BP'] = 'HIGH'
    elif re.search(r'\b(low blood pressure|low bp|bp low|hypotension)\b', text, re.IGNORECASE):
        out['BP'] = 'LOW'
    elif re.search(r'\b(normal bp|bp normal|normal blood pressure)\b', text, re.IGNORECASE):
        out['BP'] = 'NORMAL'
    # Cholesterol
    if re.search(r'\b(high cholesterol|cholesterol high)\b', text, re.IGNORECASE):
        out['Cholesterol'] = 'HIGH'
    elif re.search(r'\b(normal cholesterol|cholesterol normal)\b', text, re.IGNORECASE):
        out['Cholesterol'] = 'NORMAL'
    # Na and K numeric
    m_na = re.search(r'\bna[:= ]?([0-9]*\.?[0-9]+)\b', text, re.IGNORECASE)
    if m_na:
        try:
            out['Na'] = float(m_na.group(1))
        except:
            pass
    m_k = re.search(r'\bk[:= ]?([0-9]*\.?[0-9]+)\b', text, re.IGNORECASE)
    if m_k:
        try:
            out['K'] = float(m_k.group(1))
        except:
            pass
    # look for pattern like "Na 0.7 K 0.05"
    m_pair = re.search(r'na[: ]?([0-9]*\.?[0-9]+).*k[: ]?([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
    if m_pair:
        try:
            out['Na'] = float(m_pair.group(1))
            out['K'] = float(m_pair.group(2))
        except:
            pass
    return out

# ---------------------------
# Chatbot page (DecisionTree only, interactive)
# ---------------------------
if page == "Chatbot":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("💬 Smart Medical Assistant (DecisionTree)")

    # Show chat history
    for role, msg in st.session_state["chat_history"]:
        if role == "user":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg}")

    user_input = st.text_input(
        "Say hi 👋 to start an interactive assessment, ask about a drug 💊, or type 'reset' to start again.",
        key="chat_input"
    )

    col1, col2 = st.columns([1,1])

    def run_decisiontree_prediction(features_dict):
        """Run DecisionTree prediction using collected features."""
        median_age = int(df_full['Age'].median())
        default_sex = df_full['Sex'].mode()[0]
        default_bp = df_full['BP'].mode()[0]
        default_chol = df_full['Cholesterol'].mode()[0]
        default_na = float(df_full['Na'].median())
        default_k = float(df_full['K'].median())

        a = int(features_dict.get('Age', median_age))
        s = features_dict.get('Sex', default_sex)
        bpv = features_dict.get('BP', default_bp)
        chol = features_dict.get('Cholesterol', default_chol)
        na = float(features_dict.get('Na', default_na))
        k = float(features_dict.get('K', default_k))

        input_df = pd.DataFrame(
            [[a, s, bpv, chol, na, k]],
            columns=['Age','Sex','BP','Cholesterol','Na','K']
        )

        dt_probs = dt_model.predict_proba(input_df)[0]
        dt_idx = int(np.argmax(dt_probs))
        dt_label = dt_model.classes_[dt_idx]
        dt_conf = dt_probs[dt_idx] * 100.0

        return {
            "Age": a, "Sex": s, "BP": bpv, "Cholesterol": chol, "Na": na, "K": k,
            "label": dt_label, "confidence": dt_conf
        }

    with col1:
        if st.button("Send"):
            if not user_input.strip():
                st.warning("Please type something.")
            else:
                # Store user message
                st.session_state["chat_history"].append(("user", user_input))
                q_lower = user_input.lower().strip()
                clean_msg = re.sub(r'[^a-zA-Z ]', '', q_lower).strip()
                response_lines = []

                # Handle reset of interactive flow
                if clean_msg in ["reset", "restart", "new patient"]:
                    st.session_state["patient_step"] = None
                    st.session_state["patient_form"] = {}
                    response_lines.append("✅ Patient assessment has been reset.")
                    response_lines.append("Say **Hi** to start a new interactive assessment.")
                    assistant_reply = "\n".join(response_lines)
                    st.session_state["chat_history"].append(("assistant", assistant_reply))
                    st.rerun()

                # If we are in the middle of interactive patient Q&A
                patient_step = st.session_state.get("patient_step")

                if patient_step is not None:
                    form = st.session_state["patient_form"]

                    # Step: Age
                    if patient_step == "age":
                        m = re.search(r'([0-9]{1,3})', user_input)
                        if m:
                            age_val = int(m.group(1))
                            form["Age"] = age_val
                            st.session_state["patient_step"] = "sex"
                            response_lines.append(f"Got it, Age: **{age_val}** years.")
                            response_lines.append("Next, what is the **sex** of the patient? (Reply with **M** or **F**)")
                        else:
                            response_lines.append("Please enter a valid age in years (example: 45).")

                    # Step: Sex
                    elif patient_step == "sex":
                        if re.search(r'\bf\b', q_lower) or q_lower.strip() == "f":
                            form["Sex"] = "F"
                            st.session_state["patient_step"] = "bp"
                            response_lines.append("Sex: **Female (F)**.")
                            response_lines.append("Do you have **low BP**, **high BP**, or **normal BP**?")
                            response_lines.append("Reply with: **LOW / HIGH / NORMAL**.")
                        elif re.search(r'\bm\b', q_lower) or q_lower.strip() == "m":
                            form["Sex"] = "M"
                            st.session_state["patient_step"] = "bp"
                            response_lines.append("Sex: **Male (M)**.")
                            response_lines.append("Do you have **low BP**, **high BP**, or **normal BP**?")
                            response_lines.append("Reply with: **LOW / HIGH / NORMAL**.")
                        else:
                            response_lines.append("Please reply with **M** for male or **F** for female.")

                    # Step: BP
                    elif patient_step == "bp":
                        bp_val = None
                        if "high" in q_lower:
                            bp_val = "HIGH"
                        elif "low" in q_lower:
                            bp_val = "LOW"
                        elif "normal" in q_lower:
                            bp_val = "NORMAL"

                        if bp_val is not None:
                            form["BP"] = bp_val
                            st.session_state["patient_step"] = "chol"
                            response_lines.append(f"Blood Pressure: **{bp_val}**.")
                            response_lines.append("Do you have **high cholesterol** or **no cholesterol problem**?")
                            response_lines.append("Reply with **HIGH** or **NORMAL**.")
                        else:
                            response_lines.append("Please reply with **HIGH**, **LOW**, or **NORMAL** for BP.")

                    # Step: Cholesterol
                    elif patient_step == "chol":
                        chol_val = None
                        # Interpret "no / not / normal" as NORMAL
                        if "high" in q_lower:
                            chol_val = "HIGH"
                        elif "no" in q_lower or "not" in q_lower or "normal" in q_lower:
                            chol_val = "NORMAL"

                        if chol_val is not None:
                            form["Cholesterol"] = chol_val

                            # Try to parse Na/K if user gave in the same sentence, else use default later
                            parsed_ions = parse_patient_text(user_input)
                            if "Na" in parsed_ions:
                                form["Na"] = parsed_ions["Na"]
                            if "K" in parsed_ions:
                                form["K"] = parsed_ions["K"]

                            # Ready to predict via DecisionTree
                            try:
                                result = run_decisiontree_prediction(form)
                                st.session_state["patient_step"] = None
                                st.session_state["patient_form"] = {}

                                response_lines.append("✅ Thank you. I have the details I need.")
                                response_lines.append(
                                    f"Age **{result['Age']}**, Sex **{result['Sex']}**, "
                                    f"BP **{result['BP']}**, Cholesterol **{result['Cholesterol']}**."
                                )
                                # Inform about Na/K assumption if not provided
                                response_lines.append(
                                    "Sodium (Na) and Potassium (K) are taken from patient input if given; "
                                    "otherwise, average values from the dataset are used."
                                )
                                response_lines.append("")
                                response_lines.append("🤖 **DecisionTree Model Prediction:**")
                                response_lines.append(
                                    f"💊 Recommended drug: **{result['label']} ({result['confidence']:.2f}% confidence)**"
                                )

                                if result["label"] in drug_details:
                                    dd = drug_details[result["label"]]
                                    response_lines.append(f"ℹ️ {result['label']} is typically used for: {dd['use']}")
                                    response_lines.append(f"📏 Usual dosage: {dd['dosage']}")
                            except Exception as e:
                                response_lines.append("❌ Prediction failed: " + str(e))
                                st.session_state["patient_step"] = None
                                st.session_state["patient_form"] = {}
                        else:
                            response_lines.append(
                                "Please reply with **HIGH** if you have high cholesterol, "
                                "or **NORMAL** if you do not have a cholesterol problem."
                            )

                else:
                    # ---------------------------
                    # No active interactive flow: handle normal chatbot logic
                    # ---------------------------
                    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

                    if clean_msg in greetings:
                        # Start interactive Q&A
                        st.session_state["patient_step"] = "age"
                        st.session_state["patient_form"] = {}
                        response_lines.append("Hello! 👋 I'm your Smart Medical Assistant.")
                        response_lines.append("I will ask you a few questions and then predict a suitable drug using a **DecisionTree model**.")
                        response_lines.append("")
                        response_lines.append("First, what is the **age** of the patient? (example: 45)")
                    elif clean_msg in ["thanks", "thank you", "thx", "ty"]:
                        response_lines.append("You're most welcome! 😊")
                        response_lines.append("If you need any more medical help, I'm always here 💊❤️")
                    elif clean_msg in ["bye", "goodbye", "see you", "exit", "see ya"]:
                        response_lines.append("Goodbye! 👋 Take care of your health!")
                        response_lines.append("🛡 Smart Drug Shield is always here when you need me.")
                    elif "high bp" in q_lower or "hypertension" in q_lower:
                        response_lines.append("✅ I understand you are concerned about **High Blood Pressure**.")
                        response_lines.append("Let's do an interactive assessment.")
                        st.session_state["patient_step"] = "age"
                        st.session_state["patient_form"] = {}
                        response_lines.append("What is the **age** of the patient?")
                    elif "low bp" in q_lower or "hypotension" in q_lower:
                        response_lines.append("✅ I understand you are concerned about **Low Blood Pressure**.")
                        response_lines.append("Let's do an interactive assessment.")
                        st.session_state["patient_step"] = "age"
                        st.session_state["patient_form"] = {}
                        response_lines.append("What is the **age** of the patient?")
                    else:
                        # Drug information handler
                        found_drug = None
                        for dname in drug_details.keys():
                            if re.search(r'\b' + re.escape(dname) + r'\b', user_input, re.IGNORECASE):
                                found_drug = dname
                                break

                        if found_drug:
                            d = drug_details[found_drug]
                            response_lines.append(f"💊 **{found_drug} Information:**")
                            response_lines.append(f"• Use: {d['use']}")
                            response_lines.append(f"• Mechanism: {d['mechanism']}")
                            response_lines.append(f"• Side effects: {', '.join(d['side_effects'])}")
                            response_lines.append(f"• Dosage: {d['dosage']}")
                        else:
                            # Try parsing free text patient details directly for DecisionTree prediction
                            parsed = parse_patient_text(user_input)

                            if parsed:
                                try:
                                    result = run_decisiontree_prediction(parsed)
                                    response_lines.append("🤖 **DecisionTree Model Prediction (from your text):**")
                                    response_lines.append(
                                        f"💊 Recommended drug: **{result['label']} ({result['confidence']:.2f}% confidence)**"
                                    )
                                    if result["label"] in drug_details:
                                        dd = drug_details[result["label"]]
                                        response_lines.append(f"ℹ️ {result['label']} is typically used for: {dd['use']}")
                                        response_lines.append(f"📏 Usual dosage: {dd['dosage']}")
                                except Exception as e:
                                    response_lines.append("❌ Prediction failed: " + str(e))
                            else:
                                # Final generic help message
                                response_lines.append("🤖 I'm here to help!")
                                response_lines.append("You can:")
                                response_lines.append("• Say **Hi / Hello** to start interactive prediction")
                                response_lines.append("• Ask about a drug (example: *Amlodipine*)")
                                response_lines.append("• Or type patient info like: `45 M HIGH BP Na 0.70 K 0.05`")

                assistant_reply = "\n".join(response_lines)
                st.session_state["chat_history"].append(("assistant", assistant_reply))
                st.rerun()

    with col2:
        if st.button("Clear Chat"):
            st.session_state["chat_history"] = []
            st.session_state["patient_step"] = None
            st.session_state["patient_form"] = {}
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------
# Predictor page (multi-model, top-3 models)
# ---------------------------
if page == "Predictor":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🧪 Predictor — single patient input")

    # Build all models (cached by name)
    model_names = ["DecisionTree", "RandomForest", "LogisticRegression", "KNN", "SVM"]

    with st.spinner("Training models (cached on first use)..."):
        model_pipes = {}
        try:
            for name in model_names:
                model_pipes[name] = build_pipeline(name)
        except Exception as e:
            st.error("Training error: " + str(e))
            st.stop()

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=int(df_full['Age'].median()))
        sex = st.selectbox("Sex", ["F","M"])
        bpv = st.selectbox("Blood Pressure (BP)", ["LOW","NORMAL","HIGH"])
    with col2:
        chol = st.selectbox("Cholesterol", ["HIGH","NORMAL"])
        na = st.number_input("Sodium (Na)", format="%.6f", value=float(df_full['Na'].median()))
        k = st.number_input("Potassium (K)", format="%.6f", value=float(df_full['K'].median()))

    if st.button("Predict"):
        input_df = pd.DataFrame([[age, sex, bpv, chol, na, k]], columns=['Age','Sex','BP','Cholesterol','Na','K'])
        model_results = []

        try:
            for name, pipe in model_pipes.items():
                try:
                    probs = pipe.predict_proba(input_df)[0]
                    idx = int(np.argmax(probs))
                    label = pipe.classes_[idx]
                    conf = probs[idx] * 100.0
                    model_results.append({
                        "Model": name,
                        "Predicted Drug": label,
                        "Confidence (%)": conf
                    })
                except Exception as e:
                    st.warning(f"Prediction error for {name}: {e}")

            if not model_results:
                st.error("No model could produce a prediction.")
            else:
                # Sort by confidence descending
                model_results_sorted = sorted(model_results, key=lambda x: x["Confidence (%)"], reverse=True)

                # Top 3 models (as requested)
                st.markdown("### 🏆 Top 3 model predictions")
                for rank, res in enumerate(model_results_sorted[:3], start=1):
                    st.write(
                        f"{rank}. **{res['Model']}** → {res['Predicted Drug']} "
                        f"(**{res['Confidence (%)']:.2f}%** confidence)"
                    )

                # Primary model = DecisionTree (to match chatbot)
                dt_primary = next((r for r in model_results_sorted if r["Model"] == "DecisionTree"), None)
                if dt_primary is not None:
                    st.markdown("---")
                    st.markdown("### ⭐ Primary model (DecisionTree) — same as Chatbot")
                    st.success(
                        f"DecisionTree predicts **{dt_primary['Predicted Drug']} "
                        f"({dt_primary['Confidence (%)']:.2f}% confidence)**"
                    )

                    pred_label = dt_primary["Predicted Drug"]
                else:
                    # Fallback: use best model overall
                    best = model_results_sorted[0]
                    st.markdown("---")
                    st.markdown("### ⭐ Best available model")
                    st.success(
                        f"{best['Model']} predicts **{best['Predicted Drug']} "
                        f"({best['Confidence (%)']:.2f}% confidence)**"
                    )
                    pred_label = best["Predicted Drug"]

                st.info(f"Features: Age {age}, Sex {sex}, BP {bpv}, Cholesterol {chol}, Na {na}, K {k}")

                # Show drug details if available (based on primary DecisionTree / best)
                if pred_label in drug_details:
                    dd = drug_details[pred_label]
                    st.markdown("---")
                    st.markdown(f"### About {pred_label}")
                    st.write(dd['use'])
                    st.write("Dosage:", dd['dosage'])

        except Exception as e:
            st.error("Prediction error: " + str(e))

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Drug Information page
# ---------------------------
if page == "Drug Information":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("💊 Drug Information")
    for name, info in drug_details.items():
        with st.expander(f"📌 {name}"):
            st.markdown(f"**Use:** {info['use']}")
            st.markdown(f"**Mechanism:** {info['mechanism']}")
            st.markdown("**Side effects:**")
            for s in info['side_effects']:
                st.write("•", s)
            st.markdown(f"**Precautions:** {info['precautions']}")
            st.markdown(f"**Dosage:** {info['dosage']}")
            st.markdown("**Foods to avoid:**")
            for f in info.get('foods_to_avoid', []):
                st.write("•", f)
            st.markdown("**Foods to eat:**")
            for f in info.get('foods_to_eat', []):
                st.write("•", f)
            st.markdown("**Drug interactions:**")
            for di in info.get('drug_interactions', []):
                st.write("•", di)
            st.markdown("**Adverse reactions:**")
            for a in info.get('adverse_reactions', info.get('adverse_reactions', [])):
                st.write("•", a)
            st.markdown(f"**Hospitalization risk:** {info.get('hospital_risk','N/A')}")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Admin page: manage in-memory users
# ---------------------------
if page == "Admin":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("👤 Admin — User Management (in-memory demo)")
    st.write("Current users (demo):")
    for u in list(USERS.keys()):
        st.write("-", u)
    st.markdown("---")
    st.write("Add new user (in-memory only):")
    new_u = st.text_input("Username", key="admin_new_user")
    new_p = st.text_input("Password", key="admin_new_pass")
    if st.button("Add user"):
        if not new_u or not new_p:
            st.error("Provide username and password.")
        elif new_u in USERS:
            st.error("User already exists.")
        else:
            USERS[new_u] = new_p
            st.success("User added (in-memory).")
    st.markdown("---")
    st.write("Remove user:")
    remove_u = st.selectbox("Select user to remove", list(USERS.keys()), key="admin_remove")
    if st.button("Remove user"):
        if remove_u == st.session_state.get("username"):
            st.error("Cannot remove your own account while logged in.")
        else:
            USERS.pop(remove_u, None)
            st.success("User removed (in-memory).")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# About page
# ---------------------------
if page == "About":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("ℹ️ About Smart Drug Shield")
    st.markdown("""
**Smart Drug Shield** is an educational demonstration app that trains classification models on a small clinical dataset and provides:
- A **Chatbot** that interactively asks for Age, Sex, BP (LOW/HIGH/NORMAL), Cholesterol (HIGH/NORMAL) and predicts a drug using a **DecisionTree** model.
- A **Predictor** page that compares multiple models and shows the **top 3 models** with their prediction and confidence. The primary model is DecisionTree (same as chatbot).
- A **Drug Information** tab containing drug uses, mechanism, side effects, food advice, interactions and hospital risk.

**Important:** This is a demo for learning only — *not* a clinical decision tool.
""")
    st.markdown('</div>', unsafe_allow_html=True)
