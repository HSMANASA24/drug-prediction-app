# app.py - Smart Drug Shield (NL prediction, persisted models -> models/, Chatbot uses RF)
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import joblib
from pathlib import Path
from datetime import datetime
from collections import Counter

# ML imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# plotting
import matplotlib.pyplot as plt

# ---------------------------
# App config & basic UI CSS
# ---------------------------
st.set_page_config(layout="wide", page_title="🛡 Smart Drug Shield", page_icon="💊")

# simple glass + medical theme
st.markdown(
    """
    <style>
    body { background: linear-gradient(135deg,#e8f9ff,#f1fff0); }
    .glass { background: rgba(255,255,255,0.90); border-radius:12px; padding:16px; box-shadow:0 8px 30px rgba(0,0,0,0.06); border:1px solid rgba(190,233,255,0.6); }
    h1,h2,h3,h4 { color:#0A3D62 !important; font-weight:800; }
    .stButton>button { background-color:#0A3D62 !important; color:white !important; border-radius:8px !important; padding:8px 12px !important; }
    .small-muted { color:#145A32; font-size:12px; opacity:0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Paths and persistence (writable folder)
# ---------------------------
MODEL_DIR = Path("models")
try:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    # fallback to current directory
    MODEL_DIR = Path(".")
ENSEMBLE_PATH = MODEL_DIR / "ensemble_models.joblib"
RF_PATH = MODEL_DIR / "rf_model.joblib"

LOCAL_CSV = Path("/mnt/data/Drug.csv")
GITHUB_RAW = "https://raw.githubusercontent.com/HSMANASA24/drug-prediction-app/c476f30acf26ddc14b6b4a7eb796786c23a23edd/Drug.csv"

# ---------------------------
# Load dataset (local or GitHub raw)
# ---------------------------
@st.cache_data
def load_dataset():
    if LOCAL_CSV.exists():
        df = pd.read_csv(LOCAL_CSV)
    else:
        df = pd.read_csv(GITHUB_RAW)
    df.columns = [c.strip() for c in df.columns]
    mapping = {"drugA": "Amlodipine", "drugB": "Atenolol", "drugC": "ORS-K", "drugX": "Atorvastatin", "drugY": "Losartan"}
    if "Drug" in df.columns:
        df["Drug"] = df["Drug"].map(mapping).fillna(df["Drug"])
    return df

try:
    df_full = load_dataset()
except Exception as e:
    st.error("Failed to load dataset: " + str(e))
    st.stop()

# ---------------------------
# Drug details (with severity labels)
# ---------------------------
drug_details = {
    "Amlodipine": {
    "use": "Used to lower blood pressure and treat angina.",
    "mechanism": "Calcium channel blocker that relaxes blood vessels and improves blood flow.",
    "side_effects": ["Swelling of ankles", "Dizziness", "Fatigue", "Headache"],
    "precautions": "Use caution in liver disease; avoid sudden standing due to dizziness.",
    "dosage": "5–10 mg once daily.",
    "food_to_eat": ["Leafy greens", "Bananas", "Whole grains", "Low-salt foods"],
    "food_to_avoid": ["Grapefruit juice", "High-sodium foods", "Alcohol"],
    "drug_interactions": [
        "Simvastatin (increases risk of muscle damage)",
        "Blood pressure medicines (may lead to hypotension)",
        "Antifungals (increase Amlodipine levels)"
    ],
    "adverse_reactions": ["Severe leg swelling", "Shortness of breath", "Irregular heartbeat"],
    "hospitality_risk": "Moderate — hospital attention required if severe swelling or chest pain occurs."
},

    "Atenolol": {
    "use": "Used for blood pressure control and heart rate reduction.",
    "mechanism": "Beta-1 blocker that reduces heart workload.",
    "side_effects": ["Fatigue", "Cold hands/feet", "Slow heart rate"],
    "precautions": "Avoid in asthma; taper slowly—do not stop suddenly.",
    "dosage": "25–50 mg daily.",
    "food_to_eat": ["High-fiber foods", "Fruits and vegetables"],
    "food_to_avoid": ["High-salt foods", "Caffeine", "Alcohol"],
    "drug_interactions": [
        "Calcium channel blockers (risk of severe bradycardia)",
        "Insulin (may mask low blood sugar symptoms)",
        "NSAIDs (reduce BP effect)"
    ],
    "adverse_reactions": ["Very slow heartbeat", "Breathing difficulty", "Fainting"],
    "hospitality_risk": "High — emergency care needed if heart rate becomes dangerously low."
},

    "ORS-K": {
    "use": "Restores sodium–potassium balance in dehydration.",
    "mechanism": "Replenishes electrolytes lost from sweat, diarrhea, fever.",
    "side_effects": ["Nausea", "Stomach bloating"],
    "precautions": "Avoid excess potassium intake.",
    "dosage": "As required depending on dehydration level.",
    "food_to_eat": ["Coconut water", "Bananas", "Rice porridge"],
    "food_to_avoid": ["Excess salty snacks", "High-sugar drinks"],
    "drug_interactions": [
        "ACE inhibitors (risk of hyperkalemia)",
        "Potassium supplements",
        "NSAIDs (may increase potassium)"
    ],
    "adverse_reactions": ["Severe vomiting", "High potassium leading to heart rhythm issues"],
    "hospitality_risk": "Low — but rises to moderate if potassium levels spike."
},

    "Atorvastatin": {
    "use": "Used for lowering cholesterol and reducing cardiac risk.",
    "mechanism": "Inhibits HMG-CoA reductase to reduce LDL cholesterol.",
    "side_effects": ["Muscle pain", "Liver enzyme elevation"],
    "precautions": "Avoid in liver disease; monitor cholesterol and liver enzymes.",
    "dosage": "10–20 mg once daily (evening).",
    "food_to_eat": ["Oats", "Nuts", "Olive oil", "Fatty fish"],
    "food_to_avoid": ["Grapefruit juice", "High-fat meals", "Alcohol"],
    "drug_interactions": [
        "Amlodipine (increases statin levels)",
        "Antibiotics like clarithromycin",
        "Antifungals"
    ],
    "adverse_reactions": ["Severe muscle breakdown (rare)", "Liver injury"],
    "hospitality_risk": "Moderate — serious muscle pain requires evaluation."
},

    "Losartan": {
    "use": "Used to treat high blood pressure and kidney protection in diabetes.",
    "mechanism": "Blocks angiotensin II receptors, causing vasodilation.",
    "side_effects": ["Dizziness", "High potassium"],
    "precautions": "Avoid in pregnancy; monitor potassium and kidney function.",
    "dosage": "25–50 mg once daily.",
    "food_to_eat": ["Low-salt foods", "Berries", "Garlic"],
    "food_to_avoid": ["Potassium-rich salt substitutes", "Bananas", "Tomatoes"],
    "drug_interactions": [
        "Potassium supplements (hyperkalemia)",
        "ACE inhibitors",
        "NSAIDs (reduce effectiveness)"
    ],
    "adverse_reactions": ["Severe kidney issues", "Very high potassium", "Fainting"],
    "hospitality_risk": "High — severe electrolyte imbalance needs medical attention."
},

}

# ---------------------------
# Compatibility helper for OneHotEncoder param differences
# ---------------------------
def onehot_factory():
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown='ignore')

# ---------------------------
# Build pipeline factory
# ---------------------------
def build_pipeline(estimator):
    pre = ColumnTransformer([
        ("num", StandardScaler(), ['Age','Na','K']),
        ("cat", onehot_factory(), ['Sex','BP','Cholesterol'])
    ])
    return Pipeline([("pre", pre), ("clf", estimator)])

# ---------------------------
# Train & persist models (ensemble + RF)
# ---------------------------
def train_and_persist_models(df):
    df_train = df.dropna().copy()
    X = df_train[['Age','Sex','BP','Cholesterol','Na','K']]
    y = df_train['Drug']

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True)
    }

    trained = []
    for name, est in models.items():
        pipe = build_pipeline(est)
        pipe.fit(X, y)
        trained.append((name, pipe))

    # persist
    try:
        joblib.dump(trained, ENSEMBLE_PATH)
    except Exception as e:
        st.warning("Could not persist ensemble: " + str(e))

    # Random Forest for chatbot
    rf_pipe = build_pipeline(RandomForestClassifier())
    rf_pipe.fit(X, y)
    try:
        joblib.dump(rf_pipe, RF_PATH)
    except Exception as e:
        st.warning("Could not persist RF model: " + str(e))

    return trained, rf_pipe

def load_models_if_exist():
    trained = None
    rf_pipe = None
    if ENSEMBLE_PATH.exists():
        try:
            trained = joblib.load(ENSEMBLE_PATH)
        except Exception:
            trained = None
    if RF_PATH.exists():
        try:
            rf_pipe = joblib.load(RF_PATH)
        except Exception:
            rf_pipe = None
    return trained, rf_pipe

# load or train
ensemble_models, rf_model = load_models_if_exist()
if ensemble_models is None or rf_model is None:
    with st.spinner("Training models (first run)..."):
        ensemble_models, rf_model = train_and_persist_models(df_full)

# store in session state
if "ensemble_models" not in st.session_state:
    st.session_state["ensemble_models"] = ensemble_models
if "rf_model" not in st.session_state:
    st.session_state["rf_model"] = rf_model

# ---------------------------
# Ensemble prediction helper
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
    if not preds:
        return None, None, []
    final = Counter(preds).most_common(1)[0][0]
    probs_for_final = []
    for classes, proba in prob_list:
        if final in classes:
            idx = list(classes).index(final)
            probs_for_final.append(proba[idx])
    conf = float(np.mean(probs_for_final))*100.0 if probs_for_final else None
    agg = {}
    for classes, proba in prob_list:
        for cls, p in zip(classes, proba):
            agg.setdefault(cls, []).append(p)
    avg_probs = {cls: float(np.mean(ps)) for cls, ps in agg.items()}
    sorted_avg = sorted(avg_probs.items(), key=lambda x: x[1], reverse=True)
    top3 = sorted_avg[:3]
    return final, conf, top3

# ---------------------------
# Natural language parser (heuristic)
# ---------------------------
def parse_nl_for_features(text):
    text = text.lower()
    res = {"Age": None, "Sex": None, "BP": None, "Cholesterol": None, "Na": None, "K": None}
    m = re.search(r'age\s*(?:is\s*)?(\d{1,3})', text)
    if m:
        age = int(m.group(1))
        if 1 <= age <= 120:
            res["Age"] = age
    else:
        m2 = re.search(r'(\d{1,3})\s*(?:years|yrs|y/o|yo)', text)
        if m2:
            age = int(m2.group(1))
            if 1 <= age <= 120:
                res["Age"] = age
    if re.search(r'\bfemale\b|\bwoman\b|\bgirl\b', text):
        res["Sex"] = "F"
    elif re.search(r'\bmale\b|\bman\b|\bboy\b', text):
        res["Sex"] = "M"
    if "high bp" in text or "high blood pressure" in text:
        res["BP"] = "HIGH"
    elif "low bp" in text or "low blood pressure" in text:
        res["BP"] = "LOW"
    elif "normal bp" in text:
        res["BP"] = "NORMAL"
    if "high cholesterol" in text:
        res["Cholesterol"] = "HIGH"
    elif "normal cholesterol" in text:
        res["Cholesterol"] = "NORMAL"
    m_na = re.search(r'(?:na|sodium)\s*(?:=|:)?\s*([0-9]*\.?[0-9]+)', text)
    if m_na:
        try:
            res["Na"] = float(m_na.group(1))
        except:
            pass
    m_k = re.search(r'(?:k|potassium)\s*(?:=|:)?\s*([0-9]*\.?[0-9]+)', text)
    if m_k:
        try:
            res["K"] = float(m_k.group(1))
        except:
            pass
    floats = re.findall(r'([0-9]*\.[0-9]+)', text)
    if res["Na"] is None and len(floats) >= 1:
        try:
            v = float(floats[0])
            if 0.1 <= v <= 1.5:
                res["Na"] = v
        except:
            pass
    if res["K"] is None and len(floats) >= 2:
        try:
            v = float(floats[1])
            if 0.01 <= v <= 0.5:
                res["K"] = v
        except:
            pass
    any_feature = any([res[k] is not None for k in res.keys()])
    return res if any_feature else None

# ---------------------------
# Simple medically smart responses
# ---------------------------
def medical_answer(query):
    q = query.lower()
    if "bp" in q or "blood pressure" in q or "hypertension" in q:
        return ("High blood pressure (hypertension) can be treated with Amlodipine, Losartan or Atenolol depending on patient details. "
                "Use the Predictor for a data-driven recommendation.")
    if "cholesterol" in q or "statin" in q:
        return ("High cholesterol is often treated with statins like Atorvastatin. Monitor for muscle pain and check liver enzymes.")
    m = re.search(r'side effects of ([a-zA-Z0-9\- ]+)', q)
    if m:
        drug = m.group(1).strip().title()
        if drug in drug_details:
            se = ', '.join([f"{s} ({sev})" for s,sev in drug_details[drug]['side_effects']])
            return f"Common side-effects of {drug}: {se}."
        else:
            return f"I don't have detailed side-effect info for {drug} in the database."
    return ("I can explain drugs and run a Random Forest prediction using the current Predictor inputs. "
            "Try 'predict' with values, or click the Chatbot Predict button.")

# ---------------------------
# Small in-memory user list (plain text passwords per your request)
# ---------------------------
USERS = {
    "admin": "Admin@123",
    "manasa": "Manasa@2005",
    "doctor": "Doctor@123",
    "student": "Student@123"
}

# session init
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "predictor_inputs" not in st.session_state:
    st.session_state["predictor_inputs"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ---------------------------
# Login UI
# ---------------------------
def login_page():
    st.markdown('<div class="glass" style="max-width:760px; margin:auto;">', unsafe_allow_html=True)
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
            st.error("Invalid username or password")
    st.markdown('</div>', unsafe_allow_html=True)

if not st.session_state["authenticated"]:
    login_page()
    st.stop()

# ---------------------------
# Top header + tabs
# ---------------------------
st.markdown("<div class='glass'><h1 style='margin:0;'>🛡 Smart Drug Shield</h1>"
            "<p class='small-muted' style='margin:0;'>AI-powered drug prescription classifier — Medical theme</p></div>",
            unsafe_allow_html=True)

tabs = st.tabs(["Predictor", "Chatbot", "Drug Information", "Dashboard", "Admin", "About"])
tab_predictor, tab_chatbot, tab_druginfo, tab_dashboard, tab_admin, tab_about = tabs

# ---------------------------
# Predictor tab
# ---------------------------
with tab_predictor:
    st.markdown("<div class='glass'><h3>Predictor (Ensemble)</h3>", unsafe_allow_html=True)
    left, right = st.columns([2,1])

    with left:
        st.subheader("Enter patient details")
        age = st.number_input("Age", min_value=1, max_value=120, value=45, key="p_age")
        sex = st.selectbox("Sex", ["F","M"], key="p_sex")
        bp = st.selectbox("Blood Pressure (BP)", ["LOW","NORMAL","HIGH"], key="p_bp")
        chol = st.selectbox("Cholesterol", ["NORMAL","HIGH"], key="p_chol")
        na = st.number_input("Sodium (Na)", format="%.6f", value=0.700000, key="p_na")
        k = st.number_input("Potassium (K)", format="%.6f", value=0.050000, key="p_k")

        st.session_state['predictor_inputs'] = {"Age": age, "Sex": sex, "BP": bp, "Cholesterol": chol, "Na": na, "K": k}

        if st.button("Run Ensemble Prediction"):
            input_df = pd.DataFrame([[age, sex, bp, chol, na, k]], columns=['Age','Sex','BP','Cholesterol','Na','K'])
            final, conf, top3 = ensemble_predict(st.session_state["ensemble_models"], input_df)
            if final is None:
                st.error("Ensemble prediction failed.")
            else:
                if conf is not None:
                    st.success(f"Ensemble recommends: **{final}** ({conf:.2f}% confidence)")
                else:
                    st.success(f"Ensemble recommends: **{final}** (confidence not available)")

                st.write("Top 3 (aggregated):")
                for i,(lab,p) in enumerate(top3, start=1):
                    st.write(f"{i}. {lab} — {p*100:.2f}%")

                st.markdown("**Why?** Feature summary used:")
                st.info(f"Age: {age}  •  Sex: {sex}  •  BP: {bp}  •  Cholesterol: {chol}  •  Na: {na}  •  K: {k}")

                if final in drug_details:
                    st.markdown("---")
                    d = drug_details[final]
                    st.subheader(f"About {final}")
                    st.write(f"**Use:** {d['use']}")
                    st.write(f"**Mechanism:** {d['mechanism']}")
                    st.write("**Side effects (severity):**")
                    for s, sev in d['side_effects']:
                        st.write(f"- {s} ({sev})")
                    st.write(f"**Precautions:** {d['precautions']}")
                    st.write(f"**Dosage:** {d['dosage']}")

    with right:
        st.subheader("Ensemble models")
        st.write("Models trained and persisted (if possible).")
        for name, _ in st.session_state["ensemble_models"]:
            st.write("•", name)
        st.markdown("---")
        st.write("Dataset sample:")
        st.dataframe(df_full.head(6))
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Chatbot tab (uses single Random Forest)
# ---------------------------
with tab_chatbot:
    st.markdown("<div class='glass'><h3>💬 Smart Drug Assistant (Random Forest)</h3>", unsafe_allow_html=True)
    st.write("Ask questions or run a RF prediction using your current Predictor inputs (chatbot uses one RF model).")

    col_input, col_buttons = st.columns([3,1])
    with col_input:
        user_text = st.text_input("Your message:", key="chat_input_field")
    with col_buttons:
        send = st.button("Send")
        predict_chat = st.button("Predict (Chatbot RF)")

    if send and user_text:
        st.session_state.setdefault("chat_history", [])
        parsed = parse_nl_for_features(user_text)
        if parsed:
            st.session_state["chat_history"].append(("You", user_text))
            st.session_state["chat_history"].append(("Bot", "I found some features in your sentence — click 'Predict (Chatbot RF)' to run prediction with these values (or I'll combine them with Predictor inputs)."))
            st.session_state["_nl_parsed"] = parsed
        else:
            reply = medical_answer(user_text)
            st.session_state["chat_history"].append(("You", user_text))
            st.session_state["chat_history"].append(("Bot", reply))

    if predict_chat:
        st.session_state.setdefault("chat_history", [])
        parsed = st.session_state.get("_nl_parsed")
        if parsed and any(parsed.values()):
            inputs = st.session_state.get("predictor_inputs", {})
            combined = {
                "Age": parsed.get("Age") or inputs.get("Age"),
                "Sex": parsed.get("Sex") or inputs.get("Sex"),
                "BP": parsed.get("BP") or inputs.get("BP"),
                "Cholesterol": parsed.get("Cholesterol") or inputs.get("Cholesterol"),
                "Na": parsed.get("Na") or inputs.get("Na"),
                "K": parsed.get("K") or inputs.get("K"),
            }
        else:
            combined = st.session_state.get("predictor_inputs")
        if not combined:
            bot_reply = "No predictor inputs available. Fill the Predictor tab or supply values in chat."
            st.session_state["chat_history"].append(("You", "Predict (Chatbot RF)"))
            st.session_state["chat_history"].append(("Bot", bot_reply))
        else:
            try:
                in_df = pd.DataFrame([[int(combined["Age"]), combined["Sex"], combined["BP"], combined["Cholesterol"], float(combined["Na"]), float(combined["K"])]],
                                     columns=['Age','Sex','BP','Cholesterol','Na','K'])
            except Exception:
                st.session_state["chat_history"].append(("You", "Predict (Chatbot RF)"))
                st.session_state["chat_history"].append(("Bot", "Invalid/incomplete inputs for prediction."))
                in_df = None
            if in_df is not None:
                try:
                    rf = st.session_state["rf_model"]
                    pred = rf.predict(in_df)[0]
                    proba = None
                    try:
                        proba = rf.predict_proba(in_df)[0]
                    except Exception:
                        proba = None
                    if proba is not None and pred in rf.classes_:
                        idx = list(rf.classes_).index(pred)
                        conf = proba[idx]*100.0
                        bot_reply = f"Random Forest predicts **{pred}** with **{conf:.2f}%** confidence."
                    else:
                        bot_reply = f"Random Forest predicts **{pred}** (confidence unavailable)."
                except Exception as e:
                    bot_reply = "Prediction error: " + str(e)
                st.session_state["chat_history"].append(("You", "Predict (Chatbot RF)"))
                st.session_state["chat_history"].append(("Bot", bot_reply))

    # -------------------------------------------------
# Sidebar Navigation (MUST COME AFTER LOGIN)
# -------------------------------------------------
if st.session_state["authenticated"]:

    with st.sidebar:
        st.header(f"Welcome, {st.session_state['username']}")

        page = st.radio(
            "📄 Navigate",
            ["Predictor", "Drug Information", "Admin", "About"]
        )

        if st.button("Logout"):
            st.session_state["authenticated"] = False
            st.session_state["username"] = None
            st.experimental_rerun()


if page == "Drug Information":
    for drug, info in drug_details.items():
        with st.expander(f"📌 {drug_name}"):
            st.markdown(f"**Use:** {d['use']}")
            st.markdown(f"**Mechanism:** {d['mechanism']}")

            st.markdown("### Side Effects")
            for s in d['side_effects']:
                st.write(f"• {s}")

            st.markdown("### Foods to Avoid")
            for f in d['avoid_foods']:
                st.write(f"• {f}")

            st.markdown("### Foods to Eat")
            for f in d['recommended_foods']:
                st.write(f"• {f}")

            st.markdown("### Drug Interactions")
            for inter in d['interactions']:
                st.write(f"• {inter}")

            st.markdown("### Adverse Drug Reactions")
            for a in d['adr']:
                st.write(f"• {a}")

            st.markdown("### Hospitalization Risks")
            for h in d['hospital_risk']:
                st.write(f"• {h}")


    

# ---------------------------
# Dashboard tab (charts)
# ---------------------------
with tab_dashboard:
    st.markdown("<div class='glass'><h3>📈 Dashboard</h3>", unsafe_allow_html=True)
    st.write("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df_full.shape[0])
    c2.metric("Unique Drugs", int(df_full['Drug'].nunique()) if 'Drug' in df_full.columns else "N/A")
    c3.metric("Features", df_full.shape[1])

    if 'Drug' in df_full.columns:
        counts = df_full['Drug'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)
    else:
        st.info("No 'Drug' column found.")

    st.write("Feature histograms")
    fig2, axes = plt.subplots(1,3, figsize=(12,3))
    df_full['Age'].hist(ax=axes[0], bins=20)
    axes[0].set_title("Age")
    if 'Na' in df_full.columns:
        df_full['Na'].hist(ax=axes[1], bins=20)
        axes[1].set_title("Na")
    if 'K' in df_full.columns:
        df_full['K'].hist(ax=axes[2], bins=20)
        axes[2].set_title("K")
    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Admin tab
# ---------------------------
with tab_admin:
    st.markdown("<div class='glass'><h3>🔧 Admin</h3>", unsafe_allow_html=True)
    st.write("Model persistence and user management (in-memory).")
    if st.button("Retrain & Persist Models (overwrite)"):
        with st.spinner("Retraining models..."):
            ensemble_models, rf_model = train_and_persist_models(df_full)
            st.session_state["ensemble_models"] = ensemble_models
            st.session_state["rf_model"] = rf_model
            st.success("Retrained and persisted models (if writing allowed).")
    st.write("Model storage folder:", str(MODEL_DIR))
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# About tab
# ---------------------------
with tab_about:
    st.markdown("<div class='glass'><h3>ℹ️ About</h3>", unsafe_allow_html=True)
    st.write("""
    Smart Drug Shield — educational demo.
    - Predictor: ensemble majority vote (LR, KNN, DT, RF, SVM).
    - Chatbot: single Random Forest model for predictions + confidence.
    - Natural language parsing for quick inline predictions.
    - Models persist to `models/` (inside app folder) if permitted.
    - NOT a clinical decision tool — demonstration/learning only.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
