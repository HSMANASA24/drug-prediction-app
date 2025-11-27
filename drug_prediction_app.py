# app.py - Smart Drug Shield (Full: NL prediction, persistence, chatbot (RF), UI, dashboard)
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import joblib
from pathlib import Path
from datetime import datetime
from collections import Counter

# ML
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
# Config and constants
# ---------------------------
st.set_page_config(layout="wide", page_title="🛡 Smart Drug Shield", page_icon="💊")

MODEL_DIR = Path("/mnt/data/smart_drug_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_CSV = Path("/mnt/data/Drug.csv")
GITHUB_RAW = "https://raw.githubusercontent.com/HSMANASA24/drug-prediction-app/c476f30acf26ddc14b6b4a7eb796786c23a23edd/Drug.csv"

# ---------------------------
# CSS (glass + medical theme)
# ---------------------------
st.markdown(
    """
    <style>
    /* page bg */
    body { background: linear-gradient(135deg,#e8f9ff,#f1fff0); }

    /* glass panel */
    .glass {
      background: rgba(255,255,255,0.86);
      border-radius: 12px;
      padding: 18px;
      box-shadow: 0 8px 30px rgba(0,0,0,0.08);
      border: 1px solid rgba(190,233,255,0.6);
    }

    .card {
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(245,255,250,0.98));
      border-radius: 10px;
      padding: 12px;
      margin-bottom: 10px;
      border: 1px solid rgba(180,240,220,0.6);
    }

    h1, h2, h3, h4 { color:#0A3D62 !important; font-weight:800; }
    .stButton>button { background-color:#0A3D62 !important; color:white !important; border-radius:8px !important; padding:8px 12px!important; }
    .small-muted { color:#145A32; font-size:12px; opacity:0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers: load dataset
# ---------------------------
@st.cache_data
def load_dataset():
    if LOCAL_CSV.exists():
        df = pd.read_csv(LOCAL_CSV)
    else:
        df = pd.read_csv(GITHUB_RAW)
    df.columns = [c.strip() for c in df.columns]
    # mapping short codes -> friendly names if present
    mapping = {"drugA": "Amlodipine", "drugB": "Atenolol", "drugC": "ORS-K", "drugX": "Atorvastatin", "drugY": "Losartan"}
    if "Drug" in df.columns:
        df["Drug"] = df["Drug"].map(mapping).fillna(df["Drug"])
    return df

df_full = load_dataset()

# ---------------------------
# Drug details (with severity levels)
# ---------------------------
drug_details = {
    "Amlodipine": {
        "use": "Lowers blood pressure by relaxing blood vessels (calcium channel blocker).",
        "mechanism": "Calcium channel blocker.",
        "side_effects": [("Dizziness","Mild"), ("Edema","Moderate"), ("Headache","Mild")],
        "precautions": "Monitor BP; report severe dizziness or swelling.",
        "dosage": "5–10 mg once daily."
    },
    "Atenolol": {
        "use": "Used for blood pressure control and heart rate reduction (beta-blocker).",
        "mechanism": "Selective β1-blocker.",
        "side_effects": [("Fatigue","Mild"), ("Bradycardia","Moderate"), ("Cold extremities","Mild")],
        "precautions": "Avoid in asthma; monitor heart rate.",
        "dosage": "50 mg once daily."
    },
    "ORS-K": {
        "use": "Oral rehydration / electrolyte replacement for sodium–potassium balance.",
        "mechanism": "Replenishes Na+ and K+.",
        "side_effects": [("Nausea","Mild"), ("Bloating","Mild")],
        "precautions": "Monitor electrolytes in severe cases.",
        "dosage": "As required during dehydration or imbalance."
    },
    "Atorvastatin": {
        "use": "Lowers LDL cholesterol and cardiovascular risk.",
        "mechanism": "HMG-CoA reductase inhibitor (statin).",
        "side_effects": [("Muscle pain","Moderate"), ("Liver enzyme changes","Severe")],
        "precautions": "Check liver enzymes; avoid during pregnancy.",
        "dosage": "10–20 mg in the evening."
    },
    "Losartan": {
        "use": "Used to treat high blood pressure (angiotensin receptor blocker).",
        "mechanism": "Blocks angiotensin II receptors causing vasodilation.",
        "side_effects": [("Dizziness","Mild"), ("Increased potassium","Moderate")],
        "precautions": "Avoid during pregnancy; monitor potassium.",
        "dosage": "25–50 mg once daily."
    }
}

# ---------------------------
# OneHot encoder compatibility
# ---------------------------
def onehot_factory():
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown='ignore')

# ---------------------------
# Model training & persistence
# ---------------------------
ENSEMBLE_PATH = MODEL_DIR / "ensemble_models.joblib"
RF_PATH = MODEL_DIR / "rf_model.joblib"

@st.cache_resource
def build_pipeline(estimator):
    pre = ColumnTransformer([
        ("num", StandardScaler(), ['Age','Na','K']),
        ("cat", onehot_factory(), ['Sex','BP','Cholesterol'])
    ])
    return Pipeline([("pre", pre), ("clf", estimator)])

def train_and_persist_models(df):
    # trains ensemble (LR, KNN, DT, RF, SVM) and RF for chatbot; persists to disk
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

    # persist ensemble
    joblib.dump(trained, ENSEMBLE_PATH)

    # train RF for chatbot separately (can be same pipeline type)
    rf_pipe = build_pipeline(RandomForestClassifier())
    rf_pipe.fit(X, y)
    joblib.dump(rf_pipe, RF_PATH)

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

# Try load; if not present, train
ensemble_models, rf_model = load_models_if_exist()
if ensemble_models is None or rf_model is None:
    with st.spinner("Training models for the first time (this may take a moment)..."):
        ensemble_models, rf_model = train_and_persist_models(df_full)

# Store in session for quick reuse
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
        # probabilities if available
        try:
            proba = pipe.predict_proba(input_df)[0]
            prob_list.append((pipe.classes_, proba))
        except Exception:
            pass
    if not preds:
        return None, None, []
    final = Counter(preds).most_common(1)[0][0]
    # average prob for final
    probs_for_final = []
    for classes, proba in prob_list:
        if final in classes:
            idx = list(classes).index(final)
            probs_for_final.append(proba[idx])
    conf = float(np.mean(probs_for_final))*100.0 if probs_for_final else None
    # aggregated top3
    agg = {}
    for classes, proba in prob_list:
        for cls, p in zip(classes, proba):
            agg.setdefault(cls, []).append(p)
    avg_probs = {cls: float(np.mean(ps)) for cls, ps in agg.items()}
    sorted_avg = sorted(avg_probs.items(), key=lambda x: x[1], reverse=True)
    top3 = sorted_avg[:3]
    return final, conf, top3

# ---------------------------
# Natural language parser for quick predictions
# ---------------------------
def parse_nl_for_features(text):
    """
    Tries to extract Age, Sex, BP, Cholesterol, Na, K from free text.
    Returns dict with keys or None if not parsed.
    """
    text = text.lower()
    res = {"Age": None, "Sex": None, "BP": None, "Cholesterol": None, "Na": None, "K": None}

    # Age: look for numbers near 'age' or standalone number in typical range 10-120
    m = re.search(r'age\s*(?:is\s*)?(\d{1,3})', text)
    if m:
        age = int(m.group(1))
        if 1 <= age <= 120:
            res["Age"] = age
    else:
        # fallback: any number in 1..120 preceded by "year" or "y/o"
        m2 = re.search(r'(\d{1,3})\s*(?:years|yrs|y/o|yo)', text)
        if m2:
            age = int(m2.group(1))
            if 1 <= age <= 120:
                res["Age"] = age

    # Sex
    if re.search(r'\bfemale\b|\bwoman\b|\bgirl\b', text):
        res["Sex"] = "F"
    elif re.search(r'\bmale\b|\bman\b|\bboy\b', text):
        res["Sex"] = "M"

    # BP
    if "high bp" in text or "high blood pressure" in text or re.search(r'\bbp.*high\b', text):
        res["BP"] = "HIGH"
    elif "low bp" in text or "low blood pressure" in text:
        res["BP"] = "LOW"
    elif "normal bp" in text or "bp normal" in text:
        res["BP"] = "NORMAL"
    else:
        # accept words high/low/normal near 'bp' or 'blood pressure'
        m = re.search(r'(high|low|normal).{0,10}(bp|blood pressure)', text)
        if m:
            res["BP"] = m.group(1).upper()

    # Cholesterol
    if "high cholesterol" in text or re.search(r'cholesterol.*high', text):
        res["Cholesterol"] = "HIGH"
    elif "normal cholesterol" in text:
        res["Cholesterol"] = "NORMAL"

    # Na and K: look for floats
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

    # fallback: if we find two floats maybe na and k in order
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

    # If at least Age and either BP or Cholesterol or Na/K found, return partial dict
    any_feature = any([res[k] is not None for k in res.keys()])
    return res if any_feature else None

# ---------------------------
# Simple medically smart responses (expandable)
# ---------------------------
def medical_answer(query):
    q = query.lower()
    if "bp" in q or "blood pressure" in q or "hypertension" in q:
        return ("High blood pressure (hypertension) is commonly treated with drugs such as "
                "Amlodipine (calcium channel blocker), Losartan (ARB) or Atenolol (beta-blocker). "
                "Selection depends on age, comorbidities, electrolytes and other features.")
    if "cholesterol" in q or "statin" in q:
        return ("High cholesterol is often treated with statins like Atorvastatin. "
                "Monitor liver enzymes and muscle symptoms while on statins.")
    m = re.search(r'side effects of ([a-zA-Z0-9\- ]+)', q)
    if m:
        drug = m.group(1).strip().title()
        if drug in drug_details:
            se = ', '.join([f"{s} ({sev})" for s,sev in drug_details[drug]['side_effects']])
            return f"Common side-effects of {drug}: {se}."
        else:
            return f"I don't have detailed side-effect info for {drug} in the database."
    if "interaction" in q or "interact" in q:
        return ("This demo doesn't have a complete drug–drug interaction engine. "
                "Be cautious and consult official resources (drug compendia) for interactions.")
    return ("I can explain drugs (uses, side effects) or run a Random Forest prediction using the current predictor inputs. "
            "Try 'predict' with values (e.g. 'predict age 45 female bp high na 0.7 k 0.05') or click the chatbot's Predict button.")

# ---------------------------
# UI: top heading and tabs (multi-page)
# ---------------------------
st.markdown("<div class='glass'><h1 style='margin:0;'>🛡 Smart Drug Shield</h1>"
            "<p class='small-muted' style='margin:0;'>AI-powered drug prescription classifier — Medical theme</p></div>",
            unsafe_allow_html=True)

tabs = st.tabs(["Predictor", "Chatbot", "Drug Information", "Dashboard", "Admin", "About"])
tab_predictor, tab_chatbot, tab_druginfo, tab_dashboard, tab_admin, tab_about = tabs

# ---------------------------
# Predictor Tab (left: inputs, right: ensemble details)
# ---------------------------
with tab_predictor:
    st.markdown("<div class='glass'><h3>Predictor (Ensemble)</h3>", unsafe_allow_html=True)
    colL, colR = st.columns([2,1])

    with colL:
        st.subheader("Enter patient details")
        age = st.number_input("Age", min_value=1, max_value=120, value=45, key="p_age")
        sex = st.selectbox("Sex", ["F","M"], key="p_sex")
        bp = st.selectbox("Blood Pressure (BP)", ["LOW","NORMAL","HIGH"], key="p_bp")
        chol = st.selectbox("Cholesterol", ["NORMAL","HIGH"], key="p_chol")
        na = st.number_input("Sodium (Na)", format="%.6f", value=0.700000, key="p_na")
        k = st.number_input("Potassium (K)", format="%.6f", value=0.050000, key="p_k")

        # store inputs
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
                    for s,sev in d['side_effects']:
                        st.write(f"- {s} ({sev})")
                    st.write(f"**Precautions:** {d['precautions']}")
                    st.write(f"**Dosage:** {d['dosage']}")

    with colR:
        st.subheader("Ensemble models")
        st.write("Models trained and persisted to disk.")
        # list models
        for name, _ in st.session_state["ensemble_models"]:
            st.write(f"• {name}")
        st.markdown("---")
        st.write("Quick dataset snapshot:")
        st.dataframe(df_full.head(8))

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Chatbot Tab
# ---------------------------
with tab_chatbot:
    st.markdown("<div class='glass'><h3>💬 Smart Drug Assistant (Random Forest)</h3>", unsafe_allow_html=True)
    st.write("Ask medical/drug questions or ask the chatbot to predict using current predictor inputs. Chatbot uses a single Random Forest model for predictions.")

    # chat area and controls
    col_in, col_buttons = st.columns([3,1])
    with col_in:
        user_text = st.text_input("Your message:", key="chat_input_field")
    with col_buttons:
        send = st.button("Send")
        predict_chat = st.button("Predict (Chatbot RF)")

    if send and user_text:
        st.session_state.setdefault("chat_history", [])
        # NL parsing for prediction if user wrote a sentence with numeric features
        parsed = parse_nl_for_features(user_text)
        if parsed:
            # if parsed has at least Age or Na/K set, offer a prediction
            st.session_state["chat_history"].append(("You", user_text))
            st.session_state["chat_history"].append(("Bot", "I parsed some features from your sentence. Do you want me to run a Random Forest prediction? Click 'Predict (Chatbot RF)'."))
            # store parsed as temp
            st.session_state["_nl_parsed"] = parsed
        else:
            # give medically smart answer
            reply = medical_answer(user_text)
            st.session_state["chat_history"].append(("You", user_text))
            st.session_state["chat_history"].append(("Bot", reply))

    if predict_chat:
        st.session_state.setdefault("chat_history", [])
        # priority: use NL parsed if exists, else use predictor inputs
        parsed = st.session_state.get("_nl_parsed")
        if parsed and any(parsed.values()):
            # fill missing features from predictor inputs if available
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
            # use predictor inputs
            combined = st.session_state.get("predictor_inputs")
        if not combined:
            bot_reply = "No predictor inputs available. Please fill values on Predictor tab or type them in natural language."
            st.session_state["chat_history"].append(("You", "Predict request"))
            st.session_state["chat_history"].append(("Bot", bot_reply))
        else:
            # ensure all keys exist and are valid
            try:
                in_df = pd.DataFrame([[int(combined["Age"]), combined["Sex"], combined["BP"], combined["Cholesterol"], float(combined["Na"]), float(combined["K"])]],
                                     columns=['Age','Sex','BP','Cholesterol','Na','K'])
            except Exception as e:
                st.session_state["chat_history"].append(("You", "Predict request"))
                st.session_state["chat_history"].append(("Bot", "Invalid or incomplete inputs for prediction."))
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

    # display chat history
    if st.session_state.get("chat_history"):
        st.markdown("---")
        for who, msg in st.session_state["chat_history"][-12:]:
            if who == "You":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**Bot:** {msg}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Drug Information Tab
# ---------------------------
with tab_druginfo:
    st.markdown("<div class='glass'><h3>💊 Drug Information</h3>", unsafe_allow_html=True)
    for name, info in drug_details.items():
        with st.expander(name):
            st.write(f"**Use:** {info['use']}")
            st.write(f"**Mechanism:** {info['mechanism']}")
            st.write("**Side effects (severity):**")
            for s, sev in info['side_effects']:
                sev_color = {"Mild":"#2E8B57","Moderate":"#FF8C00","Severe":"#C0392B"}.get(sev,"#333")
                st.markdown(f"- **{s}** — <span style='color:{sev_color};'>{sev}</span>", unsafe_allow_html=True)
            st.write(f"**Precautions:** {info['precautions']}")
            st.write(f"**Dosage:** {info['dosage']}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Dashboard Tab (charts & model comparison)
# ---------------------------
with tab_dashboard:
    st.markdown("<div class='glass'><h3>📈 Dashboard</h3>", unsafe_allow_html=True)
    # Dataset summary
    st.write("### Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df_full.shape[0])
    col2.metric("Unique Drugs", df_full['Drug'].nunique() if 'Drug' in df_full.columns else "N/A")
    col3.metric("Features", df_full.shape[1])

    # drug distribution pie chart
    st.write("### Drug distribution")
    if 'Drug' in df_full.columns:
        counts = df_full['Drug'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)
    else:
        st.info("No 'Drug' column in dataset.")

    # Feature histograms
    st.write("### Feature distributions")
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

    # Model comparison: show train time? Using simple cross-validated accuracy would require more time.
    st.write("### Model quick predictions sample (single row)")
    sample = df_full.sample(1, random_state=42)
    st.write("Sample row from dataset:")
    st.dataframe(sample)
    try:
        inp = sample[['Age','Sex','BP','Cholesterol','Na','K']]
        final, conf, top3 = ensemble_predict(st.session_state["ensemble_models"], inp)
        st.write(f"Ensemble sample prediction: **{final}** ({conf:.2f}% confidence)" if conf else f"Ensemble sample prediction: **{final}**")
    except Exception:
        st.info("Could not run sample ensemble prediction.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Admin Tab
# ---------------------------
with tab_admin:
    st.markdown("<div class='glass'><h3>🔧 Admin</h3>", unsafe_allow_html=True)
    st.write("Model persistence & user management (in-memory).")
    # retrain models button
    if st.button("Retrain & Persist Models (overwrite)"):
        with st.spinner("Retraining models..."):
            ensemble_models, rf_model = train_and_persist_models(df_full)
            st.session_state["ensemble_models"] = ensemble_models
            st.session_state["rf_model"] = rf_model
            st.success("Models retrained and saved to disk.")
    st.write("Model files location:", str(MODEL_DIR))
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# About Tab
# ---------------------------
with tab_about:
    st.markdown("<div class='glass'><h3>ℹ️ About & Notes</h3>", unsafe_allow_html=True)
    st.write("""
    **Smart Drug Shield** — demonstration app for educational purposes.
    - Predictor uses a majority-vote ensemble (Logistic Regression, KNN, DecisionTree, RandomForest, SVM).
    - Chatbot uses a single Random Forest model for prediction and confidence.
    - Natural language parsing allows quick inline predictions from chat input.
    - Models are persisted to disk in /mnt/data/smart_drug_models to avoid retraining each run.
    - Not a clinical decision tool. For learning/demos only.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
