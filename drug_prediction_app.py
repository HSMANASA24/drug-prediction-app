import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------------------
# Page Config
# --------------------------------------------------------------
st.set_page_config(page_title="Drug Prescription Classifier", page_icon="💊", layout="centered")

# --------------------------------------------------------------
# Sidebar (define first — required for mode)
# --------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("🌗 Theme Mode", ["Light Mode", "Dark Mode"], key="theme_switch")

    st.markdown("---")
    page = st.radio(
        "📄 Navigate",
        ["Predictor", "Drug Information", "Bulk Prediction", "About"],
        key="page_selector"
    )
    st.markdown("---")
    st.write("Customize your experience.")

# --------------------------------------------------------------
# Base Light Theme CSS
# --------------------------------------------------------------
st.markdown("""
<style>
    .css-10trblm { text-align: center; }

    .stButton>button {
        background-color: #4CAF50; 
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #45A049;
    }

    label { font-weight: 600 !important; }

    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------
# Dark Mode (Sidebar stays LIGHT)
# --------------------------------------------------------------
if mode == "Dark Mode":
    dark_css = """
    <style>
        html, body, [class*="stApp"] {
            background-color: #1e1e1e !important;
            color: white !important;
        }

        /* Keep sidebar light */
        section[data-testid="stSidebar"] * {
            background-color: inherit !important;
            color: inherit !important;
        }

        input, select, textarea {
            background-color: #2d2d2d !important;
            color: white !important;
            border: 1px solid #444 !important;
        }

        div[data-baseweb="select"] div {
            background-color: #2d2d2d !important;
            color: white !important;
        }

        div[data-testid="stFileUploader"] {
            background-color: #2d2d2d !important;
            border-radius: 8px;
        }

        .stButton>button {
            background-color: #444 !important;
            color: white !important;
        }
        .stButton>button:hover {
            background-color: #555 !important;
        }

        div[role="radiogroup"] label {
            color: white !important;
        }

        .stAlert {
            background-color: #2d2d2d !important;
            color: white !important;
        }

        h1, h2, h3, h4, h5 {
            color: #4CAF50 !important;
        }
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)


# --------------------------------------------------------------
# Header
# --------------------------------------------------------------
st.title("💊 Drug Prescription Classifier")
st.markdown("<h4 style='text-align:center; color:#4CAF50;'>AI-Powered Drug Prediction System</h4>", 
            unsafe_allow_html=True)


# --------------------------------------------------------------
# Drug Emoji Icons (not real icons)
# --------------------------------------------------------------
drug_images = {
    "drugA": "💊",
    "drugB": "🩺",
    "drugC": "⚗️",
    "drugX": "🧬",
    "drugY": "🩸"
}


# --------------------------------------------------------------
# Drug Details
# --------------------------------------------------------------
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
        "precautions": "Not safe in pregnancy.",
        "dosage": "1 tablet/day."
    },
    "drugC": {
        "name": "Drug C",
        "use": "For electrolyte imbalance.",
        "mechanism": "Balances Na/K levels.",
        "side_effects": ["Nausea", "Stomach upset"],
        "precautions": "Monitor Na/K regularly.",
        "dosage": "1–2 tablets/day."
    },
    "drugX": {
        "name": "Drug X",
        "use": "Normal BP + high cholesterol.",
        "mechanism": "Reduces cholesterol synthesis.",
        "side_effects": ["Muscle pain", "Weakness"],
        "precautions": "Avoid high-fat diet.",
        "dosage": "Take in evening."
    },
    "drugY": {
        "name": "Drug Y",
        "use": "High BP + high cholesterol.",
        "mechanism": "Reduces BP + cholesterol.",
        "side_effects": ["Dizziness", "Muscle fatigue"],
        "precautions": "Check BP regularly.",
        "dosage": "1 tablet/day."
    }
}


# --------------------------------------------------------------
# Sample Dataset
# --------------------------------------------------------------
def load_sample_df():
    return pd.DataFrame([
        [23, 'F', 'HIGH', 'HIGH', 0.792535, 0.031258, 'drugY'],
        [47, 'M', 'LOW', 'HIGH', 0.739309, 0.056468, 'drugC'],
        [47, 'M', 'LOW', 'HIGH', 0.697269, 0.068944, 'drugC'],
        [28, 'F', 'NORMAL', 'HIGH', 0.563682, 0.072289, 'drugX'],
        [61, 'F', 'LOW', 'HIGH', 0.559294, 0.030998, 'drugY'],
        [45, 'M', 'NORMAL', 'NORMAL', 0.700000, 0.050000, 'drugA']
    ], columns=['Age','Sex','BP','Cholesterol','Na','K','Drug'])


# --------------------------------------------------------------
# Model Training
# --------------------------------------------------------------
@st.cache_resource
def train_model(df):
    X = df[['Age','Sex','BP','Cholesterol','Na','K']]
    y = df['Drug']

    numeric = ['Age','Na','K']
    categorical = ['Sex','BP','Cholesterol']

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
    ])

    model = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, multi_class="multinomial"))
    ])

    model.fit(X, y)
    return model


# --------------------------------------------------------------
# PAGE 1 → Predictor
# --------------------------------------------------------------
if page == "Predictor":
    st.subheader("📂 Dataset")

    uploaded = st.file_uploader("Upload CSV (optional)", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("Dataset uploaded!")
    else:
        df = load_sample_df()
        st.info("Using sample dataset.")

    model = train_model(df)

    st.subheader("🧪 Enter Patient Details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, 45)
        sex = st.selectbox("Sex", ["F", "M"])
        bp = st.selectbox("BP", ["LOW", "NORMAL", "HIGH"])
    with col2:
        cholesterol = st.selectbox("Cholesterol", ["HIGH", "NORMAL"])
        na = st.number_input("Sodium (Na)", value=0.70, format="%.4f")
        k = st.number_input("Potassium (K)", value=0.05, format="%.4f")

    if st.button("🔍 Predict Drug"):
        input_df = pd.DataFrame([[age, sex, bp, cholesterol, na, k]],
                                columns=['Age','Sex','BP','Cholesterol','Na','K'])
        pred = model.predict(input_df)[0]
        st.success(f"💊 Predicted Drug: **{pred}**")


# --------------------------------------------------------------
# PAGE 2 → Drug Information
# --------------------------------------------------------------
if page == "Drug Information":

    st.title("📘 Drug Information")

    drug_choice = st.selectbox("Select Drug", list(drug_details.keys()))
    info = drug_details[drug_choice]

    icon = drug_images[drug_choice]
    st.markdown(f"## {icon} {info['name']}")

    st.markdown("### 🧭 Use")
    st.write(info["use"])

    st.markdown("### 🔬 Mechanism")
    st.write(info["mechanism"])

    st.markdown("### ⚠️ Side Effects")
    for s in info["side_effects"]:
        st.markdown(f"- {s}")

    st.markdown("### 🔒 Precautions")
    st.write(info["precautions"])

    st.markdown("### 💉 Dosage")
    st.write(info["dosage"])


# --------------------------------------------------------------
# PAGE 3 → Bulk Prediction
# --------------------------------------------------------------
if page == "Bulk Prediction":

    st.title("📂 Bulk Drug Prediction")
    csv_file = st.file_uploader("Upload CSV for bulk prediction", type=['csv'])

    if csv_file:
        bulk_df = pd.read_csv(csv_file)
        st.success("File uploaded!")
        st.write("### Preview")
        st.dataframe(bulk_df.head())

        required = ['Age','Sex','BP','Cholesterol','Na','K']
        missing = [c for c in required if c not in bulk_df.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            model = train_model(load_sample_df())
            bulk_df["Predicted_Drug"] = model.predict(bulk_df[required])

            st.success("Predictions complete!")
            st.dataframe(bulk_df.head())

            st.download_button(
                "📥 Download Results CSV",
                data=bulk_df.to_csv(index=False).encode("utf-8"),
                file_name="drug_predictions.csv",
                mime="text/csv"
            )
    else:
        st.info("Upload a CSV file to start.")


# --------------------------------------------------------------
# PAGE 4 → About
# --------------------------------------------------------------
if page == "About":

    st.title("ℹ️ About This App")

    st.markdown("""
    ### 💊 Drug Prescription Classifier  
    This AI-powered system predicts drug types using patient information such as:

    - Age  
    - Sex  
    - Blood Pressure  
    - Cholesterol  
    - Sodium  
    - Potassium  

    ### 🧠 Features  
    - Single prediction  
    - Drug Information  
    - Bulk prediction  
    - Dark/Light mode  

    ### 🏗️ Built With  
    - Streamlit  
    - Scikit-Learn  
    - Pandas  
    - Python  

    ### ❤️ Developer  
    Created for healthcare learning and prediction insights.
    """)

# --------------------------------------------------------------
# Footer
# --------------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ in Streamlit • Drug Predictor App")
