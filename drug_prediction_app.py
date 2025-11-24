import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------------------
# Streamlit Page Settings
# --------------------------------------------------------------
st.set_page_config(
    page_title="Drug Prescription Classifier",
    page_icon="💊",
    layout="centered"
)

# --------------------------------------------------------------
# Sidebar (First → ensures 'mode' + 'page' defined early)
# --------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("🌗 Theme Mode", ["Light Mode", "Dark Mode"], key="theme_switch")
    
    st.markdown("---")
    page = st.radio("📄 Navigate", ["Predictor", "Drug Information"], key="page_selector")
    st.markdown("---")
    st.write("Customize the look and browse pages.")

# --------------------------------------------------------------
# Base CSS (Light Mode Styling)
# --------------------------------------------------------------
st.markdown("""
    <style>
        .css-10trblm { text-align: center; }

        .stButton>button {
            background-color: #4CAF50; color: white;
            border-radius: 8px; padding: 10px 20px;
            border: none; font-size: 18px;
        }
        .stButton>button:hover {
            background-color: #45A049; color: white;
        }

        label { font-weight: 600 !important; }

        .stSuccess {
            border-left: 5px solid #4CAF50;
            padding-left: 10px;
        }

        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------
# DARK MODE OVERRIDE
# --------------------------------------------------------------
if mode == "Dark Mode":
    dark_css = """
    <style>
        /* MAIN PAGE ONLY */
        html, body, [class*="stApp"] {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
        }

        /* Keep sidebar light → do not modify it */

        /* Inputs, selects, textareas */
        input, select, textarea {
            background-color: #2d2d2d !important;
            color: white !important;
            border: 1px solid #444444 !important;
        }

        /* Selectbox internal */
        div[data-baseweb="select"] div {
            background-color: #2d2d2d !important;
            color: white !important;
        }

        /* File uploader */
        div[data-testid="stFileUploader"] {
            background-color: #2d2d2d !important;
            border-radius: 8px;
        }

        /* Buttons */
        .stButton>button {
            background-color: #444444 !important;
            color: white !important;
        }
        .stButton>button:hover {
            background-color: #555555 !important;
        }

        /* Radio/checkbox text */
        div[role="radiogroup"] label {
            color: white !important;
        }

        /* Success boxes */
        .stAlert {
            background-color: #2d2d2d !important;
            color: white !important;
        }

        /* Titles */
        h1, h2, h3, h4 {
            color: #4CAF50 !important;
        }
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)

# --------------------------------------------------------------
# App Header (Visible for both pages)
# --------------------------------------------------------------
st.title("💊 Drug Prescription Classifier")
st.markdown("<h4 style='text-align:center; color:#4CAF50;'>AI-Powered Drug Prediction System</h4>", unsafe_allow_html=True)


# --------------------------------------------------------------
# Drug Information Data
# --------------------------------------------------------------
drug_details = {
    "drugA": {
        "name": "Drug A",
        "use": "Used for normal BP and normal cholesterol levels.",
        "mechanism": "Supports circulatory function.",
        "side_effects": ["Headache", "Dry mouth", "Dizziness"],
        "precautions": "Avoid alcohol; stay hydrated.",
        "dosage": "1 tablet daily after meals."
    },
    "drugB": {
        "name": "Drug B",
        "use": "For high blood pressure patients.",
        "mechanism": "Relaxes blood vessels to reduce BP.",
        "side_effects": ["Low BP", "Fatigue"],
        "precautions": "Not safe in pregnancy.",
        "dosage": "1 tablet/day or as prescribed."
    },
    "drugC": {
        "name": "Drug C",
        "use": "For abnormal sodium or potassium levels.",
        "mechanism": "Balances electrolyte concentration.",
        "side_effects": ["Nausea", "Stomach upset"],
        "precautions": "Monitor Na/K levels regularly.",
        "dosage": "1–2 tablets per day."
    },
    "drugX": {
        "name": "Drug X",
        "use": "For normal BP + high cholesterol.",
        "mechanism": "Reduces cholesterol synthesis.",
        "side_effects": ["Muscle pain", "Weakness"],
        "precautions": "Avoid high-fat foods.",
        "dosage": "Take in the evening."
    },
    "drugY": {
        "name": "Drug Y",
        "use": "For high BP + high cholesterol.",
        "mechanism": "Slows cholesterol production + reduces BP.",
        "side_effects": ["Dizziness", "Muscle fatigue"],
        "precautions": "Check BP regularly.",
        "dosage": "One tablet per day."
    }
}

# --------------------------------------------------------------
# SAMPLE DATASET
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
# MODEL TRAINING FUNCTION
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
# PAGE 1 → PREDICTOR PAGE
# --------------------------------------------------------------
if page == "Predictor":

    st.write("Upload your dataset or use the sample data below.")

    uploaded = st.file_uploader("📂 Upload CSV Dataset (optional)", type=['csv'])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("Dataset uploaded successfully!")
    else:
        df = load_sample_df()
        st.info("Using built-in sample dataset.")

    model = train_model(df)

    st.subheader("🧪 Enter Patient Details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", ["F", "M"])
        bp = st.selectbox("Blood Pressure (BP)", ["LOW", "NORMAL", "HIGH"])

    with col2:
        cholesterol = st.selectbox("Cholesterol", ["HIGH", "NORMAL"])
        na = st.number_input("Sodium (Na)", value=0.70, format="%.4f")
        k = st.number_input("Potassium (K)", value=0.05, format="%.4f")

    if st.button("🔍 Predict Drug Type"):
        input_df = pd.DataFrame([[age, sex, bp, cholesterol, na, k]],
                                columns=['Age','Sex','BP','Cholesterol','Na','K'])
        prediction = model.predict(input_df)[0]
        st.success(f"💊 **Predicted Drug:** {prediction}")


# --------------------------------------------------------------
# PAGE 2 → DRUG INFORMATION PAGE
# --------------------------------------------------------------
if page == "Drug Information":

    st.title("📘 Drug Information Guide")
    st.write("Select any drug to see detailed medical information.")

    drug_choice = st.selectbox("Choose Drug", list(drug_details.keys()))

    info = drug_details[drug_choice]

    st.markdown(f"## 💊 {info['name']}")
    st.markdown(f"### 🧭 Use")
    st.write(info["use"])

    st.markdown(f"### 🔬 Mechanism")
    st.write(info["mechanism"])

    st.markdown("### ⚠️ Side Effects")
    for item in info["side_effects"]:
        st.markdown(f"- {item}")

    st.markdown("### 🔒 Precautions")
    st.write(info["precautions"])

    st.markdown("### 💉 Dosage")
    st.write(info["dosage"])


# --------------------------------------------------------------
# Footer
# --------------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit • Drug Predictor App")
