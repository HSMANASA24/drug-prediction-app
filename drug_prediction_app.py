import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(page_title="Drug Prescription Classifier", page_icon="💊", layout="centered")

# ----------------------------------------------------
# Sidebar First (for mode + navigation)
# ----------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("🌗 Theme Mode", ["Light Mode", "Dark Mode"], key="theme_switch")

    st.markdown("---")
    page = st.radio(
        "📄 Navigate",
        ["Predictor", "Drug Information", "Bulk Prediction", "About"],
        key="nav_page"
    )
    st.markdown("---")
    st.write("Use the sidebar to switch pages or themes.")

# ----------------------------------------------------
# Gradient Glassmorphism CSS (no f-strings!)
# ----------------------------------------------------
base_css = """
<style>

body {
  background: linear-gradient(135deg, rgba(31,58,147,1) 0%, rgba(110,43,168,1) 40%, rgba(245,69,145,1) 100%);
  background-attachment: fixed;
}

/* Keep sidebar light */
section[data-testid="stSidebar"] {
  background-color: rgba(255,255,255,0.94) !important;
  color: #111 !important;
}

/* Glass card */
.glass-panel {
  backdrop-filter: blur(10px) saturate(160%);
  -webkit-backdrop-filter: blur(10px) saturate(160%);
  background: rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 18px;
  border: 1px solid rgba(255,255,255,0.30);
  box-shadow: 0 10px 40px rgba(0,0,0,0.40);
  margin-bottom: 22px;
}

/* Slightly darker glass */
.glass-panel-2 {
  backdrop-filter: blur(8px);
  background: rgba(0,0,0,0.20);
  border-radius: 14px;
  padding: 14px;
  border: 1px solid rgba(255,255,255,0.18);
  margin-bottom: 18px;
}

/* Inputs */
input, textarea, select {
  background: rgba(255,255,255,0.15) !important;
  color: white !important;
  border-radius: 8px !important;
  border: 1px solid rgba(255,255,255,0.25) !important;
}

/* File uploader */
div[data-testid="stFileUploader"] {
  background: rgba(255,255,255,0.12) !important;
  padding: 10px;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.20);
}

/* Buttons */
.stButton>button {
  background: rgba(255,255,255,0.10) !important;
  color: #fff !important;
  border: 1px solid rgba(255,255,255,0.20);
  padding: 10px 18px;
  border-radius: 12px;
  box-shadow: 0 6px 25px rgba(0,0,0,0.35);
}
.stButton>button:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 30px rgba(0,0,0,0.55);
}

h1, h2, h3, h4 {
  color: #eaf5ff !important;
}

table, th, td {
  color: white !important;
}

</style>
"""
st.markdown(base_css, unsafe_allow_html=True)

# ----------------------------------------------------
# Dark Mode CSS
# ----------------------------------------------------
if mode == "Dark Mode":
    dark_css = """
    <style>
    .glass-panel {
      background: rgba(0,0,0,0.35) !important;
      border: 1px solid rgba(255,255,255,0.10);
    }
    .glass-panel-2 {
      background: rgba(0,0,0,0.45) !important;
      border: 1px solid rgba(255,255,255,0.08);
    }
    input, textarea, select {
      background: rgba(255,255,255,0.08) !important;
      color: #eaf5ff !important;
    }
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)

# ----------------------------------------------------
# Header Glass Panel
# ----------------------------------------------------
st.markdown("""
<div class="glass-panel">
  <h1>💊 Drug Prescription Classifier</h1>
  <p style="color:rgba(240,245,255,0.9);">AI-based prediction system using Glassmorphism UI</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Emoji Icons
# ----------------------------------------------------
drug_images = {
    "drugA": "💊",
    "drugB": "🩺",
    "drugC": "⚗️",
    "drugX": "🧬",
    "drugY": "🩸"
}

# ----------------------------------------------------
# Drug Details
# ----------------------------------------------------
drug_details = {
    "drugA": {
        "name": "Drug A",
        "use": "Used for normal BP and cholesterol.",
        "mechanism": "Improves circulatory health.",
        "side_effects": ["Headache", "Dry mouth", "Dizziness"],
        "precautions": "Avoid alcohol.",
        "dosage": "1 tablet daily."
    },
    "drugB": {
        "name": "Drug B",
        "use": "Used for high blood pressure.",
        "mechanism": "Relaxes blood vessels.",
        "side_effects": ["Low BP", "Fatigue"],
        "precautions": "Not for pregnancy.",
        "dosage": "1 per day."
    },
    "drugC": {
        "name": "Drug C",
        "use": "Treats electrolyte imbalance.",
        "mechanism": "Balances Na/K levels.",
        "side_effects": ["Nausea", "Upset stomach"],
        "precautions": "Monitor Na/K.",
        "dosage": "1–2 daily."
    },
    "drugX": {
        "name": "Drug X",
        "use": "Normal BP + high cholesterol.",
        "mechanism": "Reduces cholesterol.",
        "side_effects": ["Muscle pain", "Weakness"],
        "precautions": "Avoid high-fat food.",
        "dosage": "Evening dose."
    },
    "drugY": {
        "name": "Drug Y",
        "use": "High BP + high cholesterol.",
        "mechanism": "BP reduction + cholesterol control.",
        "side_effects": ["Dizziness", "Muscle fatigue"],
        "precautions": "Check BP regularly.",
        "dosage": "1 daily."
    }
}

# ----------------------------------------------------
# Sample Dataset
# ----------------------------------------------------
def load_sample_df():
    return pd.DataFrame([
        [23, 'F', 'HIGH', 'HIGH', 0.792535, 0.031258, 'drugY'],
        [47, 'M', 'LOW', 'HIGH', 0.739309, 0.056468, 'drugC'],
        [47, 'M', 'LOW', 'HIGH', 0.697269, 0.068944, 'drugC'],
        [28, 'F', 'NORMAL', 'HIGH', 0.563682, 0.072289, 'drugX'],
        [61, 'F', 'LOW', 'HIGH', 0.559294, 0.030998, 'drugY'],
        [45, 'M', 'NORMAL', 'NORMAL', 0.7, 0.05, 'drugA']
    ], columns=['Age','Sex','BP','Cholesterol','Na','K','Drug'])

# ----------------------------------------------------
# Train Model
# ----------------------------------------------------
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

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, multi_class='multinomial'))
    ])

    pipeline.fit(X, y)
    return pipeline

# ----------------------------------------------------
# PAGE: Predictor
# ----------------------------------------------------
if page == "Predictor":
    st.markdown('<div class="glass-panel"><h3>Single Prediction</h3></div>', unsafe_allow_html=True)

    # Upload or sample
    st.markdown('<div class="glass-panel-2">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    df = pd.read_csv(uploaded) if uploaded else load_sample_df()
    model = train_model(df)

    # Input Glass Card
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("🧪 Enter Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 45)
        sex = st.selectbox("Sex", ["F", "M"])
        bp = st.selectbox("Blood Pressure", ["LOW", "NORMAL", "HIGH"])

    with col2:
        cholesterol = st.selectbox("Cholesterol", ["HIGH", "NORMAL"])
        na = st.number_input("Sodium (Na)", value=0.7, format="%.4f")
        k = st.number_input("Potassium (K)", value=0.05, format="%.4f")

    if st.button("🔍 Predict Drug"):
        data = pd.DataFrame([[age, sex, bp, cholesterol, na, k]],
                            columns=['Age','Sex','BP','Cholesterol','Na','K'])
        pred = model.predict(data)[0]
        st.success(f"💊 Predicted Drug: **{pred}**")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# PAGE: Drug Information
# ----------------------------------------------------
if page == "Drug Information":
    st.markdown('<div class="glass-panel"><h3>Drug Information</h3></div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-panel-2">', unsafe_allow_html=True)
    choice = st.selectbox("Select Drug", list(drug_details.keys()))
    st.markdown('</div>', unsafe_allow_html=True)

    info = drug_details[choice]
    icon = drug_images[choice]

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown(f"<h2>{icon} {info['name']}</h2>", unsafe_allow_html=True)

    st.markdown("### 🧭 Use")
    st.write(info["use"])

    st.markdown("### 🔬 Mechanism")
    st.write(info["mechanism"])

    st.markdown("### ⚠ Side Effects")
    for s in info["side_effects"]:
        st.markdown(f"- {s}")

    st.markdown("### 🔒 Precautions")
    st.write(info["precautions"])

    st.markdown("### 💉 Dosage")
    st.write(info["dosage"])
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# PAGE: Bulk Prediction
# ----------------------------------------------------
if page == "Bulk Prediction":
    st.markdown('<div class="glass-panel"><h3>Bulk Prediction</h3></div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-panel-2">', unsafe_allow_html=True)
    csv_file = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if csv_file:
        bulk_df = pd.read_csv(csv_file)
        st.dataframe(bulk_df.head())

        required = ['Age','Sex','BP','Cholesterol','Na','K']
        missing = [c for c in required if c not in bulk_df.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            model = train_model(load_sample_df())
            bulk_df["Predicted_Drug"] = model.predict(bulk_df[required])
            st.success("Prediction Complete!")
            st.dataframe(bulk_df)

            st.download_button(
                "📥 Download Output CSV",
                bulk_df.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv"
            )
    else:
        st.info("Please upload a CSV file to begin.")

# ----------------------------------------------------
# PAGE: About
# ----------------------------------------------------
if page == "About":
    st.markdown('<div class="glass-panel"><h3>About This App</h3></div>', unsafe_allow_html=True)

    st.markdown("""
    ### 💊 Drug Prescription Classifier  
    This app predicts drug type based on patient information.

    Built with:
    - Streamlit  
    - Scikit-Learn  
    - Pandas  
    - Gradient Glassmorphism UI  

    ### ✨ Features
    - Single Prediction  
    - Drug Details  
    - Bulk Prediction  
    - Dark/Light Theme  
    - Fully Responsive  
    """)

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.caption("Built with ❤️ using Streamlit + Gradient Glassmorphism Theme")
