import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Drug Prescription Classifier", page_icon="💊", layout="centered")

# ---------------------------
# Sidebar (define first)
# ---------------------------
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
    st.write("Tip: Toggle theme or switch pages here.")

# ---------------------------
# Glassmorphism + Gradient CSS
# ---------------------------
# Gradient background for the whole app + glass panels for main content
base_css = f"""
<style>
/* Page background gradient */
body {{
  background: linear-gradient(135deg, rgba(31,58,147,1) 0%, rgba(110,43,168,1) 40%, rgba(245,69,145,1) 100%);
  background-attachment: fixed;
}}

/* Keep sidebar light to preserve contrast */
section[data-testid="stSidebar"] {{
  background: rgba(255,255,255,0.96) !important;
  color: #111 !important;
}}

/* Glass panel style for main content containers */
.glass-panel {{
  backdrop-filter: blur(8px) saturate(140%);
  -webkit-backdrop-filter: blur(8px) saturate(140%);
  background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.06));
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.14);
  box-shadow: 0 8px 30px rgba(2,6,23,0.6);
  padding: 18px;
  margin-bottom: 18px;
}

/* Card header accent */
.glass-accent {{
  border-left: 6px solid rgba(255,255,255,0.18);
  padding-left: 12px;
  margin-bottom: 12px;
}}

/* Secondary, slightly darker glass card (for contrast) */
.glass-panel-2 {{
  backdrop-filter: blur(6px);
  background: linear-gradient(135deg, rgba(0,0,0,0.12), rgba(255,255,255,0.02));
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.06);
  padding: 14px;
  margin-bottom: 14px;
}}

/* Headings */
h1,h2,h3,h4 {{
  color: #eaf5ff;
}}

/* Make buttons look elevated and neon-accented */
.stButton>button {{
  background: rgba(255,255,255,0.08) !important;
  color: #ffffff !important;
  border-radius: 10px;
  padding: 10px 16px;
  border: 1px solid rgba(255,255,255,0.12);
  box-shadow: 0 6px 18px rgba(0,0,0,0.35);
}}
.stButton>button:hover {{
  transform: translateY(-2px);
  box-shadow: 0 10px 28px rgba(0,0,0,0.5);
}}

/* Inputs & selects inside the main area (glass-style) */
input, textarea, select {
  background: rgba(255,255,255,0.04) !important;
  color: #ffffff !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 8px !important;
}

/* File uploader card */
div[data-testid="stFileUploader"] {{
  background: rgba(255,255,255,0.03) !important;
  border-radius: 12px !important;
  padding: 8px !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
}}

/* Keep sidebar controls readable in light mode */
section[data-testid="stSidebar"] .stRadio > div,
section[data-testid="stSidebar"] .stSelectbox > div {{
  color: #111 !important;
}}

/* Footer caption style */
footer {{
  visibility: hidden;
}}

/* Ensure tables in dark mode look clean */
table, th, td {{
  color: #eaf5ff !important;
}}

/* Responsive tweaks */
@media (max-width: 600px) {{
  .glass-panel {{ padding: 12px; border-radius: 12px; }}
}}
</style>
"""
st.markdown(base_css, unsafe_allow_html=True)

# ---------------------------
# Dark Mode tweaks (main area only)
# ---------------------------
if mode == "Dark Mode":
    dark_extra = """
    <style>
    /* Intensify glass and reduce brightness for dark mode main area */
    .glass-panel {
      background: linear-gradient(135deg, rgba(10,10,10,0.36), rgba(255,255,255,0.02));
      border: 1px solid rgba(255,255,255,0.05);
    }
    .glass-panel-2 {
      background: linear-gradient(135deg, rgba(5,5,5,0.42), rgba(255,255,255,0.01));
      border: 1px solid rgba(255,255,255,0.04);
    }
    /* Inputs darker */
    input, textarea, select {
      background: rgba(255,255,255,0.02) !important;
      color: #eaf5ff !important;
      border: 1px solid rgba(255,255,255,0.06) !important;
    }
    </style>
    """
    st.markdown(dark_extra, unsafe_allow_html=True)

# ---------------------------
# Header (inside a glass panel)
# ---------------------------
st.markdown('<div class="glass-panel glass-accent"><h1 style="margin:6px 0;">💊 Drug Prescription Classifier</h1>'
            '<p style="color:rgba(234,245,255,0.9); margin-top:6px;">AI-Powered Drug Prediction System — enter patient info or upload CSV.</p></div>',
            unsafe_allow_html=True)

# ---------------------------
# Emoji icons (no real icons)
# ---------------------------
drug_images = {
    "drugA": "💊",
    "drugB": "🩺",
    "drugC": "⚗️",
    "drugX": "🧬",
    "drugY": "🩸"
}

# ---------------------------
# Drug details
# ---------------------------
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
        "precautions": "Not suitable for pregnancy.",
        "dosage": "1 tablet/day."
    },
    "drugC": {
        "name": "Drug C",
        "use": "For electrolyte imbalance.",
        "mechanism": "Balances Na/K levels.",
        "side_effects": ["Nausea", "Stomach upset"],
        "precautions": "Monitor Na/K levels regularly.",
        "dosage": "1–2 tablets/day."
    },
    "drugX": {
        "name": "Drug X",
        "use": "Normal BP + high cholesterol.",
        "mechanism": "Reduces cholesterol synthesis.",
        "side_effects": ["Muscle pain", "Weakness"],
        "precautions": "Avoid high-fat diet.",
        "dosage": "Take in the evening."
    },
    "drugY": {
        "name": "Drug Y",
        "use": "High BP + high cholesterol.",
        "mechanism": "Lowers cholesterol synthesis and reduces BP.",
        "side_effects": ["Dizziness", "Muscle fatigue"],
        "precautions": "Check BP regularly.",
        "dosage": "1 tablet/day."
    }
}

# ---------------------------
# Sample dataset
# ---------------------------
def load_sample_df():
    return pd.DataFrame([
        [23, 'F', 'HIGH', 'HIGH', 0.792535, 0.031258, 'drugY'],
        [47, 'M', 'LOW', 'HIGH', 0.739309, 0.056468, 'drugC'],
        [47, 'M', 'LOW', 'HIGH', 0.697269, 0.068944, 'drugC'],
        [28, 'F', 'NORMAL', 'HIGH', 0.563682, 0.072289, 'drugX'],
        [61, 'F', 'LOW', 'HIGH', 0.559294, 0.030998, 'drugY'],
        [45, 'M', 'NORMAL', 'NORMAL', 0.700000, 0.050000, 'drugA']
    ], columns=['Age','Sex','BP','Cholesterol','Na','K','Drug'])

# ---------------------------
# Model training
# ---------------------------
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

# ---------------------------
# PREDICTOR PAGE
# ---------------------------
if page == "Predictor":
    st.markdown('<div class="glass-panel"><h3 style="margin:0 0 6px 0;">Predictor</h3>'
                '<p style="margin:0 0 6px 0; color:rgba(234,245,255,0.9);">Single patient prediction</p></div>',
                unsafe_allow_html=True)

    # data upload (inside a second glass panel)
    st.markdown('<div class="glass-panel-2">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV (optional)", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("Dataset uploaded!")
    else:
        df = load_sample_df()
        st.info("Using sample dataset.")
    st.markdown('</div>', unsafe_allow_html=True)

    model = train_model(df)

    # Input form in a glass-like block
    with st.container():
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
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

        if st.button("🔍 Predict Drug"):
            input_df = pd.DataFrame([[age, sex, bp, cholesterol, na, k]],
                                    columns=['Age','Sex','BP','Cholesterol','Na','K'])
            prediction = model.predict(input_df)[0]
            st.success(f"💊 Predicted Drug: **{prediction}**")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# DRUG INFORMATION PAGE
# ---------------------------
if page == "Drug Information":
    st.markdown('<div class="glass-panel"><h3 style="margin:0 0 6px 0;">Drug Information</h3>'
                '<p style="margin:0 0 6px 0; color:rgba(234,245,255,0.9);">Descriptions, side effects, dosage</p></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="glass-panel-2">', unsafe_allow_html=True)
    drug_choice = st.selectbox("Select Drug", list(drug_details.keys()))
    st.markdown('</div>', unsafe_allow_html=True)

    info = drug_details[drug_choice]
    icon = drug_images[drug_choice]

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown(f"<h2 style='margin:4px 0 8px 0;'>{icon} {info['name']}</h2>", unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# BULK PREDICTION PAGE
# ---------------------------
if page == "Bulk Prediction":
    st.markdown('<div class="glass-panel"><h3 style="margin:0 0 6px 0;">Bulk Prediction</h3>'
                '<p style="margin:0 0 6px 0; color:rgba(234,245,255,0.9);">Upload CSV and download predictions</p></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="glass-panel-2">', unsafe_allow_html=True)
    csv_file = st.file_uploader("Upload CSV for bulk prediction", type=['csv'], key="bulk_csv")
    st.markdown('</div>', unsafe_allow_html=True)

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
            # Use existing train function (sample dataset used to train here)
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
        st.info("Upload a CSV file (columns: Age, Sex, BP, Cholesterol, Na, K).")

# ---------------------------
# ABOUT PAGE
# ---------------------------
if page == "About":
    st.markdown('<div class="glass-panel"><h3 style="margin:0 0 6px 0;">About</h3>'
                '<p style="margin:0 0 6px 0; color:rgba(234,245,255,0.9);">What this app does & technologies used</p></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="glass-panel-2">', unsafe_allow_html=True)
    st.markdown("""
    ### 💊 Drug Prescription Classifier
    Predicts drug type using patient info (Age, Sex, BP, Cholesterol, Na, K) with a small demo model.
    """)
    st.markdown("#### Features included")
    st.markdown("- Single prediction\n- Drug information\n- Bulk CSV prediction\n- Light/Dark theme\n- Glassmorphism UI")
    st.markdown("#### Built with: Streamlit, Scikit-Learn, Pandas, Python")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown('<div style="margin-top:18px;"></div>', unsafe_allow_html=True)
st.caption("Built with ❤️ in Streamlit • Glassmorphism Gradient Theme")
