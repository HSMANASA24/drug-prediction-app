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
# Custom CSS Theme Enhancements
# --------------------------------------------------------------
st.markdown("""
    <style>
        /* Center the title */
        .css-10trblm {
            text-align: center;
        }

        /* Buttons */
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
            color: white;
        }

        /* Labels bold */
        label {
            font-weight: 600 !important;
        }

        /* Success message styling */
        .stSuccess {
            border-left: 5px solid #4CAF50;
            padding-left: 10px;
        }

        /* Improve spacing */
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)


# Apply theme based on selection
if mode == "Dark Mode":
    dark_css = """
    <style>
        body {
            background-color: #1e1e1e;
            color: white;
        }
        .css-18e3th9, .css-1pahdxg {
            background-color: #1e1e1e !important;
            color: white !important;
        }
        .stButton>button {
            background-color: #444444 !important;
            color: white !important;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #555555 !important;
        }
        .css-10trblm {
            color: #4CAF50 !important;
        }
        .stSuccess {
            background-color: #2d2d2d !important;
            color: white !important;
        }
        .stTextInput>div>div>input {
            background-color: #2d2d2d !important;
            color: white !important;
        }
        .stSelectbox>div>div {
            background-color: #2d2d2d !important;
            color: white !important;
        }
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)
else:
    light_css = """
    <style>
        body {
            background-color: white;
            color: black;
        }
    </style>
    """
    st.markdown(light_css, unsafe_allow_html=True)


# --------------------------------------------------------------
# App Header
# --------------------------------------------------------------
st.title("💊 Drug Prescription Classifier")
st.markdown("<h4 style='text-align:center; color:#4CAF50;'>AI-Powered Drug Prediction System</h4>", unsafe_allow_html=True)

st.write("Upload your dataset or use the built-in sample data to predict drug types.")

with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("🌗 Theme Mode", ["Light Mode", "Dark Mode"], key="theme_switch")
    st.markdown("---")
    st.write("Customize the look and feel of the app.")



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
# Model Training Function
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
# File Upload Section
# --------------------------------------------------------------
uploaded = st.file_uploader("📂 Upload CSV Dataset (optional)", type=['csv'])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("Dataset uploaded successfully!")
else:
    df = load_sample_df()
    st.info("Using built-in sample dataset.")

model = train_model(df)

# --------------------------------------------------------------
# Prediction Form UI
# --------------------------------------------------------------
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


# --------------------------------------------------------------
# Prediction Button
# --------------------------------------------------------------
if st.button("🔍 Predict Drug Type"):
    input_data = pd.DataFrame([[age, sex, bp, cholesterol, na, k]],
                              columns=['Age','Sex','BP','Cholesterol','Na','K'])
    prediction = model.predict(input_data)[0]
    st.success(f"💊 **Predicted Drug:** {prediction}")

# --------------------------------------------------------------
# Footer
# --------------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit • Drug Predictor App")
