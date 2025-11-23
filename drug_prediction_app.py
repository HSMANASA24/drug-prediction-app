import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------------------
# Streamlit Page Settings
# --------------------------------------------------------------
st.set_page_config(page_title="Drug Prescription Classifier", layout="centered")

st.title("💊 Drug Prescription Classifier")
st.write("Predict drug type using patient health information.")

# --------------------------------------------------------------
# Sample Dataset (Used if no CSV uploaded)
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
# Model Training Function (Cached)
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
# Upload Dataset
# --------------------------------------------------------------
uploaded = st.file_uploader("📂 Upload CSV Dataset (optional)", type=['csv'])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("Dataset uploaded successfully!")
else:
    df = load_sample_df()
    st.info("Using built-in sample dataset.")

# Train model
model = train_model(df)

# --------------------------------------------------------------
# Prediction Form
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

if st.button("🔍 Predict Drug Type"):
    input_data = pd.DataFrame([[age, sex, bp, cholesterol, na, k]],
                              columns=['Age','Sex','BP','Cholesterol','Na','K'])
    prediction = model.predict(input_data)[0]
    st.success(f"💊 **Predicted Drug:** {prediction}")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit • Ideal for Cloud Deployment")
