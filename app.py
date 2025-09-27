import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os # Added os import for path handling

# =============================================================================
# Streamlit UI Configuration and Model Loading
# =============================================================================

# Configuration (must be at the top)
st.set_page_config(
    page_title="Heart Disease Predictor (Optimized SVM)",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Define the file path for the exported model
# NOTE: This path assumes the 'models' directory is relative to the directory 
# where 'ui/app.py' is run, or that the path is absolute in deployment.
MODEL_PATH = 'models/final_model.pkl'

# Define the 12 features selected during the Feature Selection phase
# CRITICAL: These names MUST match the feature names used during model training!
FEATURE_NAMES = [
    'cp', 'thalach', 'oldpeak', 'thal', 'ca', 'sex', 'chol', 'trestbps',
    'exang', 'fbs', 'restecg', 'age'
]

# Function to load the model artifact. Caches the model for speed.
@st.cache_resource
def load_model(path):
    """Loads the final model pipeline artifact."""
    try:
        # Load the model using joblib
        model_pipeline = joblib.load(path)
        return model_pipeline
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {path}. "
                 f"Please ensure you have run the Model Export step (2.7) correctly and the path is accessible.")
        # Attempt to fix the path if running from root directory vs. ui directory
        if not os.path.exists(path):
             # This assumes we are running from 'ui/' and need to look up one level for 'models/'
             path = os.path.join(os.path.dirname(__file__), '..', path)
             if os.path.exists(path):
                 return joblib.load(path)
        st.stop()
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

# Load the model globally
model = load_model(MODEL_PATH)


# =============================================================================
# Prediction Logic
# =============================================================================

def predict_risk(input_data):
    """Makes a prediction and calculates probability."""
    
    # 1. Convert input data to DataFrame with correct feature order
    # Streamlit inputs are ordered by the FEATURE_NAMES list.
    input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
    
    # 2. Make Prediction (0 or 1)
    prediction = model.predict(input_df)[0]
    
    # 3. Get Probability (P(Class 0) and P(Class 1))
    probabilities = model.predict_proba(input_df)[0]
    risk_prob = probabilities[1] # Probability of the positive class (Heart Disease Present)
    
    return prediction, risk_prob


# =============================================================================
# Streamlit UI Components
# =============================================================================

st.title("🫀 Heart Disease Risk Predictor")
st.markdown("Enter the patient's data below to get a prediction using the **Hyperparameter Optimized SVM Model**.")

# --- Sidebar for Feature Descriptions (For user guidance) ---
with st.sidebar:
    st.header("Feature Guide")
    st.info("The model was trained using these 12 features.")
    st.markdown("""
        * **Age**: Age in years.
        * **Sex**: (1 = male; 0 = female).
        * **CP (Chest Pain Type)**: 0 (Asymptomatic) to 3 (Typical Angina).
        * **Trestbps**: Resting blood pressure (mm Hg).
        * **Chol**: Serum cholestoral (mg/dl).
        * **Fbs (Fasting Blood Sugar)**: (> 120 mg/dl, 1 = true; 0 = false).
        * **Restecg**: Resting electrocardiographic results (0, 1, or 2).
        * **Thalach**: Maximum heart rate achieved.
        * **Exang**: Exercise induced angina (1 = yes; 0 = no).
        * **Oldpeak**: ST depression induced by exercise relative to rest.
        * **CA (Major Vessels)**: Number of major vessels colored by fluoroscopy (0-3).
        * **Thal**: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect).
    """)


# --- Input Form ---

# Create a dictionary to hold user inputs
input_data = {}

# Layout the inputs in three columns for a clean look
col1, col2, col3 = st.columns(3)

with col1:
    input_data['age'] = st.slider("Age (years)", 29, 77, 54, help="Patient's age.")
    # Sex is categorical but binary
    sex_options = [(1, "Male"), (0, "Female")]
    input_data['sex'] = st.selectbox("Sex", options=sex_options, format_func=lambda x: x[1], help="1=Male, 0=Female")[0]
    input_data['trestbps'] = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 130, help="Systolic pressure at rest.")
    input_data['chol'] = st.number_input("Cholesterol (mg/dl)", 100, 564, 246, help="Serum cholesterol level.")

with col2:
    # CP is a multi-category feature
    input_data['cp'] = st.slider("Chest Pain Type (CP)", 0, 3, 1, help="Type of chest pain experienced (0-3).")
    input_data['thalach'] = st.slider("Max Heart Rate (beats/min)", 71, 202, 149, help="Maximum heart rate achieved during exercise.")
    # Exang is binary
    exang_options = [(1, "Yes"), (0, "No")]
    input_data['exang'] = st.selectbox("Exercise Induced Angina (Exang)", options=exang_options, format_func=lambda x: x[1], help="Angina (chest pain) induced by exercise.")[0]
    # Fbs is binary
    fbs_options = [(1, "True"), (0, "False")]
    input_data['fbs'] = st.selectbox("Fasting Blood Sugar (>120 mg/dl)", options=fbs_options, format_func=lambda x: x[1], help="1=True (high blood sugar), 0=False.")[0]

with col3:
    input_data['oldpeak'] = st.slider("Oldpeak (ST Depression)", 0.0, 6.2, 1.0, 0.1, help="ST depression induced by exercise relative to rest.")
    input_data['restecg'] = st.selectbox("Resting ECG Results (Restecg)", options=[0, 1, 2], help="Electrocardiographic results (0, 1, or 2).")
    input_data['ca'] = st.slider("Major Vessels (CA)", 0, 3, 0, help="Number of major vessels (0-3) colored by fluoroscopy.")
    # Thal is multi-category
    thal_options = [(3, "Normal"), (6, "Fixed Defect"), (7, "Reversible Defect")]
    input_data['thal'] = st.selectbox("Thalassemia (Thal)", options=thal_options, format_func=lambda x: x[1], help="A type of blood disorder.")[0]

# --- Prediction Button and Output ---

st.divider()

if st.button("Analyze Risk", type="primary"):
    with st.spinner('Analyzing data and predicting risk...'):
        
        # We need to map the dictionary inputs to the ordered list of features
        ordered_input = [input_data[feature] for feature in FEATURE_NAMES]
        
        # Get prediction and probability
        prediction, risk_prob = predict_risk(ordered_input)
        
        # Format probability as percentage
        risk_percent = risk_prob * 100
        
        st.header("Prediction Result")
        
        # Display results
        if prediction == 1:
            st.error(f"⚠️ High Risk Detected! Prediction: **Heart Disease Present**")
            st.metric(label="Probability of Disease", value=f"{risk_percent:.2f}%", delta_color="inverse")
            st.balloons()
        else:
            st.success(f"✅ Low Risk Detected! Prediction: **No Heart Disease**")
            st.metric(label="Probability of Disease", value=f"{risk_percent:.2f}%")

        st.info("This is a machine learning prediction and should not replace professional medical advice.")
