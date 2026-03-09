import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Clinical Diabetes Risk Dashboard")
st.markdown("""
Welcome to the diagnostic predictor tool. 
Please enter the patient's vitals and laboratory metrics in the **sidebar on the left** to generate a real-time risk assessment.
""")
st.caption("Enter the patient's measurements and click Predict.")
st.divider()

@st.cache_resource
def load_model():
    model_path="patients_pipeline.pkl"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model = load_model()

if model is None:
    st.error("⚠️ Model file 'patients_pipeline.pkl' not found. Please ensure it is in the same folder as this app.")
    st.stop


# side bar

with st.sidebar:
    st.header("Patient Demographics & Vitals")
    st.caption("Please input the laboratory results below.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age         = st.number_input("Age",                min_value=21,  max_value=90,  value=35, help="Years")
        pregnancies = st.number_input("Pregnancies",        min_value=0,   max_value=20,  value=3)
        glucose     = st.number_input("Glucose",            min_value=40,  max_value=300, value=120, help="Plasma glucose concentration (mg/dL)")
        bp          = st.number_input("Blood Pressure",     min_value=20,  max_value=130, value=72, help="Diastolic blood pressure (mm Hg)")
        skin        = st.number_input("Skin Thickness",     min_value=5,   max_value=100, value=23, help="Triceps skin fold thickness (mm)")

    with col2:
        bmi         = st.number_input("BMI",                min_value=15.0,max_value=70.0,value=32.0, step=0.1, help="Body mass index (kg/m2)")
        insulin     = st.number_input("Insulin",            min_value=0,   max_value=900, value=79, help="2-Hour serum insulin (mu U/ml)")
        dpf         = st.number_input("Pedigree",           min_value=0.0, max_value=3.0, value=0.47, step=0.001, format="%.3f", help="Diabetes pedigree function")
        
        bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"], index=2)
        blood_type   = st.selectbox("Blood Type", ["A", "B", "AB", "O"], index=3)

    st.divider()
    
    predict_button = st.button("Generate Risk Assessment", use_container_width=True, type="primary")

def build_patient_profile():
    bmi_mapping = ["Underweight", "Normal", "Overweight", "Obese"]
    bmi_cat_enc = bmi_mapping.index(bmi_category)
    
    insulin_glucose_ratio = insulin / (glucose + 1)
    
    blood_B  = 1 if blood_type == "B" else 0
    blood_AB = 1 if blood_type == "AB" else 0
    blood_O  = 1 if blood_type == "O" else 0
    
    return pd.DataFrame([{
        "pregnancies"               : pregnancies,
        "glucose"                   : glucose,
        "blood_pressure"            : bp,
        "skin_thickness"            : skin,
        "insulin"                   : insulin,
        "bmi"                       : bmi,
        "diabetes_pedigree_function": dpf,
        "age"                       : age,
        "bmi_category_enc"          : bmi_cat_enc,
        "insulin_glucose_ratio"     : insulin_glucose_ratio,
        "blood_B"                   : blood_B,
        "blood_AB"                  : blood_AB,
        "blood_O"                   : blood_O,
    }])

import plotly.graph_objects as go

if predict_button:
    
    patient_data = build_patient_profile()
    prediction = int(model.predict(patient_data)[0])
    probabilities = model.predict_proba(patient_data)[0] 
    
    p_no_dm = probabilities[0]
    p_dm = probabilities[1]
    
    st.subheader("Diagnostic Results")
    
    if prediction == 1:
        st.error(f"### Diabetes Risk Detected\nProbability of diabetes: **{p_dm:.1%}**")
    else:
        st.success(f"### Low Diabetes Risk\nProbability of no diabetes: **{p_no_dm:.1%}**")
        
    st.write("")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("P(No Diabetes)", f"{p_no_dm:.1%}")
    m2.metric("P(Diabetes)",    f"{p_dm:.1%}")
    
    risk_level = "High" if p_dm >= 0.60 else "Moderate" if p_dm >= 0.30 else "Low"
    m3.metric("Risk level", risk_level)
    
    st.write("---")
    
    st.write("**Diabetes Probability Analysis**")
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = p_dm * 100,
        number = {'suffix': "%", 'font': {'size': 40}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Calculated Risk Score", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "lightyellow"},
                {'range': [60, 100], 'color': "salmon"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))
    
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("For educational use only. Not a medical device.")

else:

    st.info("Please enter the patient details in the sidebar and click 'Generate Risk Assessment'.")
