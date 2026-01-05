import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time

model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.markdown("""
    <style>
    .block-container {
        padding-top: 0rem;
    }
    header, footer {
        visibility: hidden;
    }
    .main {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
    }
    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        color: #ff4b4b;
    }
    .result-card {
        background-color: #fff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-top: 20px;
        text-align: center;
    }
    .footer {
        margin-top: 30px;
        font-size: 0.9rem;
        color: gray;
        text-align: center;
    }
    .stButton button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
    .stButton button:hover {
        background-color: #c23434;
    }
    [data-testid="stSidebar"] {
        background-color: white;
        padding: 2rem 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }     
    .sidebar-title {
        font-size: 1.6rem;
        font-weight: bold;
        color: #ff4b4b;
        margin-bottom: 1rem;
    }
    .sidebar-section {
        margin-bottom: 1.5rem;
    }
    .sidebar-text {
        font-size: 1rem;
        color: #333333;
        margin: 0.3rem 0;
        line-height: 1.5;
    }
    .sidebar-link {
        font-size: 1rem;
        font-weight: bold;
        color: #ff4b4b;
        text-decoration: none;
    }
    .sidebar-link:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)


st.sidebar.markdown('<div class="sidebar-title">Heart Risk Predictor</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="sidebar-section sidebar-text">
<b>Purpose:</b><br>
This app predicts the <b>risk of heart disease</b> based on patient health details using machine learning.
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-section sidebar-text">
<b>Model Used:</b><br>
K-Nearest Neighbors (KNN) Classifier trained on clinical heart health data.
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-section sidebar-text">
⚠ <b>Disclaimer:</b><br>
This tool is for educational use only. Please consult a doctor for real medical advice.
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-section sidebar-text">
🌐 <a class="sidebar-link" href="https://www.cdc.gov/heartdisease/" target="_blank">CDC: Heart Disease Info</a>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-section sidebar-text">
👩‍💻 <b>Created By:</b><br>
<i>Akshaya Vardhini<br><i>

</div>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Heart Disease Risk Predictor</div>', unsafe_allow_html=True)
st.markdown("Fill the patient's health details to check the risk for heart disease")

with st.expander("💡 What do the below fields mean?"):
    st.markdown("""
    - *cp*: Chest pain type (0-3) <br>
        &nbsp;&nbsp;&nbsp;&nbsp;0 - Pain during exercise <br>
        &nbsp;&nbsp;&nbsp;&nbsp;1 - Unusual chest pain <br>
        &nbsp;&nbsp;&nbsp;&nbsp;2 - Pain not from the heart <br>
        &nbsp;&nbsp;&nbsp;&nbsp;3 - No chest pain
        
    - *trestbps*: Resting BP <br>
        &nbsp;&nbsp;&nbsp;&nbsp;Men - 120/80 mm Hg(Normal) <br>
        &nbsp;&nbsp;&nbsp;&nbsp;Women - 115/75 mm Hg(Normal) <br>
                
    - *restech*: Resting ECG (0-2) <br>
        &nbsp;&nbsp;&nbsp;&nbsp;0 - Normal <br>
        &nbsp;&nbsp;&nbsp;&nbsp;1 - Minor issue <br>
        &nbsp;&nbsp;&nbsp;&nbsp;2 - Signs of heart strain 
    
    - *exang*: Exercise Induced-Angina <br>
        &nbsp;&nbsp;&nbsp;&nbsp;Experience chest pain during or after exercise <br>

    - *scope*: Scope of ST (0-2) <br>
        &nbsp;&nbsp;&nbsp;&nbsp;0 - Downslopping(possible heart issue)  <br>
        &nbsp;&nbsp;&nbsp;&nbsp;1 - Flat(warning sign)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;2 - Unsloping(normal)               
                
    - *ca*: Coronary ArteryDisease (0-4) <br>
        &nbsp;&nbsp;&nbsp;&nbsp;0 - No Blockage<br>
        &nbsp;&nbsp;&nbsp;&nbsp;1 - Mild Blockage<br>
        &nbsp;&nbsp;&nbsp;&nbsp;2 - Moderate Blockage <br>
        &nbsp;&nbsp;&nbsp;&nbsp;3 - Severe Blockage <br>
        &nbsp;&nbsp;&nbsp;&nbsp;3 - Very Severe Blockage in Multiple Arteries
                
    - *thal*: Thalassemia (0-3) <br>
        &nbsp;&nbsp;&nbsp;&nbsp;0 - Normal  <br>
        &nbsp;&nbsp;&nbsp;&nbsp;1 - Minor<br>
        &nbsp;&nbsp;&nbsp;&nbsp;2 - Intermediate <br>
        &nbsp;&nbsp;&nbsp;&nbsp;3 - Major
    """, unsafe_allow_html=True)

with st.form("heart_form"):
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 1 else "Male")
    cp = st.selectbox("Constrictive Pericarditis (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting BP (trestbps)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Sugar(fbs) > 120", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate(thalach)", 60, 250, 150)
    exang = st.selectbox("Exercise Induced-Angina (exang)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of ST", [0, 1, 2])
    ca = st.selectbox("Coronary ArteryDisease (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])
    submit = st.form_submit_button("🔍 Predict")

if submit:
    with st.spinner("Analyzing patient data..."):
        time.sleep(1.5)
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                    thalach, exang, oldpeak, slope, ca, thal]],
                                  columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                           'restecg', 'thalach', 'exang', 'oldpeak',
                                           'slope', 'ca', 'thal'])
        scaled = scaler.transform(input_data)
        prediction = model.predict(scaled)[0]

    if prediction == 1:
        st.markdown("""
            <div class="result-card" style="background-color:#dc143c; color:white;">
                <p style="font-weight:bold;">High risk of heart disease detected.</p>
                <p>Consider scheduling an immediate check-up with your cardiologist</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="result-card" style="background-color:#006400; color:white;">
                <p style="font-weight:bold;">Low risk of heart disease.</p>
                <p>Keep a healthy lifestyle and stay regular with check-ups</p>
            </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="footer">This app is for informational purposes only. Always consult a medical professional for accurate health advice.</div>', unsafe_allow_html=True)