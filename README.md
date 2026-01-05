# Heart-Risk-Predictor
A Machine Learning–based Heart Disease Risk Predictor built using Python, Scikit-learn, and Streamlit.
This application predicts whether a person is at high or low risk of heart disease based on clinical health parameters.

Project Overview
Heart disease is one of the leading causes of death worldwide. Early prediction can help in timely medical intervention.
This project uses machine learning classification models trained on clinical data to predict heart disease risk.
The final deployed app uses a K-Nearest Neighbors (KNN) model with standardized input features.

Features
Interactive Streamlit web interface
User-friendly form for health inputs
Machine Learning–based prediction
Real-time result display (High / Low Risk)
Educational medical disclaimer
Clean UI with custom styling

Machine Learning Models Used (During Training)
The following models were trained and evaluated:
Random Forest Classifier
Support Vector Classifier(SVC)
K-Nearest Neighbors(KNN) - Final Model
Naive Bayes
Histogram Gradient Boosting
Voting Classifier (Ensemble)
KNN was selected based on performance and stability.

Dataset
Dataset Name: Heart Disease Dataset
Target Column: target
Task Type: Binary Classification (0 → Low Risk, 1 → High Risk)
Input Features:
Age
Sex
Chest Pain Type (cp)
Resting Blood Pressure (trestbps)
Cholesterol (chol)
Fasting Blood Sugar (fbs)
Resting ECG (restecg)
Maximum Heart Rate (thalach)
Exercise-Induced Angina (exang)
Oldpeak
ST Slope (slope)
Number of Major Vessels (ca)
Thalassemia (thal)

Tech Stack
Programming Language: Python
Frontend: Streamlit
Machine Learning: Scikit-learn, XGBoost
Data Handling: Pandas, NumPy
Model Persistence: Joblib
Visualization: Matplotlib, Seaborn
