# Heart-Risk-Predictor
A Machine Learning–based Heart Disease Risk Predictor built using Python, Scikit-learn, and Streamlit.
This application predicts whether a person is at high or low risk of heart disease based on clinical health parameters.

📌 Project Overview

Heart disease is one of the leading causes of death worldwide. Early prediction can help in timely medical intervention.
This project uses machine learning classification models trained on clinical data to predict heart disease risk.

The final deployed app uses a K-Nearest Neighbors (KNN) model with standardized input features.

🚀 Features

Interactive Streamlit web interface

User-friendly form for health inputs

Machine Learning–based prediction

Real-time result display (High / Low Risk)

Educational medical disclaimer

Clean UI with custom styling

🧠 Machine Learning Models Used (During Training)

The following models were trained and evaluated:

Logistic Regression

Random Forest Classifier

Support Vector Classifier (SVC)

K-Nearest Neighbors (KNN) ✅ (Final Model)

Naive Bayes

Histogram Gradient Boosting

Voting Classifier (Ensemble)

KNN was selected based on performance and stability.

📊 Dataset

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

🛠 Tech Stack

Programming Language: Python

Frontend: Streamlit

Machine Learning: Scikit-learn, XGBoost

Data Handling: Pandas, NumPy

Model Persistence: Joblib

Visualization: Matplotlib, Seaborn

📁 Project Structure
├── app.py                 # Streamlit web application
├── model_train.ipynb      # Model training & evaluation notebook
├── heart.csv              # Dataset
├── knn_model.pkl          # Trained KNN model
├── scaler.pkl             # StandardScaler object
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/heart-disease-predictor.git
cd heart-disease-predictor

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Application
streamlit run app.py


The app will open in your browser at:

http://localhost:8501

🧪 Model Training (Optional)

If you want to retrain the model:

Open model_train.ipynb

Run all cells

New .pkl files will be generated automatically

⚠️ Disclaimer

This application is developed for educational and learning purposes only.
It is not a medical diagnostic tool.
Always consult a certified medical professional for accurate diagnosis and treatment.

👩‍💻 Author

Akshaya Vardhini
Pre-final year student | Machine Learning & Python Enthusiast

🌟 Acknowledgements

UCI / Kaggle Heart Disease Dataset

Scikit-learn Documentation

Streamlit Community

📌 Future Improvements

Add probability-based risk score

Deploy on Streamlit Cloud

Add model comparison dashboard

Improve feature explanations for users

If you want, I can also:

✨ Make it more placement-oriented

🚀 Customize it for Streamlit Cloud deployment

📄 Add badges (Python, Streamlit, ML)

🧠 Rewrite it in resume-project style

Just tell me 💙

give description of the project for uploading in github

Here are clean, concise project descriptions you can use directly on GitHub.
I’m giving you 3 versions — pick the one that fits your style best 👇

🔹 Short Description (Recommended for GitHub repo “About” section)

A Machine Learning–based web application built with Streamlit that predicts heart disease risk using clinical health data. The model is trained using Scikit-learn and provides real-time, user-friendly predictions for educational purposes.

🔹 Medium Description (Professional & Placement-Friendly)

This project is a Heart Disease Risk Prediction web application developed using Python, Scikit-learn, and Streamlit. Multiple machine learning models were trained and evaluated, and the final KNN model was deployed to predict the risk of heart disease based on patient health parameters. The application provides an interactive UI and real-time predictions for educational use.

🔹 Detailed Description (If GitHub allows longer text)

Heart Disease Risk Prediction is a machine learning–based web application that predicts whether a patient is at high or low risk of heart disease using clinical health attributes. The project involves data preprocessing, feature scaling, training and evaluating multiple classification models, and deploying the best-performing model using Streamlit. The application is designed for learning and demonstration purposes and offers an intuitive interface for users to input health data and receive instant predictions.

✅ My Recommendation

Use Version 1 for:

GitHub About section

Use Version 2 inside:

README.md

Resume project descriptions

If you want, I can also:

tailor it to ATS-friendly resume language

make it more technical

write a one-line LinkedIn project description
