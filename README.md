# Heart Disease Risk Prediction

A machine learning–based web application that predicts the risk of heart disease using clinical health parameters.

## Description

The Heart Disease Risk Prediction project focuses on identifying whether an individual is at high or low risk of developing heart disease based on medical attributes. The system uses supervised machine learning techniques, where multiple classification models are trained and evaluated. The best-performing model is deployed through an interactive Streamlit web application. This project is developed for educational and demonstration purposes.

## Features

- User-friendly Streamlit web interface
- Real-time heart disease risk prediction
- Machine learning–based classification
- Standardized data preprocessing
- Clear prediction output (High Risk / Low Risk)

## Getting Started

### Dependencies

- Python 3.8 or higher
- Operating System: Windows / macOS / Linux

Required Python libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- joblib
- xgboost

All required dependencies are listed in the requirements.txt file.

### Installing
1. Clone the repository
git clone https://github.com/akshayavardhini/heart-disease-risk-prediction.git

2. Navigate to the project directory
cd heart-disease-risk-prediction

3. Install dependencies
pip install -r requirements.txt

### Executing Program
1. Ensure all dependencies are installed

2. Run the application
streamlit run app.py

3. Open the browser and visit
http://localhost:8501

4. Enter the required health parameters to view the prediction result

## Project Structure
├── app.py
├── model_train.ipynb
├── heart.csv
├── knn_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md

## Help

If you encounter issues while running the application, try the following:
pip install --upgrade pip
pip install -r requirements.txt

Ensure Python is properly installed and added to the system PATH.

## Disclaimer

This project is intended for educational and learning purposes only.  
It is not a medical diagnostic tool.  
Always consult a certified healthcare professional for medical advice.

## Authors

Akshaya Vardhini  
Pre-Final Year M.Tech Integrated Software Engineering Student, VIT  
Machine Learning | Data Science | Web Development

## Acknowledgments

- UCI / Kaggle Heart Disease Dataset
- Scikit-learn Documentation
- Streamlit Community
- Open-source Machine Learning Contributors



