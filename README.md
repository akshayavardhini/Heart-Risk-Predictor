# Heart Disease Risk Prediction

A machine learning–based web application that predicts the risk of heart disease using clinical health parameters.

## Description

The Heart Disease Risk Prediction project focuses on identifying whether an individual is at high or low risk of developing heart disease based on medical attributes. The system uses supervised machine learning techniques where multiple classification models are trained and evaluated. The best-performing model is deployed using an interactive Streamlit web application. This project is developed for educational and demonstration purposes.

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

All required dependencies are listed in the `requirements.txt` file.

### Installing

Clone the repository:
```bash
git clone https://github.com/akshayavardhini/heart-disease-risk-prediction.git
```

Navigate to the project directory:
```bash
cd heart-disease-risk-prediction
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Executing Program

Run the Streamlit application:
```bash
streamlit run app.py
```

Open your browser and navigate to:
```
http://localhost:8501
```

Enter the required health parameters to view the prediction result.

## Project Structure

```
heart-disease-risk-prediction/
│
├── app.py                 # Streamlit web application
├── model_train.ipynb      # Model training and evaluation
├── heart.csv              # Dataset used for training
├── knn_model.pkl          # Trained KNN model
├── scaler.pkl             # Feature scaling object
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```


## Help

If you encounter issues while installing dependencies or running the application, try:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Ensure Python is installed correctly and added to the system PATH.

## Disclaimer

This project is intended for educational and learning purposes only.  
It is not a medical diagnostic tool.  
Always consult a certified healthcare professional for medical advice.

## Authors

**Akshaya Vardhini**  
Pre-Final Year M.Tech Integrated Software Engineering Student, VIT  
Machine Learning | Data Science | Web Development  

## Acknowledgments

- Kaggle Heart Disease Dataset  
- Scikit-learn Documentation  
- Streamlit Community  
- Open-source Machine Learning Contributors  
