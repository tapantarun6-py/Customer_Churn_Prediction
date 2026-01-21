📊 Customer Churn Prediction Using Machine Learning

📌 Project Overview

Customer Churn Prediction is a machine learning project that predicts whether a customer will leave a service based on historical customer data.
The goal is to help businesses identify customers at risk of churn and take proactive retention measures.

This project uses a real-world telecom dataset and applies supervised classification techniques to analyze customer behavior and identify key churn factors.

🎯 Objectives

Predict customer churn (Yes / No)

Perform data preprocessing and feature engineering

Train and evaluate machine learning models

Identify key factors influencing customer churn

Save the trained model for future use

🧠 Problem Type

Binary Classification

Target variable:

Churn (Yes = Customer leaves, No = Customer stays)

🛠️ Technologies Used

Python

Pandas & NumPy – Data processing

Matplotlib & Seaborn – Data visualization

Scikit-learn – Machine learning models

Joblib – Model persistence

VS Code / Jupyter Notebook

📂 Project Structure
Customer_Churn_Prediction/
│
├── data/
│   └── churn.csv
│
├── model/
│   ├── churn_model.pkl
│   └── scaler.pkl
│
├── churn_prediction.py
├── check_data.py
├── requirements.txt
└── README.md

📊 Dataset

Dataset Name: Telco Customer Churn Dataset

Source: Kaggle

Records: ~7,000 customers

Features: Demographics, services used, contract details, billing information

Target Column: Churn

🔄 Project Workflow

Load dataset

Data cleaning and preprocessing

Encode categorical variables

Feature scaling

Train-test split

Model training (Random Forest)

Model evaluation

Feature importance analysis

Save trained model

🤖 Machine Learning Model

Random Forest Classifier

Chosen for:

High accuracy

Robustness to noise

Feature importance extraction

📈 Model Evaluation Metrics

Accuracy

Confusion Matrix

Precision, Recall, F1-Score

Feature Importance Visualization

🚀 How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Verify Dataset (Optional but Recommended)
python check_data.py

3️⃣ Train Model
python churn_prediction.py

📌 Expected Results

Accuracy: ~80–85%

Printed evaluation metrics

Feature importance graph

Saved model files in model/ directory

🔍 Key Insights

Contract type significantly impacts churn

Customers with higher monthly charges are more likely to churn

Longer tenure customers tend to stay

Payment method influences churn probability

💾 Model Persistence

The trained model and scaler are saved using Joblib:

churn_model.pkl

scaler.pkl

These can be used later for deployment or real-time predictions.

📌 Future Enhancements

Add XGBoost or Gradient Boosting

Build a Flask / FastAPI web app

Create a dashboard using Power BI / Tableau

Perform hyperparameter tuning

Deploy on cloud (AWS / Render / Heroku)

👨‍💻 Author

Tarun Tapan Tripathy
AI Intern | Machine Learning Enthusiast
B.Tech – Computer Science & Engineering

📜 License

This project is for educational and learning purposes.