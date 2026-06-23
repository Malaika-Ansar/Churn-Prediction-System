# 📊 Customer Churn Prediction System

A machine learning web application that predicts whether a telecom customer is likely to churn based on demographic, service usage, and billing information. Built using Python, Scikit-learn, and Streamlit with a One-Hot Encoded classification pipeline.

---

# 📌 Project Overview

Customer churn is a major problem in the telecom industry. This project uses machine learning to analyze customer behavior and predict churn probability, helping businesses take proactive retention actions.

---

# 🎯 Objective

- Predict customer churn (Yes / No)
- Estimate churn probability
- Identify high-risk customers
- Build an interactive ML web app

---

# 📂 Dataset

- **Source:** Telco Customer Churn Dataset (IBM / Kaggle)
- **Size:** ~7043 customers
- **Target Variable:** Churn (0 = No, 1 = Yes)

### Features include:
- Customer demographics (Gender, Senior Citizen, Partner, Dependents)
- Services (Internet, Phone, Streaming, Security, etc.)
- Billing info (Monthly Charges, Total Charges, Payment Method)
- Contract details (Contract type, tenure)

---

# ⚙️ Machine Learning Pipeline

## 🔹 Preprocessing
- Missing value handling (TotalCharges)
- One-Hot Encoding for categorical variables
- Feature alignment using saved column schema

## 🔹 Models Tested
- Logistic Regression (Best Performer)
- K-Nearest Neighbors
- Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

## 🔹 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Cross-validation F1

---

# 🏆 Best Model

- **Model:** Logistic Regression (or best selected model)
- **Pipeline:** StandardScaler + Classifier
- **Optimized Metric:** ROC-AUC / F1 Score

---

# 🌐 Web App (Streamlit)

Interactive UI built using Streamlit where users can:

- Enter customer details
- Get churn prediction instantly
- View churn probability
- See real-time model output

---

# 🔄 App Workflow

1. User enters customer data  
2. Data is encoded using One-Hot Encoding  
3. Features are aligned with training schema  
4. Model predicts churn probability  
5. Result displayed in UI  



# 🛠 Tech Stack

- Python 🐍  
- Pandas & NumPy  
- Scikit-learn  
- Streamlit  
- Joblib (Model saving)  
- Matplotlib / Seaborn (EDA)



# 📁 Project Structure

churn-prediction/
│
├── app1.py # Streamlit app
├── best_onehot_model.pkl # Trained ML model
├── columns_onehot.pkl # Feature columns schema
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset
├── Churnn.ipynb # Model training notebook
└── README.md

# How to Run Locally
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run Streamlit app
streamlit run app.py

# Key Insights
Month-to-month contracts show highest churn risk
Electronic check users churn more frequently
Low tenure customers are more likely to leave
Fiber optic users have higher churn probability

# Future Improvements
Add XGBoost / LightGBM model
Implement SHAP explainability
Deploy on AWS / Render
Add user authentication
Improve threshold tuning for recall optimization
