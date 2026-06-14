# 📦 Vendor Invoice Intelligence Portal

A Machine Learning-powered procurement analytics platform that predicts freight costs and identifies high-risk invoices to support smarter purchasing decisions and improve operational efficiency.

## 📌 Overview

Procurement teams often handle large volumes of vendor invoices and transportation costs, making it difficult to identify unusual expenses and potential invoice risks manually.

The Vendor Invoice Intelligence Portal leverages Machine Learning techniques to automate procurement analytics through:

* Freight Cost Prediction
* Invoice Risk Flagging
* Feature Engineering
* Interactive Business Intelligence Dashboard

The system enables procurement teams to make data-driven decisions, improve cost visibility, and proactively identify potentially risky transactions.

---

## 🚀 Key Features

### Freight Cost Prediction

Predicts expected freight costs using historical procurement and transportation data.

### Invoice Risk Flagging

Identifies potentially risky invoices based on procurement patterns and transactional indicators.

### Feature Engineering Pipeline

Transforms raw procurement data into business-relevant predictive features.

### Real-Time Predictions

Allows users to generate predictions through an interactive Streamlit application.

### Business Decision Support

Provides actionable insights for procurement optimization and invoice review processes.

---

## 🧠 Machine Learning Approach

### Regression Module

Used for freight cost estimation.

**Models Evaluated**

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

### Classification Module

Used for invoice risk detection.

**Model Used**

* Decision Tree Classifier

### Hyperparameter Optimization

* GridSearchCV
* Cross Validation
* Model Performance Comparison

---

## 📊 Data Processing Pipeline

1. Procurement Data Collection
2. Data Cleaning
3. Missing Value Treatment
4. Feature Engineering
5. Categorical Encoding
6. Model Training
7. Model Evaluation
8. Risk Classification
9. Freight Cost Prediction
10. Streamlit Deployment

---

## 🔧 Feature Engineering

The system derives business-focused features from procurement and invoice records, including:

* Vendor Performance Indicators
* Delivery Delay Metrics
* Purchase Aggregation Features
* Invoice Characteristics
* Cost-Related Variables
* Procurement Activity Patterns

These engineered features improve predictive accuracy and business relevance.

---

## 📈 Model Performance

### Freight Cost Prediction

| Metric                    | Result |
| ------------------------- | ------ |
| R² Score                  | 0.97   |
| Mean Absolute Error (MAE) | 24.46  |

### Invoice Risk Flagging

| Metric            | Result |
| ----------------- | ------ |
| Accuracy          | 82%    |
| Weighted F1 Score | 0.82   |

The models demonstrated strong predictive capability and reliable performance on unseen data.

---

## 📊 Business Impact

The platform helps procurement teams:

* Estimate transportation expenses more accurately
* Detect potentially risky invoices before approval
* Reduce manual review effort
* Improve procurement transparency
* Support data-driven purchasing decisions

---

## 💻 Technology Stack

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Joblib

### Deployment

* Streamlit

### Machine Learning

* Linear Regression
* Decision Tree Regression
* Random Forest Regression
* Decision Tree Classification

---

## 🖥️ Application Workflow

### Freight Cost Prediction

1. User enters procurement information.
2. Data is preprocessed and transformed.
3. Regression model estimates freight cost.
4. Predicted cost is displayed.

### Invoice Risk Flagging

1. User enters invoice details.
2. Features are generated and processed.
3. Classification model evaluates risk level.
4. Risk prediction is displayed instantly.

---

## 📂 Project Structure

```text
Vendor-Invoice-Intelligence-Portal/
│
├── Freight Cost Prediction/
    ├── data_preprocessing.py
    ├── model_evaluation.py
    ├── train.py
    ├── models/
        ├── predict_freight_model.pkl
├── Inventory/
    ├── begin_inventory.csv
    ├── end_inventory.csv
    ├── purchase_prices.csv
    ├── vendor_invoice.csv
├── Invoice Flagging/
    ├── data_preprocessing.py
    ├── model_evaluation.py
    ├── train.py
├── inference/
    ├── predict_freight.py
    ├── predict_invoice_flag.py
├── models/
    ├── predict_flag_invoice.pkl
    ├── predict_freight_model.pkl
├── notebooks/
    ├── Freight Cost notebook.ipynb
    ├── Invoice Flagging notebook.ipynb
├── screenshots/
├── README.md
└── app.py
```

## 🌟 Future Enhancements

* Advanced Ensemble Models
* Explainable AI (XAI) Integration
* Vendor Performance Dashboard
* Automated Procurement Risk Scoring
* Cloud Deployment
* Real-Time Data Integration

---

## 👨‍💻 Author

**Arjun Goel**

Master of Computer Applications (2025–2027)

Vellore Institute of Technology, Vellore

LinkedIn: linkedin.com/in/arjun11goel

GitHub: github.com/arjun11goel
