# Phishing Website Detection Using Machine Learning

This project focuses on detecting phishing websites using machine learning classification models.  
The dataset was sourced from the UCI Machine Learning Repository and contains 1353 website instances with 9 features.

## 🔍 Project Goal
Classify websites as **phishing (1)** or **legitimate (0)** based on URL, SSL certificate, pop-up behavior, and other website attributes.

## 🧠 Methods Used
- Exploratory Data Analysis
- Data Preprocessing + Standardization
- Model Training + Performance Evaluation
- Feature Selection (SelectKBest)

## 🤖 Models Implemented
| Model | Type |
|------|------|
| Decision Tree Classifier | Supervised |
| Logistic Regression | Supervised |
| K-Nearest Neighbors (KNN) | Supervised |
| Support Vector Machine (SVM) | Supervised |
| Gaussian Naive Bayes | Probabilistic |

Performance metrics used:
- Accuracy
- Precision
- Recall
- F1 Score

## 🎯 Feature Selection Result
The top 3 most informative features were:
1. **SFH**
2. **popUpWindow**
3. **SSLfinal_State**

## 📊 Observations
Feature selection reduced dimensionality but **did not improve model performance** in most models.

## 📄 Files
- `phishing_detection.ipynb` — Complete code
- `report.pdf` — Full analysis, visualizations, and discussion

## 📚 Dataset Source
UCI Machine Learning Repository  
https://doi.org/10.24432/C5B301
