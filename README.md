# Pima Indians Diabetes Prediction Pipeline

This repository contains a comprehensive machine learning pipeline designed to predict diabetes onset based on clinical diagnostic measurements. Using the **Pima Indians Diabetes Dataset**, this project explores data imputation, feature scaling, and the comparative performance of multiple classification algorithms.

## 🚀 Project Highlights
* **Problem Type:** Binary Classification
* **Dataset:** 768 records with 8 medical features (Glucose, BMI, Age, etc.)
* **Core Goal:** Build a robust predictor while minimizing false negatives for clinical safety.
* **Key Tech:** Scikit-Learn, XGBoost, Pandas, Seaborn.

---

## 🛠️ The Pipeline

### 1. Data Cleaning & Imputation
A critical step in this project was identifying "hidden" missing values. While the dataset had no nulls, features like `Glucose` and `BMI` contained zeros that are physiologically impossible.
* **Strategy:** Replaced `0` with `NaN` and applied median/mean imputation.
* **Result:** Preserved dataset size while removing bias from invalid data points.

### 2. Exploratory Data Analysis (EDA)
Used statistical visualization to uncover feature relationships.
* **Correlation Heatmaps:** Identified `Glucose` and `BMI` as the strongest predictors.
* **Distributions:** Analyzed class balance (~2:1 ratio) to inform evaluation strategy.



### 3. Feature Engineering
* **Standardization:** Applied `StandardScaler` to normalize features (Mean ≈ 0, Std ≈ 1). This was essential for the performance of KNN and SVM.
* **Data Splitting:** Used an 80/20 train-test split with stratification to ensure the target distribution remained consistent.

### 4. Model Building & Hyperparameter Tuning
I implemented and compared seven different models:
1. **Logistic Regression** (Baseline)
2. **K-Nearest Neighbors (KNN)**
3. **Gaussian Naïve Bayes**
4. **Support Vector Machine (SVM)**
5. **Decision Tree**
6. **Random Forest** (Ensemble)
7. **XGBoost** (Gradient Boosting)

**Optimization:** Every model underwent `GridSearchCV` with 5-fold cross-validation to tune hyperparameters (e.g., $C$ for SVM, `n_estimators` for Random Forest).

---

## 📊 Results & Evaluation

The models were evaluated using **Accuracy**, **ROC-AUC**, and **Confusion Matrices**. The primary focus was the **ROC-AUC score** because it evaluates the model's ability to distinguish between classes across all thresholds.

| Model | Accuracy | ROC-AUC |
| :--- | :--- | :--- |
| **Random Forest (Best)** | **~77%** | **0.XX** |
| XGBoost | ~76% | 0.XX |
| Logistic Regression | ~75% | 0.XX |
| SVM (RBF) | ~74% | 0.XX |

### Confusion Matrix Insights
For medical applications, **Recall** is vital. Our final model was selected because it successfully minimized False Negatives compared to baseline linear models.



---

## 📁 Project Structure
* `diabetes_prediction.ipynb`: Main Jupyter Notebook with code and analysis.
* `data/`: Contains the Pima Indians CSV file.
* `requirements.txt`: List of dependencies.
* `plots/`: Generated EDA and Evaluation visuals.

## ⚙️ Installation & Usage
1. Clone the repo: `git clone https://github.com/yourusername/diabetes-prediction.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook to see the full analysis.

---
