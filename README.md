
# Credit Risk Prediction for Loan Applicants

## **Author:** Daniel Lawrence - DT20234270647  
### Links:


**Streamlit Demo [Streamlit Link](https://credit-risk-for-loan.streamlit.app/)**
or 
https://credit-risk-for-loan.streamlit.app/


**🎥 Demo [Watch Demo](https://drive.google.com/file/d/1NUE_JNjyujkoXCAUCWBmHkJdk5WDtvnA/view?usp=sharing)**
or 
https://drive.google.com/file/d/1NUE_JNjyujkoXCAUCWBmHkJdk5WDtvnA/view?usp=sharing


**REPORT LINK: [📄 Download Report - Daniel Lawrence - DT20234270647 (PDF)](./Report%20-%20Daniel%20Lawrence%20-%20DT20234270647.pdf)**



**Colab Link: [Run on Google Colab](https://colab.research.google.com/drive/1DqkOasBa4ykOP05PbxLLIQftcdxJGrHZ?usp=sharing)**
or 
https://colab.research.google.com/drive/1DqkOasBa4ykOP05PbxLLIQftcdxJGrHZ?usp=sharing





## 📌 Overview

This project focuses on building a **machine learning pipeline** to assess **credit risk** of loan applicants. We aim to predict whether a loan applicant is a **good** or **bad** credit risk based on demographic and financial data.

By cleaning data, transforming features, and trying a range of classification models, the goal was to find the **most accurate and interpretable** solution for a real-world finance problem.

---

## 🧠 Problem Statement

Loan default can cost financial institutions *millions*. This project helps **predict risk** before giving out a loan, turning raw data into **insightful decisions**.  
Using historical loan data, we label applicants as:

- ✅ **Good Credit Risk** (Low risk)
- ❌ **Bad Credit Risk** (High risk)

---

## 📊 Dataset

The dataset contains **1000 entries**, each representing a loan applicant.  
Key features include:

- **Demographics**: Age, Sex, Job
- **Financials**: Credit Amount, Duration, Checking & Saving Accounts
- **Loan Info**: Purpose, Housing, etc.

### Handling Missing Values:
- `Saving accounts` and `Checking account` had missing values.
- Replaced with **'unknown'** to preserve potential signal of missingness.

---

## 🔧 Data Preprocessing

- **Feature Engineering**: Binned durations, encoded sex, and defined credit risk using median threshold.
- **Encoding**:
  - Binary encoding for `Sex`, `Credit Amount` (as target).
  - One-hot encoding for `Saving accounts`, `Checking account`, `Housing`, and `Purpose`.

- **Class Balance**: Created a balanced binary classification task by splitting based on the median credit amount.

---

## 📈 Exploratory Data Analysis

Visual and statistical analysis revealed:

- **Most loans** were for **2 years or less**.
- **Larger credit amounts** are less common and likely require higher creditworthiness.
- **Job type, housing status**, and **account info** show meaningful patterns in risk.
- **Gender differences** in credit risk distribution are notable.

---

## 🤖 Models Tested

We evaluated **12 classification algorithms**:

- Logistic Regression
- K-Nearest Neighbors
- Naive Bayes
- Decision Tree
- Random Forest 🌲
- XGBoost ⚡
- LightGBM
- SVM
- Gradient Boosting
- AdaBoost ⭐
- Bagging
- Extra Trees

### 🛠️ Hyperparameter Tuning:
Used **GridSearchCV** for models like Logistic Regression and XGBoost.

---

## 🏆 Best Models

- **Random Forest** and **AdaBoost** delivered the highest accuracy.
- The **Random Forest model** was selected and saved for deployment.

---

## 🧪 Evaluation Metric

- **Accuracy** was the primary performance metric.

---

## 🚀 Run the Code

Open and execute the full notebook in Colab:  
👉 [Credit Risk Prediction Notebook](https://colab.research.google.com/drive/1DqkOasBa4ykOP05PbxLLIQftcdxJGrHZ?usp=sharing)

---
