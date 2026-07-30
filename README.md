# Loan Default Risk Prediction

## Project Overview
This project predicts whether a customer is likely to default on a loan using Machine Learning. The model analyzes customer financial and personal information to estimate the risk of loan default. This helps financial institutions make informed lending decisions and reduce financial losses.

## Objectives
- Predict loan default risk.
- Improve loan approval decisions.
- Reduce financial risk for lenders.
- Build a Machine Learning classification model.

## Dataset
The dataset contains customer loan information such as:
- LoanID
- Age
- Income
- LoanAmount
- CreditScore
- MonthsEmployed
- NumCreditLines
- InterestRate
- LoanTerm
- DTIRatio
- Education
- EmploymentType
- MaritalStatus
- HasMortgage
- HasDependents
- LoanPurpose
- HasCoSigner
- Default (Target Variable)

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Machine Learning Model
- Logistic Regression

## Project Structure

Loan_Default_Risk_Prediction/
│── data/
│── models/
│── reports/
│── images/
│── train_model.py
│── requirements.txt
│── README.md

## How to Run
1. Install the required libraries:
   pip install -r requirements.txt

2. Run the model:
   python train_model.py

## Output
- Trained Machine Learning model (`model.pkl`)
- Loan default prediction model

## Future Scope
- Deploy using Streamlit.
- Improve model accuracy using advanced algorithms.
- Add data visualization dashboard.
- Integrate with real-time loan applications.

## Author
Kanika Jain