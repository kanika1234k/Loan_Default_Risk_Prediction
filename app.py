# Import required libraries
import joblib
import streamlit as st
import numpy as np


# Load trained model

model=joblib.load("model/model.pkl")


# App title

st.title("Loan Default Risk Prediction")


# User input

age = st.number_input("Age", min_value=18, max_value=100)

income = st.number_input("Income")

loan_amount = st.number_input("Loan Amount")


# Prediction button

if st.button("Predict"):

    input_data = np.array([[age, income, loan_amount]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("High Risk: Loan Default")
    else:
        st.success("Low Risk: No Default")