import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model & training column names
model = joblib.load("obesity_rf_model.pkl")
training_columns = joblib.load("training_columns.pkl")

st.title("🏥 Obesity Risk Prediction App (RandomForest Model)")
st.write("Enter your daily lifestyle details to predict your obesity level.")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120)
height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0)
weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0)

favc = st.selectbox("Do you frequently eat high-calorie food?", ["Yes", "No"])
fcvc = st.slider("Vegetable consumption frequency (1-3)", 1, 3)
ncp = st.slider("Number of main meals per day", 1, 5)
caec = st.selectbox("Food between meals", ["Always", "Frequently", "Sometimes", "No"])
ch2o = st.slider("Daily water intake (liters)", 1, 5)
faf = st.slider("Physical activity frequency per week", 0, 7)
tue = st.slider("Time using electronic devices (hours/day)", 0, 12)
mtrans = st.selectbox("Mode of Transportation", 
                      ["Automobile", "Bike", "Public_Transportation", "Walking"])

# Compute BMI
height_m = height / 100
bmi = weight / (height_m ** 2)

# Create DataFrame for prediction
input_data = pd.DataFrame({
    "Gender": [gender],
    "Age": [age],
    "Height": [height],
    "Weight": [weight],
    "Frequent_HighCalorie_Food_Consumption": [favc],
    "Vegetable_Consumption_Frequency": [fcvc],
    "Number_of_Main_Meals": [ncp],
    "Food_Between_Meals": [caec],
    "Daily_Water_Intake_Liters": [ch2o],
    "Physical_Activity_Frequency": [faf],
    "Time_Using_Electronic_Devices": [tue],
    "Mode_of_Transportation": [mtrans],
    "BMI": [bmi]
})

# One-hot encoding
input_encoded = pd.get_dummies(input_data)

# FIX: Make sure ALL training columns exist
for col in training_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# FIX: Keep only training columns (avoid KeyError)
input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"Predicted Obesity Level: **{prediction}**")
