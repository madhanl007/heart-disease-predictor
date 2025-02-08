import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

import pickle

# Load the trained model
with open("heart_disease_model.pkl", "rb") as file:
    model, scaler = pickle.load(file)


def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, thalach, exang):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, thalach, exang]])
    input_data_scaled = scaler.fit_transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction[0]

st.title("AI-Assisted Heart Disease Risk Estimator")

# User input
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["0 - None", "1 - Mild", "2 - Moderate", "3 - Severe"])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])

# Convert inputs
sex = 1 if sex == "Male" else 0
cp = int(cp[0])
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

if st.button("Predict Heart Disease Risk"):
    risk = predict_heart_disease(age, sex, cp, trestbps, chol, fbs, thalach, exang)
    if risk == 1:
        st.error("High Risk: Consult a Cardiologist Immediately.")
        st.write("**Suggestions:** Maintain a heart-healthy diet, exercise regularly, reduce stress, and monitor your blood pressure.")
    else:
        st.success("Low Risk: Maintain a Healthy Lifestyle.")
        st.write("**Suggestions:** Continue regular checkups, follow a balanced diet, and stay physically active.")

