import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("student_data.csv")

# Features and target
X = data[["study_hours", "attendance"]]
y = data["final_score"]

# Train model
model = LinearRegression()
model.fit(X, y)

# UI
st.title("🎓 Student Performance Prediction System")

st.write("Enter student details to predict final score")

study_hours = st.number_input("Study Hours per Day", min_value=0.0)
attendance = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0)

if st.button("Predict Score"):
    prediction = model.predict([[study_hours, attendance]])
    st.success(f"Predicted Final Score: {prediction[0]:.2f}")
