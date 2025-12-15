import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- Load dataset ---
data = pd.read_csv("student_data.csv")

# Input features and output
X = data[["study_hours", "attendance"]]
y = data["final_score"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Model evaluation
predictions = model.predict(X)
mse = mean_squared_error(y, predictions)

# Streamlit UI
st.title("Student Performance Prediction System")
st.write("Predict a student's final score based on study hours and attendance.")

st.subheader("Enter Student Details:")
study_hours = st.number_input("Study Hours:", min_value=0.0, max_value=24.0, value=3.0)
attendance = st.number_input("Attendance (%):", min_value=0.0, max_value=100.0, value=80.0)

if st.button("Predict Final Score"):
    result = model.predict([[study_hours, attendance]])
    st.success(f"Predicted Final Score: {round(result[0], 2)}")

st.write("---")
st.write(f"Model Mean Squared Error (MSE) on dataset: {round(mse, 2)}")
