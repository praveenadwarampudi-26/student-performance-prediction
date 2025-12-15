import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
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
print("Mean Squared Error:", mse)

# ---- LIVE PREDICTION PART ----
print("\n--- Student Performance Prediction System ---")

study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))

result = model.predict([[study_hours, attendance]])

print("Predicted Final Score:", round(result[0], 2))
