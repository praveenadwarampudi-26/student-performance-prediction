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

# Evaluate model
predictions = model.predict(X)
mse = mean_squared_error(y, predictions)

print("Model trained successfully!")
print("Mean Squared Error:", mse)

# Take user input
print("\n--- Student Performance Prediction ---")
study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))

# Predict
result = model.predict([[study_hours, attendance]])
print("Predicted Final Score:", round(result[0], 2))
