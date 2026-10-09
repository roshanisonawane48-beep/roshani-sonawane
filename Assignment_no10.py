import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Dataset: X = Hours Studied, y = Marks Scored
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([35, 42, 50, 58, 65, 71, 78, 83, 90, 96])

# 2. Split dataset using scikit-learn
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predict test set values
y_pred = model.predict(X_test)

# 5. Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Marks per hour (Slope): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.4f}")

# 6. Compare actual and predicted marks
print("\nActual Marks vs Predicted Marks:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted:.2f}")

# 7. Predict marks for a new study duration
hours = np.array([[5]])
predicted_marks = model.predict(hours)
print(f"\nPredicted marks for 5 hours of study: {predicted_marks[0]:.1f}/100")
