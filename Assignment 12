```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import plotly.express as px

# 1. Create Sample Data
months = pd.date_range('2024-01-01', periods=30, freq='MS')
rng = np.random.default_rng(42)

temperature = 25 + 8 * np.sin(
    2 * np.pi * (np.arange(30) - 2) / 12
) + rng.normal(0, 1, 30)

population = 100000 + np.arange(30) * 500

units = (
    160
    + 5 * np.arange(30) / 12
    + 3.8 * np.maximum(temperature - 24, 0)
    + 0.00005 * (population - 100000)
    + rng.normal(0, 9, 30)
)

# 2. Create DataFrame
df = pd.DataFrame({
    'Month': months,
    'Temperature_C': np.round(temperature, 1),
    'Population': population,
    'Units_kWh': np.round(units, 1)
})

# Save data
df.to_csv('electricity_consumption.csv', index=False)

# 3. Display Data
print("\nFirst Five Records:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# 4. Statistical Analysis
print("\nAverage Electricity Consumption:",
      round(df['Units_kWh'].mean(), 2), "kWh")

print("Standard Deviation:",
      round(df['Units_kWh'].std(), 2))

highest = df.loc[df['Units_kWh'].idxmax()]
lowest = df.loc[df['Units_kWh'].idxmin()]

print("Highest Consumption Month:",
      highest['Month'].strftime('%Y-%m'))

print("Lowest Consumption Month:",
      lowest['Month'].strftime('%Y-%m'))

# 5. Correlation Analysis
print("\nCorrelation Analysis:")

correlation, p_value = pearsonr(
    df['Temperature_C'],
    df['Units_kWh']
)

print("Temperature Correlation:", round(correlation, 3))
print("Temperature P-value:", round(p_value, 4))

correlation_pop, p_value_pop = pearsonr(
    df['Population'],
    df['Units_kWh']
)

print("Population Correlation:", round(correlation_pop, 3))
print("Population P-value:", round(p_value_pop, 4))

# 6. Electricity Consumption Graph
plt.figure(figsize=(10, 5))

plt.plot(
    df['Month'],
    df['Units_kWh'],
    marker='o'
)

plt.title('Monthly Electricity Consumption')
plt.xlabel('Month')
plt.ylabel('Electricity Consumption (kWh)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7. Correlation Heatmap
plt.figure(figsize=(7, 5))

sns.heatmap(
    df[['Temperature_C', 'Population', 'Units_kWh']].corr(),
    annot=True,
    cmap='Blues',
    vmin=-1,
    vmax=1
)

plt.title('Temperature, Population and Electricity Correlation')
plt.tight_layout()
plt.show()

# 8. Create High/Low Consumption Categories
threshold = df['Units_kWh'].median()

df['Consumption_Level'] = (
    df['Units_kWh'] >= threshold
).astype(int)

# 0 = Low, 1 = High

print("\nConsumption Category Counts:")
print(df['Consumption_Level'].value_counts())

# 9. Prepare Machine Learning Data
df['Month_Index'] = np.arange(len(df))

features = [
    'Temperature_C',
    'Population',
    'Month_Index'
]

# First 24 months for training
train = df.iloc[:24]

# Last 6 months for testing
test = df.iloc[24:]

X_train = train[features]
y_train = train['Consumption_Level']

X_test = test[features]
y_test = test['Consumption_Level']

# 10. Train Logistic Regression Model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42)
)

model.fit(X_train, y_train)

# 11. Predict Consumption Category
predictions = model.predict(X_test)

print("\nActual vs Predicted Consumption Level:")

comparison = test[['Month', 'Units_kWh']].copy()

comparison['Actual_Level'] = y_test.map({
    0: 'Low',
    1: 'High'
}).values

comparison['Predicted_Level'] = pd.Series(
    predictions, index=test.index
).map({
    0: 'Low',
    1: 'High'
}).values

print(comparison.to_string(index=False))

# 12. Evaluate Model
accuracy = accuracy_score(y_test, predictions)

print("\nModel Evaluation:")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=['Low', 'High'],
    zero_division=0
))

# 13. Confusion Matrix
cm = confusion_matrix(y_test, predictions, labels=[0, 1])

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Low', 'High'],
    yticklabels=['Low', 'High']
)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Level')
plt.ylabel('Actual Level')
plt.tight_layout()
plt.show()

# 14. Actual vs Predicted Categories
plt.figure(figsize=(10, 5))

plt.plot(
    test['Month'],
    y_test,
    marker='o',
    label='Actual Level'
)

plt.plot(
    test['Month'],
    predictions,
    marker='s',
    label='Predicted Level'
)

plt.title('Actual vs Predicted Consumption Level')
plt.xlabel('Month')
plt.ylabel('Consumption Level (0=Low, 1=High)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 15. Interactive Plotly Graph
fig = px.line(
    df,
    x='Month',
    y='Units_kWh',
    markers=True,
    title='Interactive Monthly Electricity Consumption'
)

fig.show()

# 16. Predict Next Month's Consumption Category
next_temp = float(
    input("\nEnter estimated temperature for next month (C): ")
)

next_population = int(
    input("Enter estimated population for next month: ")
)

next_index = len(df)

future_data = pd.DataFrame({
    'Temperature_C': [next_temp],
    'Population': [next_population],
    'Month_Index': [next_index]
})

next_prediction = model.predict(future_data)[0]

probability = model.predict_proba(future_data)[0]

if next_prediction == 1:
    print("\nPredicted Consumption Level: HIGH")
else:
    print("\nPredicted Consumption Level: LOW")

print("Probability of Low Consumption:",
      round(probability[0] * 100, 2), "%")

print("Probability of High Consumption:",
      round(probability[1] * 100, 2), "%")

print("\nNote: This project uses illustrative sample data.")
print("Use actual electricity, temperature and population data")
print("for reliable real-world analysis.")
```