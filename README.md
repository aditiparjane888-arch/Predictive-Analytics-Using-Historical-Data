# Predictive-Analytics-Using-Historical-Data
 Predictive Analytics Using Historical Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Sample Historical Sales Data
data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Sales': [120, 135, 150, 165, 180, 195, 210, 225, 240, 255]
}
df = pd.DataFrame(data)
# Features and Target
X = df[['Month']]
y = df['Sales']
# Train Model
model = LinearRegression()
model.fit(X, y)
# Predict Future Sales
future_months = np.array([[11], [12], [13]])
predictions = model.predict(future_months)
# Print Predictions
for month, sale in zip(future_months.flatten(), predictions):
    print(f"Predicted Sales for Month {month}: {sale:.2f}")
# Visualization
plt.scatter(X, y, label="Historical Data")
plt.plot(X, model.predict(X), label="Regression Line")
plt.scatter(future_months, predictions, marker='x', s=100, label="Predictions")
plt.title("Predictive Analytics Using Historical Data")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.show()