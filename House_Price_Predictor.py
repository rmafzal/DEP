# Import necessary libraries
import random
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn.preprocessing import StandardScaler

# Load the dataset
dataset = pd.read_csv('house_data.csv')

# Display the first few rows and inspect columns
#print(dataset.head())

# Separate features (X) and target (y)
X = dataset.drop('price', axis=1)
y = dataset['price']
#print(X)
#print(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
#print(X_train)
#print(X_test)
#print(y_test)
#print(y_train)

# Standardize features (optional, but recommended for linear regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Linear Regression model
model = LinearRegression()
#model = Ridge(alpha=10)
#model = Lasso(alpha=10)

# Train the model
#model.fit(X_train, y_train)
model.fit(X_train_scaled, y_train)

# Predict on the test set
#y_pred = model.predict(X_test)
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'Mean Absolute Error: {mae:.2f}')

# Example of predicting a single house price (adapt to your dataset)
# Suppose we have a new data point for prediction
random_row = dataset.sample(n=1)
new_house = random_row.drop('price', axis=1)  # Sample Data
actual_price = random_row['price']
print(actual_price)
new_house_scaled = scaler.transform(new_house)
predicted_price = model.predict(new_house_scaled)
accuracy = (predicted_price - actual_price) * 100 / actual_price
#predicted_price = model.predict(new_house)
print(f'Predicted price for the new house: ${predicted_price[0]:,.2f}')
print(f'Actual price for the new house: ${actual_price.values[0]:,.2f}')
print(f'Accuracy in % of the program %{accuracy.values[0]:,.2f}')