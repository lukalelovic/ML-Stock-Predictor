from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Load the data into a data frame
df = pd.read_csv('all_stocks_5yr.csv')

# Clean and organize the data
df.dropna(inplace=True) # Remove missing value row/cols (if any)
df['date'] = pd.to_datetime(df['date']) 
df.set_index('date', inplace=True)

# Separate date into columns for prediction
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['year'] = df.index.year-2000

X = df[['day_of_week', 'month', 'year', 'open']].values # Prediction values
Y = df['close'] # Value to predict

# Add weights to the prediction values (prioritize day of week)
X_weights = [3, 1, 3, 1]
X = np.multiply(X, X_weights)

# Split the data into a training set and a test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize model to linear regression
model = LinearRegression()

# Train the model on training data
model.fit(X_train, Y_train)

# Apply predictions on test data
Y_pred = model.predict(X_test)

# Calculate model performance
r2 = model.score(X_test, Y_test)
mse = mean_squared_error(Y_test, Y_pred)

print('--Stock Value Predictor--')
print(f'Coefficient of determination (R^2): {r2:.2f}')
print(f'Mean squared error: {mse:.2f}')
print()

# Prompt for input values
print('Enter day of week (0=Mond, 1=Tues, etc)')
day_of_week = int(input())

print('Enter month (1-12)')
month = int(input())

print('Enter year')
year = int(input())-2000

print('Enter opening market price')
open_price = float(input())

closing_value = model.predict([ [day_of_week, month, year, open_price] ])
print(f'Predicted Closing Stock Value: {float(closing_value):.2f}')