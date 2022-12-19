import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data into a data frame
df = pd.read_csv('all_stocks_5yr.csv')

# Clean and organize the data
df.dropna(inplace=True) # Remove missing values
df = df.drop('Name', axis=1)
df['date'] = pd.to_datetime(df['date']) 
df.set_index('date', inplace=True)

# Separate date into columns for prediction
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['year'] = df.index.year-2000

X = df[['day_of_week', 'month', 'year', 'open']].values
Y = df['close']

# Split the data into a training set and a test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Choose a machine learning model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, Y_train)

# Evaluate the model on the test data
Y_pred = model.predict(X_test)
r2 = model.score(X_test, Y_test)
mse = mean_squared_error(Y_test, Y_pred)

print('--Stock Value Predictor--')
print(f'Coefficient of determination (R^2): {r2:.2f}')
print(f'Mean squared error: {mse:.2f}')
print()

# Make predictions on new data
print('Enter day of week (0=Mond, 1=Tues, etc)')
day_of_week = int(input())

print('Enter month (1-12)')
month = int(input())

print('Enter year')
year = int(input())-2000

print('Enter opening market price')
open_price = float(input())

new_data = [ [day_of_week, month, year, open_price] ]
prediction = model.predict(new_data)
print(f'Predicted Closing Stock Value: {float(prediction):.2f}')