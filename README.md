# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:

Developed by : MAHALAKSHMI  K

Register no: 212222240057

### AIM:

To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:

Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program

### PROGRAM:
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
file_path = '/content/Covid Data - Asia.csv'

data = pd.read_csv(file_path)

# Clean column names by stripping any extra spaces

data.columns = data.columns.str.strip()

# Rename the first column for easier access

data.rename(columns={'Country, Other': 'Country'}, inplace=True)

# Convert 'Total Cases' to a numeric format (remove commas)

data['Total Cases'] = data['Total Cases'].str.replace(',', '').astype(float)

# Prepare the data for trend estimation

X = np.arange(len(data)).reshape(-1, 1)  # Country indices as X

y = data['Total Cases'].values           # Total Cases as y

# Linear Trend Estimation

linear_model = LinearRegression()

linear_model.fit(X, y)

linear_trend = linear_model.predict(X)

# Polynomial (Quadratic) Trend Estimation

poly_features = PolynomialFeatures(degree=2)

X_poly = poly_features.fit_transform(X)

poly_model = LinearRegression()

poly_model.fit(X_poly, y)

poly_trend = poly_model.predict(X_poly)

# Plotting the original data, linear trend, and polynomial trend

plt.figure(figsize=(12, 8))

plt.plot(data['Country'], y, label='Original Data')

plt.plot(data['Country'], linear_trend, label='Linear Trend', linestyle='--')

plt.plot(data['Country'], poly_trend, label='Polynomial Trend (Degree 2)', linestyle=':')

plt.title('Linear and Polynomial Trend Estimation')

plt.xlabel('Country')

plt.ylabel('Total Cases')

plt.xticks(rotation=90)  # Rotate country names for better readability

plt.legend()

plt.grid(True)

plt.show()

### OUTPUT
![Screenshot (557)](https://github.com/user-attachments/assets/07d5d3e3-e8c0-4520-976b-051b5115ea94)


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
