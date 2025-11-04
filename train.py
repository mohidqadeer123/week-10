import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Exercise 1 : train and save a linear regression model

# Load Dataset Function
def load_data():
    url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
    df = pd.read_csv(url)
    return df

df =load_data()
df = df.dropna(subset=['100g_USD', 'rating'])

# Set features and target variable
X = df[['100g_USD']]
y = df['rating']

# Train model
model1 = LinearRegression()
model1.fit(X, y)

# Save the model
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model1, f)


