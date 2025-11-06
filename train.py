import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Exercise 1 : train and save a linear regression model

# Dataset Load Function
def load_data():
    url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
    df = pd.read_csv(url)
    return df

# Load Data
df =load_data()
df = df.dropna(subset=['100g_USD', 'rating'])

# Set features and target variable
X = df[['100g_USD']]
y = df['rating']

# Train model
model1 = LinearRegression()
model1.fit(X, y)

# Save the model
with open('model_1.pickle', 'wb') as f:
    pickle.dump(model1, f)


# Exercise 2 : train and save a decision tree regression model

# map roast categories to numerical values
roast_mapping = {
    'Light': 0,
    'Medium-Light': 1,
    'Medium': 2,
    'Medium-Dark': 3,
    'Dark': 4
}

df['roast'] = df['roast'].map(roast_mapping)

# Drop rows with NaN values in 'roast' column
df2 = df.dropna(subset=['100g_USD', 'rating' , 'roast'])

# Prepare features and target variable
X2 = df2[['100g_USD', 'roast']]
y2 = df2['rating']

# Train model
model2 = DecisionTreeRegressor()
model2.fit(X2, y2)

# Save the model
with open('model_2.pickle', 'wb') as f:
    pickle.dump(model2, f)





