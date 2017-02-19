# Data Preprocessing Template
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('/Users/f3n1Xx/Documents/PycharmProjects/RegressionModel/Salary_Data.csv')
X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, 1].values

