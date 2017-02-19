# Data Preprocessing Template
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('/Users/f3n1Xx/Documents/PycharmProjects/RegressionModel/Salary_Data.csv')
X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Linear Regression
reg = LinearRegression()
reg.fit(X_train, Y_train)

#Predicting the Test Set Results
sal = reg.predict(X_test)

#Visualising the Training and Test set result
plt.scatter(X_train,Y_train, color = 'red')
plt.plot(X_train, reg.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test,Y_test, color = 'red')
plt.plot(X_train, reg.predict(X_train), color = 'black')
plt.title('Salary vs Experience (Testing Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()