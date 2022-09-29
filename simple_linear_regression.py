# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 20:02:35 2022

@author: 47483
"""

# A Simple Linear Regression on Salary vs Job_Experience

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv(r'C:\Study\Data Science\JobExpYears_Salary.csv')

#Matrix of features (Independent Variables) 
x = dataset.iloc[:, :-1].values

#Vector of the dependent variable
y = dataset.iloc[:, -1].values


# split the modelling dataset into the training and testing sets 
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
linear_regressor_obj = LinearRegression()
#fit = capturing the patterns from the provided data
linear_regressor_obj.fit(x_train, y_train)

# Predicting the Testing set results
y_predict = linear_regressor_obj.predict(x_test)

# Visualising the Training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, linear_regressor_obj.predict(x_train), color = 'green')
plt.title('Salary vs Job_Experience (Training set)')
plt.xlabel('Years of Job Experience')
plt.ylabel('Salary')
plt.show()

# Visualising Testing set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, linear_regressor_obj.predict(x_train), color = 'green')
plt.title('Salary vs Job_Experience (Testing set)')
plt.xlabel('Years of Job Experience')
plt.ylabel('Salary')
plt.show()
