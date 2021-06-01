# Author: Brian Bui
# Date: May 1, 2021
# File: studentLRM.py - Student Performance Linear Regression Model
# Desc: predicting grades of a student by using a linear regression model


# importing all of the necessary ML packages
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# importing the Portuguese language course dataset
df = pd.read_csv("student-por.csv", sep=';')

# selecting specific attributes we want to use for this model
df = df[['G1', 'G2', 'G3', 'studytime', 'failures', 'freetime', 'goout', 'health', 'absences']]

# what we are trying to predict: the final grade
outputPrediction = 'G3'

# creating 2 numpy arrays to hold our x and y values
x = np.array(df.drop([outputPrediction], 1))
y = np.array(df[outputPrediction])

# splitting data into testing and training
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# creating a model and implementing Linear Regression
model = LinearRegression().fit(x_train, y_train)

accuracy = model.score(x_test, y_test)

print('Accuracy of prediction:', accuracy) # score above 80% is good
print('\nSlope:', model.coef_)
print('\nIntercept:', model.intercept_)

# predicting the response
prediction = model.predict(x_test)
print('\nPredicted response:', prediction)

# plotting the model with matplotlib

plot = 'G1'
#plot = 'G2'
#plot = 'studytime'
#plot = 'failures'
#plot = 'freetime'
#plot = 'goout'
#plot = 'health'
#plot = 'absences'


plt.scatter(df[plot], df['G3'])
plt.xlabel(plot)
plt.ylabel('Final Grade')
plt.show()