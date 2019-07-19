#Simple linear regression
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pdS638773

#importing the dataset
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 1].values

#splitting thbe dataset into the Training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 1/3, random_state=0)

#featuring scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train  = sc_X.tranform(X_train)
X_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting Simple linear regression to the training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
 
#predicting the test set results
y_pred = regressor.predict(X_test)

#visualising the training set results 
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualising the test set results 
plt.scatter(X_test, y_test, color = 'blue')
plt.plot(X_test, y_pred, color = 'black')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()