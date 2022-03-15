import os
import numpy as np   
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from dataprocessing.py import test, train


train.info()
train.head()

sns.catplot(x = 'perimeter_mean', y = 'type rocket', data = train, order=['1 cluster', '0 grad'])
train.head()

X,y = train[['rocket']], train[['type']]

model = linear_model.LinearRegression()
model.fit(X, y)
preds = model.predict(X)

sns.scatterplot(x='rocket', y='type', data=train)
plt.plot(X, preds, color='r')
plt.legend(['Linear Regression Fit', 'Data'])

boundary = 15 
sns.catplot(x = 'rocket', y = 'type', data = train, order=['1 (cluster)', '0 (grad)'])
plt.plot([boundary, boundary], [-.2, 1.2], 'g', linewidth = 2)

def boundary_classifier(target_boundary, radius_mean_series):
  result = []
  for i in radius_mean_series:
    if i>target_boundary:
      result.append(1)
    else:
      result.append(0)

  return result

chosen_boundary=10
y_pred=boundary_classifier(chosen_boundary, train['perimeter_mean'])
train['predicted']=y_pred
y_true=train['type']
sns.catplot(x='perimeter_mean', y='type', hue='predicted', data=train, order=['1 (cluster)', '0(grad)'])
plt.plot([chosen_boundary, chosen_boundary], [-.2, 1.2], 'g', linewidth = 2)
plt.show()

chosen_boundary = 100

y_pred = boundary_classifier(chosen_boundary, train['perimeter_mean'])
train['predicted'] = y_pred

y_true = train['type']

sns.catplot(x = 'perimeter_mean', y = 'diagnosis_cat', hue = 'predicted', data = train, order=['1 (cluster)', '0 (grad)'])
plt.plot([chosen_boundary, chosen_boundary], [-.2, 1.2], 'g', linewidth = 2)
plt.show()

print (list(y_true))
print (y_pred)

accuracy = accuracy_score(y_true,y_pred)
print(accuracy)


X = ['']
y = ''

X_train = train[X]
print('X_train, our input variables:')
print(X_train.head())
print()

y_train = train[y]
print('y_train, our output variable:')
print(y_train.head())

logreg_model = linear_model.LogisticRegression()
logreg_model.fit(X_train, y_train)

X_test = test[X]
y_test = test[y]

y_pred = logreg_model.predict(X_test)

test['predicted'] = y_pred.squeeze()
sns.catplot(x = X[0], y = '', hue = 'predicted', data=test, order=['1', '0'])

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)



