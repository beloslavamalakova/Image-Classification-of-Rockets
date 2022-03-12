import os
import numpy as np   
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from dataprocessing.py import test
from sklearn import linear_model

test.info()
test.head()

sns.catplot(x = 'perimeter_mean', y = 'type rocket', data = test, order=['1 cluster', '0 grad'])
test.head()

X,y = test[['rocket']], test[['type']]

model = linear_model.LinearRegression()
model.fit(X, y)
preds = model.predict(X)

sns.scatterplot(x='rocket', y='type', data=test)
plt.plot(X, preds, color='r')
plt.legend(['Linear Regression Fit', 'Data'])

boundary = 15 
sns.catplot(x = 'rocket', y = 'type', data = test, order=['1 (cluster)', '0 (grad)'])
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
y_pred=boundary_classifier(chosen_boundary, test['perimeter_mean'])
test['predicted']=y_pred
y_true=test['type']
sns.catplot(x='perimeter_mean', y='type', hue='predicted', data=test, order=['1 (cluster)', '0(grad)'])
plt.plot([chosen_boundary, chosen_boundary], [-.2, 1.2], 'g', linewidth = 2)
plt.show()

chosen_boundary = 100

y_pred = boundary_classifier(chosen_boundary, test['perimeter_mean'])
test['predicted'] = y_pred

y_true = test['type']

sns.catplot(x = 'perimeter_mean', y = 'diagnosis_cat', hue = 'predicted', data = test, order=['1 (cluster)', '0 (grad)'])
plt.plot([chosen_boundary, chosen_boundary], [-.2, 1.2], 'g', linewidth = 2)
plt.show()

print (list(y_true))
print (y_pred)

accuracy = accuracy_score(y_true,y_pred)
print(accuracy)

