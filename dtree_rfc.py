"""
Created on Sat Oct  3 14:35:21 2020

DECISION TREES AND RANDOM FORESTS PROJECT

@author: Marc
"""
# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the data
loans = pd.read_csv('loan_data.csv')

# Visualization on target column
plt.figure(figsize = (15, 6))
sns.countplot(x = 'purpose', data = loans, hue = 'not.fully.paid')
plt.savefig('purpose_notpaid.png')

# Solving categorical data
final_data = pd.get_dummies(data = loans, columns = ['purpose'], drop_first = True)
final_data.info()

# Train and Test split
from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid', axis = 1)
Y = final_data['not.fully.paid']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, Y_train)
dtree_pred = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print('--------------------------------------------------')
print(' > DECISION TREE')
print('Confusion Matrix')
print(confusion_matrix(Y_test, dtree_pred))
print('\n')
print('Classification Report')
print(classification_report(Y_test, dtree_pred))
print('--------------------------------------------------')

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 500)
rfc.fit(X_train, Y_train)
rfc_pred = rfc.predict(X_test)

print(' > RANDOM FOREST')
print('Confusion Matrix')
print(confusion_matrix(Y_test, rfc_pred))
print('\n')
print('Classification Report')
print(classification_report(Y_test, rfc_pred))
