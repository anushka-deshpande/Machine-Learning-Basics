import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv("Mobile_Dataset.csv")
print("Head:\n", df.head())

X = df.drop(['price_range'],axis=1)
y = df['price_range']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.tree import DecisionTreeClassifier
myModel = DecisionTreeClassifier(random_state=0)
result = myModel.fit(X_train, y_train)

predictions = result.predict(X_test)
print("Predictions:\n", predictions)

from sklearn.metrics import mean_absolute_error, accuracy_score
absError = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error is: ", absError)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy Score: ", accuracy)

from sklearn.metrics import classification_report
matrix = classification_report(y_test, predictions, labels=[1,0])
print("Classification report matrix: \n", matrix)

from sklearn.metrics import confusion_matrix
conMatrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix: \n", conMatrix)

confusionDf = pd.DataFrame(matrix, index=['Actual Label 0','Actual Label 1','Actual Label 2','Actual Label 3'], columns=['Predicted Label 0','Predicted Label 1','Predicted Label 2','Predicted Label 3'])
print("Confusion Dataframe: \n", confusionDf)

from sklearn import metrics
print("Classification report:\n", metrics.classification_report(y_test, predictions))

newp1 = list(result.predict([[842,0,2.2,0,1,0,7,0.6,188,2,2,20,756,2549,9,7,19,0,0,1]]))
print("New prediction 1: \n", newp1)

newp2 = list(result.predict([[1821,0,1.7,4,1,10,0.8,139,8,10,381,1018,3220,13,8,18,1,0,1,1]]))
print("New prediction 2:\n", newp2)
