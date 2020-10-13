import pandas as pd
import numpy as np

df = pd.read_csv("Shopping_Dataset.csv")
print("Head:\n", df.head())

print("Is null: \n", df.isna().sum())

df['Administrative']=df['Administrative'].fillna(df['Administrative'].median())
df['Administrative_Duration']=df['Administrative_Duration'].fillna(df['Administrative_Duration'].median())
df['Informational']=df['Informational'].fillna(df['Informational'].median())
df['Informational_Duration']=df['Informational_Duration'].fillna(df['Informational_Duration'].median())
df['ProductRelated']=df['ProductRelated'].fillna(df['ProductRelated'].median())
df['ProductRelated_Duration']=df['ProductRelated_Duration'].fillna(df['ProductRelated_Duration'].median())
df['BounceRates']=df['BounceRates'].fillna(df['BounceRates'].median())
df['ExitRates']=df['ExitRates'].fillna(df['ExitRates'].median())

print("Data types: \n", df.dtypes)

from sklearn.preprocessing import LabelEncoder
catToNum = ['Month','VisitorType','Weekend','Revenue']
le = LabelEncoder()
for i in catToNum:
    df[i] = le.fit_transform(df[i])

print("New Data Head: \n", df.head(3))

print("Description:\n", df.describe())

X = df.drop('Revenue',axis=1)
y = df['Revenue']

print("X tail: \n", X.tail())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
MyModel = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 42)
result = MyModel.fit(X_train, y_train)

predictions = result.predict(X_test)
print("Predictions: \n", predictions)

from sklearn import metrics
print("Accuracy: ", metrics.accuracy_score(y_test, predictions))

import seaborn as sn
from sklearn.metrics import confusion_matrix
conf_matrix =confusion_matrix(predictions,y_test)
confusion_df = pd.DataFrame(conf_matrix, index=['Actual 0','Actual 1'], columns=['Predicted 0','Predicted 1'])
print("Confusion Dataframe: \n", confusion_df)

pred_new = result.predict([[0.0,0.0,0.0,0.0,0.1,0.0,0.5,0.2,0.0,0.0,2,1,1,1,1,2,0]])
print("New prediction 1: ", pred_new)


pred_new = result.predict([[4.0,75.0,0.0,0.0,15.0,346.000000,0.000000,0.021053,0.000000,0.0,7,2,2,3,11,2,0]])
print("New prediction 2: ", pred_new)

