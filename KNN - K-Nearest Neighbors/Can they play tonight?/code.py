import pandas as pd
df = pd.read_csv('Database_Play.csv')
print("Head: \n", df.head())

X = df.drop('play', axis=1)
y = df['play']

print("\nX head: \n", X.head())
print("\ny head: \n", y.head())

from sklearn.preprocessing import LabelEncoder
Encodex = LabelEncoder()
for col in X.columns:
    X[col] = Encodex.fit_transform(X[col])

print("\nNew X head: \n", X.head())

Encodey = LabelEncoder()
y = Encodey.fit_transform(y)

print("\nY:\n", y)

print("\nType of y: ", type(y))
print("Type of X: ", type(X))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nDescription: ", df.describe())

from sklearn.neighbors import KNeighborsClassifier
myModel = KNeighborsClassifier(n_neighbors=1)
result = myModel.fit(X_train, y_train)

predictions = result.predict(X_test)
print("\nTest predictions: \n", predictions)

from sklearn.metrics import classification_report, confusion_matrix

print("\nConfusion Matrix: \n", confusion_matrix(y_test, predictions))
print("\nClassification Report: \n", classification_report(y_test, predictions))

from sklearn import metrics
print("\nAccuracy: ", metrics.accuracy_score(y_test, predictions))

arr1 = [1,2,2,1]
pred_new = result.predict([arr1])
print("\nInput: ", arr1)
print("New Predictions: ", pred_new)

arr2 = [0,1,1,0]
pred_new = result.predict([arr2])
print("\nInput: ", arr2)
print("New Predictions: ", pred_new)

arr3 = [1,2,1,1]
pred_new = result.predict([arr3])
print("\nInput: ", arr3)
print("New Predictions: ", pred_new)

