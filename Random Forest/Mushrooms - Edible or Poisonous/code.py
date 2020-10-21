import numpy as np
import pandas as pd
import warnings
warnings.simplefilter("ignore")
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_csv("Dataset_Mushrooms.csv")
df.head()

print("number of null values: \n", df.isnull().sum())

print("Dataset Properties: \n", df.describe())

X=df.drop('class',axis=1) # Predictors
y=df['class'] # Response
print("X-Head: \n", X.head())

from sklearn.preprocessing import LabelEncoder
Encoder_X = LabelEncoder()
for col in X.columns:
    X[col] = Encoder_X.fit_transform(X[col])
Encoder_y = LabelEncoder()
y = Encoder_y.fit_transform(y)

print("X-head: \n", X.head())

X.to_csv('encoded_X_values.csv')
print("y:\n", y)
print("Data Types of X:\n", X.dtypes)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("X-train:\n", X_train)
print("Type of X_train:\n", type(X_train))

from sklearn.ensemble import RandomForestClassifier
my_model = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 42)
result=my_model.fit(X_train, y_train)

predictions = result.predict(X_test)
print("X-test: \n", X_test)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, predictions))

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(predictions,y_test)
confusion_df = pd.DataFrame(conf_matrix, index=['Actual 0','Actual 1'], columns=['Predicted 0','Predicted 1'])
sn.heatmap(confusion_df, cmap='coolwarm', annot=True)
plt.show()

from sklearn import metrics
print('\n**Classification Report:\n',metrics.classification_report(y_test,predictions))

pred_new = result.predict([[5,2,4,1,6,1,0,1,4,0,2,7,7,0,2,1,4,2,3,5.5,7,4]])
print("New Predictions: \n", pred_new)

pred_new = result.predict([[0,2,4,1,6,1,0,1,4,0,2,7,7,0,2,1,4,2,3,5.5,7,4]])
print("New Predictions: \n", pred_new)
