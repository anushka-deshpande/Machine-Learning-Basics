import pandas as pd
df = pd.read_csv('Dataset_Avocado.csv')

print("Tail: \n", df.tail(9))
print("\nShape: ", df.shape)

df = df.drop(['Unnamed: 0', 'Date'], axis=1)
print("\nHead:\n", df.head(3))
print("\nDescription: \n", df.describe())

print("\nType Value Counts: \n", df['type'].value_counts())

from sklearn.preprocessing import LabelEncoder
varMod = ['type','year','region']
le = LabelEncoder()
for i in varMod:
    df[i] = le.fit_transform(df[i])

print("\nHead:\n", df.head(3))

X = df.drop(['type'], axis=1)
y = df['type']

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.fit_transform(Xtest)

from sklearn.svm import SVC
myModel = SVC(kernel='rbf', random_state=0)
result = myModel.fit(Xtrain, ytrain)

predictions = result.predict(Xtest)
print("\nPredictions: \n", predictions)

from sklearn import metrics
print("\nAccuracy: ", metrics.accuracy_score(ytest, predictions))

from sklearn.metrics import confusion_matrix
conf_matrix =confusion_matrix(predictions,ytest)
confusion_df = pd.DataFrame(conf_matrix, index=['Actual 0','Actual 1'], columns=['Predicted 0','Predicted 1'])
print("\nConfusion Dataframe: \n",confusion_df)

new_pred= list(result.predict([[1.33,64236.62,1036.74,54454.85,48.16,8696.87,8603.62,93.25,0.0,0,0]]))
print("\nNew prediction: ", new_pred)
