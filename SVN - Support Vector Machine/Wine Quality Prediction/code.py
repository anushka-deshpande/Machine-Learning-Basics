import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

# read dataset
df = pd.read_csv('Dataset_WineQuality.csv')

# print size of database
print("Size of database:\n" ,df.shape)

# gain info about dataset
print("Infor about dataset: \n", df.info())

# extract first 5 rows from dataset
print("First 5 entries: \n", df.head())

# get info on mean, count, std and quartiles of columns
print("More info: \n", df.describe())

# data preprocessing
# categorise wine quality
# 2-6.5 are "bad" quality
# 6.5-8 are "good" wines
bins = (2,6.5,8) #range type for good and bad
group_names = ['bad','good']
categories = pd.cut(df['quality'],bins,labels=group_names)
df['quality'] = categories

# after categorising
print("Printing number ofvalues for quality: \n", df['quality'].value_counts())

# bar plot of quality vs alcohol
# more alcohol, better wine
sn.barplot(x='quality',y='alcohol',data=df)
plt.show()

# bar plot of quality vs volatile acidity
# less volatile is better
sn.barplot(x='quality',y='volatile acidity', data=df)
plt.show()

X = df.drop(['quality'],axis=1)
Y = df['quality']

# now quality is categorical: good or bad
# so to convert it inot 0 and 1 format
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(Y)
print("y: \n",y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("X_test: \n", X_test)

print("y_test: \n2, ", y_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# training the model

from sklearn.svm import SVC
my_model = SVC(kernel='rbf', random_state=0)
result = my_model.fit(X_train, y_train)

# test model
predictions = result.predict(X_test)
print("Predictions: \n", predictions)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
sn.heatmap(cm, annot=True, fmt='2.0f')
plt.show()

from sklearn import metrics
print("Accuracy: \n", metrics.accuracy_score(y_test, predictions))

# make new predictions
new_pred = list(result.predict([[11.1,0.100,0.99,4,0.99,1,2,0.1,1,0.50,9]]))
print("New prediction 1: \n", new_pred)

new_pred=list(result.predict([[10,0.41,0.45,6.2,0.071,6,14,0.99702,3.21,0.49,11.8]]))
print("New prediction 2: \n", new_pred)
