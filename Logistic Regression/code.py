import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('Dataset33.csv')
print(df.head())
print(df.shape)

print(df.describe())
sns.countplot(x='RainTomorrow', data=df)
plt.show()

sns.boxplot(x='RainTomorrow', y='MaxTemp', data=df)
plt.show()

sns.barplot(x='RainTomorrow', y='MaxTemp', data=df)
plt.show()

df.dropna(inplace=True)

from sklearn.preprocessing import LabelEncoder
category = ['Location', 'WindGustDir','WindDir9am','WindDir3pm','RainToday','RainTomorrow']
le = LabelEncoder()
for i in category:
    df[i] = le.fit_transform(df[i])

print(df.head())

X = df.drop(['Date','RISK_MM','RainTomorrow'], axis=1)
y = df['RainTomorrow']

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=156)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.fit_transform(Xtest)

from sklearn.linear_model import LogisticRegression
myModel = LogisticRegression()
result = myModel.fit(Xtrain, ytrain)
predictions=result.predict(Xtest)
print(predictions)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(ytest, predictions)
print("Accuracy Score is: ", accuracy)

from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(ytest, predictions)
confusion_df = pd.DataFrame(confusion_mat, index=['Actual Negative','Actual Positive'],columns=['Predicted Negative','Predicted Positive'])
print("Confusion Dataframe is: \n", confusion_df)

Color_conf_mat = sns.heatmap(confusion_df, cmap='coolwarm', annot=True)
plt.show()

