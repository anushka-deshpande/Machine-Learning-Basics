# IMPORTING THE LIBRARIES
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# LOADING THE DATASET
df = pd.read_csv('Dataset_Diabetes.csv')
# print head
print("Head: \n", df.head())

X = df.drop('Outcome',axis = 1)
y = df['Outcome']

# DISTRIBUTION OF DATASET INTO TRAINING AND TESTING SETS
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# TRAIN THE MODEL
from sklearn.tree import DecisionTreeClassifier
my_model = DecisionTreeClassifier(random_state=0)
result = my_model.fit(X_train,y_train)

# TEST THE MODEL
predictions = result.predict(X_test)
print("\nPredictions: \n", predictions)

print("\nDatatype of Predictions: ", type(predictions))

print("\nClassifier Score: ", round(roc_auc_score(y_test,predictions),5))

from sklearn.metrics import mean_absolute_error
print("\nMean absolute Error: ", mean_absolute_error(y_test, predictions))

print("\nAccuracy Score: ", accuracy_score(y_test, predictions))

from sklearn import metrics
print('\n**Classification Report:\n',metrics.classification_report(y_test,predictions))

# DEPLOY THE MODEL
pred_new = list(result.predict([[6,148,72,35,0,33.6,0.627,50]]))
print("\nNew Predictions: \n", pred_new)

