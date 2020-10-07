import pandas as pd
import matplotlib.pyplot as plt

#read dataset
df = pd.read_csv("CarPrice_Assignment.csv")

#data split for training and testing
from sklearn.model_selection import train_test_split
X = df.drop(["car_ID","price"], axis=1)
y = df['price']

#preprocess data
from sklearn.preprocessing import LabelEncoder
Encoder_X = LabelEncoder()
for col in X.columns:
    X[col] = Encoder_X.fit_transform(X[col].astype(str))
Encoder_y=LabelEncoder()
y = Encoder_y.fit_transform(y)

X.to_csv('Xsample.csv')

X_train,X_test,y_train,y_test= train_test_split(X,y, test_size = 0.2,random_state=112)

#train the model
from sklearn.linear_model import LinearRegression

my_model = LinearRegression()
result = my_model.fit(X_train, y_train)

#test the model
predictions = result.predict(X_test)
print("predictions: ", predictions)

#find accuracy score
from sklearn.metrics import r2_score

r2Score = r2_score(y_test,predictions)
print("r2Score: ", r2Score)

#draw a scatter plot
plt.scatter(y_test,predictions)
plt.show()

#new predictions
pred_new=result.predict([[5,2,1,0,1,0,2,0,24,22,8,1,92,0,2,9,5,23,4,23,5,10,8,10]])
print("New Predictions: ", pred_new)
