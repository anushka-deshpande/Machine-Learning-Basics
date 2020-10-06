import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data31.csv")

from sklearn.model_selection import train_test_split
X = df.drop("Son", axis = 1)
y = df['Son']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

from sklearn.linear_model import LinearRegression
myModel = LinearRegression()
result = myModel.fit(X_train, y_train)
predictions = result.predict(X_test)

plt.scatter(X_train, y_train, color = 'c')
plt.plot(X_test, predictions, color='b')
plt.show()

print("Enter number for predictions: ")
print("Enter 0 if done")
temp = 1
while temp == 1:
    ht = float(input("Fathers height: "))
    if ht == 0:
        break
    else:
        res = float(result.predict([[ht]])[0])
        print("Predicted Sons height: ", res)
