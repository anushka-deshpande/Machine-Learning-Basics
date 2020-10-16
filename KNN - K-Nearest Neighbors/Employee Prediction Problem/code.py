import pandas as pd

# acquire the data
df = pd.read_csv('Dataset_Employee.csv')
# print head
print("Head: \n", df.head(5))

# preprocess the data
print('Calculate all missing values: \n', df.isna().sum())
print("\nStatistical Summary: \n", df.describe())

df.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'], axis = 1, inplace = True)
print("After deleting unnecessary columns: \n", df.head(3))

from sklearn.preprocessing import LabelEncoder
var_mod = df.columns
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])

print("After LabelEncoder: \n", df.head(7))

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

# training the model

from sklearn.neighbors import KNeighborsClassifier
my_model = KNeighborsClassifier(n_neighbors=3)
# here we have given value of K=3

result = my_model.fit(X_train, y_train)

# testing the model
predictions = result.predict(X_test)
print("Predictions for testing data: \n", predictions)

print("The Accoracy score is: \n", result.score(X_test, y_test))

# make new prediction
pred_new=list(result.predict([[9,2,312,1,1,0,3,0,1,10,2,0,2,1,1,399,846,9,0,1,0,3,1,6,3,2,2,2,2,2]]))
print("New prediction: \n", pred_new)
