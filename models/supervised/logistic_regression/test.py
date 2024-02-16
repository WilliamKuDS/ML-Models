from logistic_regression import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression as sklearnLR
import numpy as np
import time

# Load Dataset as Numpy Array
#df = np.loadtxt(sys.path[1]+'/data/heart.csv', delimiter=",", dtype=str)
df = np.loadtxt('heart.csv', delimiter=",", dtype=str)

# Remove First Row, Remove First Column (Index), and Remove Last Column (Y values)
X = df[1:,:-1]
# Remove First Row, Create y variable which is output values
y = df[1:, -1].astype(float)

scaler = MinMaxScaler()

# Normalize the numerical data between 0 and 1
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

mse = np.mean((y_test - y_pred) ** 2)
print(f"Mean Squared Error (MSE): {mse}")

# Sklearn Logisitic Regression
skmodel = sklearnLR()
start = time.time()
skmodel.fit(X_train, y_train, 0.001)
print("Time: {}".format(time.time() - start))
sk_y_pred = skmodel.predict(X_test)

mse = np.mean((y_test - sk_y_pred) ** 2)
# Sklearn Linear Regression MSE
print(f"Mean Squared Error (MSE): {mse}")

