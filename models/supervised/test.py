from linear_regression import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys

df = np.loadtxt(sys.path[1]+'/data/CarPricesPrediction.csv', delimiter=",", dtype=str)

print(df.shape)
X = df[:,:-1]
y = df[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train, 0.01)
y_pred = model.predict(X_test)
print(y_pred)