from linear_regression import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression as sklearnLR
import numpy as np
import time

# Load Dataset as Numpy Array
#df = np.loadtxt(sys.path[1]+'/data/CarPricesPrediction.csv', delimiter=",", dtype=str)
df = np.loadtxt('CarPricesPrediction.csv', delimiter=",", dtype=str)

# Get columns of dataset
col_names = df[0]

# Remove First Row, Remove First Column (Index), and Remove Last Column (Y values)
X = df[1:,1:-1]
# Remove First Row, Create y variable which is output values
y = df[1:, -1].astype(float)

# Indices of categorical and numerical columns
categorical_indices = [0, 1, 4]
numerical_indices = [i for i in range(X.shape[1]) if i not in categorical_indices]

# Seperate the categorical and numerical data from each other
categorical_data = X[:, categorical_indices]
numerical_data = X[:, numerical_indices].astype(float)  # Convert numerical data to float

# Initialize One-Hot Encoder function from sklearn
encoder = OneHotEncoder()

# Fit and transform the categorical data
categorical_data_encoded = encoder.fit_transform(categorical_data).toarray()

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Normalize the numerical data between 0 and 1
normalized_numerical_data = scaler.fit_transform(numerical_data)

# Concatenate the numerical and encoded categorical data back together
X = np.concatenate([normalized_numerical_data, categorical_data_encoded], axis=1)

# Split Dataset into train/set, with a 80/20 split, using random state 42 to ensure consistency
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model using the local LinearRegression class in linear_regression.py
model = LinearRegression()

# Fit the dataset to LinearRegression
model.fit(X_train, y_train)

# Predict y values
y_pred = model.predict(X_test)
# Mean Squared Error
mse = np.mean((y_test - y_pred) ** 2)
print(f"Mean Squared Error (MSE): {mse}")

# Sklearn Linear Regression
skmodel = sklearnLR()
start = time.time()
skmodel.fit(X_train, y_train, 0.001)
print("Time: {}".format(time.time() - start))
sk_y_pred = skmodel.predict(X_test)

mse = np.mean((y_test - sk_y_pred) ** 2)
# Sklearn Linear Regression MSE
print(f"Mean Squared Error (MSE): {mse}")