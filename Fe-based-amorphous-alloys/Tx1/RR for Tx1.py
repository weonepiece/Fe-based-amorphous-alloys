import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Data loading
data = pd.read_csv('Dataset path/Tx1.csv')
X = data.drop(['Tx1', 'Co', 'W', 'Au', 'Y'],axis=1)
y = data.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set parameters
param_grid = {
    'alpha': [0.1, 1, 10, 100, 1000]
}

# Model training and evaluation
ridge = Ridge()
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)


grid_search.fit(X_train_scaled, y_train)


best_ridge = grid_search.best_estimator_


predictions = best_ridge.predict(X_test_scaled)


mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mse)

# output
print("best_param:", grid_search.best_params_)
print("MSE:", mse)
print("rmse:",rmse)
print("R2:", r2)