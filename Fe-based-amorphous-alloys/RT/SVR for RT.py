import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Data loading
data = pd.read_csv('Dataset path/ln(RT).csv')
X = data.drop(['W', 'Ge', 'Au', 'Hf', 'Cr', 'Ga', 'C', 'ln(Ribbon Thickness (μm))'],axis=1)
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set parameters
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'epsilon': [0.1, 0.2, 0.5],
    'kernel': ['rbf']
}

# Model training and evaluation
svr = SVR()
grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)


grid_search.fit(X_train_scaled, y_train)


best_svr = grid_search.best_estimator_


predictions = best_svr.predict(X_test_scaled)


mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mse)

# output
print("best_param:", grid_search.best_params_)
print("MSE:", mse)
print("rmse:", rmse)
print("R2:", r2)