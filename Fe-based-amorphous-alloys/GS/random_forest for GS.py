import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Data loading
data = pd.read_csv('Dataset path/ln(GS).csv')
X = data.drop(['Au', 'W' , 'Zr', 'Cr', 'Pr', 'ln(Grain Diameter (nm))'], axis=1)
y = data.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set parameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Model training and evaluation
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1, error_score='raise')


grid_search.fit(X_train, y_train)


best_rf = grid_search.best_estimator_


predictions = best_rf.predict(X_test)


mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mse)



# output
print("best_param:", grid_search.best_params_)
print("MSE:", mse)
print("rmse:",rmse)
print("R2:", r2)