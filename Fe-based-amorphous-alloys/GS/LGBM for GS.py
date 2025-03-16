import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor

# Data loading
data = pd.read_csv('Dataset path/ln(GS).csv')
X = data.drop(['Au', 'W' , 'Zr', 'Cr', 'Pr', 'ln(Grain Diameter (nm))'],axis=1)
y = data.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set parameters
param_grid = {
    'num_leaves': [20, 31, 40],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.005, 0.01, 0.05],
    'n_estimators': [100, 200, 300],
    'min_data_in_leaf': [20, 50, 100]
}

# Model training and evaluation
lgbm = LGBMRegressor(random_state=42)
grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)


grid_search.fit(X_train, y_train)


best_lgbm = grid_search.best_estimator_


predictions = best_lgbm.predict(X_test)


mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mse)

# output
print("best_param:", grid_search.best_params_)
print("MSE:", mse)
print("rmse:",rmse)
print("R2:", r2)
