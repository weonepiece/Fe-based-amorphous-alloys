from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd
import numpy as np

# Data loading
data = pd.read_csv('Dataset path/Tx2.csv')

X = data.drop(['Tx2', 'Mo', 'Co', 'V', 'Y', 'Ge', 'Au', 'Ga'],axis=1)
y = data.iloc[:, -1]
y = np.log(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and evaluation
elastic_net = ElasticNet(random_state=42)

# Set parameters
param_grid = {
    'alpha': [0.1, 0.05, 1.0, 10.0],
    'l1_ratio': [0.1, 0.05, 0.5, 0.9]
}


grid_search = GridSearchCV(estimator=elastic_net, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)
rmse = np.sqrt(mse)

print(f'Best Model Parameters: {grid_search.best_params_}')
print(f'MSE: {mse}')
print("rmse:",rmse)
print(f'r2: {r2}')