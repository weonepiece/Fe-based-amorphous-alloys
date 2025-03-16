import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import joblib

# Data loading
data = pd.read_csv('Dataset path/Tx2.csv')

X = data.drop(['Tx2', 'Mo', 'Co', 'V', 'Y', 'Ge', 'Au', 'Ga'],axis=1)
y = data.iloc[:, -1]
y = np.log(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)
# Set parameters
Tx2_model = xgb.XGBRegressor(
                         learning_rate= 0.05,
                         max_depth= 4,
                         n_estimators= 700,
                         random_state=42)
# Model training and evaluation
Tx2_model.fit(X_train, y_train)

y_pred = Tx2_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2_score_model = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print("mse", mse)
print("rmse:",rmse)
print("R2:", r2_score_model)
# ------------------------------------------------


# save model
# joblib.dump(Tx2_model, 'Tx2_model.pkl')
