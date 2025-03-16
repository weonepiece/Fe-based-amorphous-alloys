import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib

# data loading
data = pd.read_csv('Dataset path/ln(RT).csv')

X = data.drop(['W', 'Ge', 'Au', 'Hf', 'Cr', 'Ga', 'C', 'ln(Ribbon Thickness (μm))'],axis=1)
y = data.iloc[:, -1]
# y_log = np.log(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

RT_model = xgb.XGBRegressor(colsample_bytree= 1,
                         learning_rate= 0.1,
                         max_depth= 6,
                         n_estimators= 100,
                         reg_alpha= 0,
                         reg_lambda= 1,
                         subsample= 1.0,
                         random_state=42)

# Model training and evaluation
RT_model.fit(X_train, y_train)

y_pred = RT_model.predict(X_test)
r2_score_test = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("mse", mse)
print("rmse:", rmse)
print("R2:", r2_score_test)
# ------------------------------------------------

# save model
# joblib.dump(RT_model, 'RT_model.pkl')

