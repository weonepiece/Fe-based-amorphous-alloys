import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib

# Data loading
data = pd.read_csv('Dataset path/Hc.csv')

X = data.drop(['Al', 'Co', 'Zr', 'ln(Coercivity (A/m))'],axis=1)
y = data.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

# Set parameters
Hc_model = xgb.XGBRegressor(colsample_bytree=1,
                         learning_rate=0.4,
                         max_depth=7,
                         n_estimators=300,
                         reg_alpha=0,
                         reg_lambda=1,
                         subsample=1.0,
                         random_state=42)

# Model training and evaluation
Hc_model.fit(X_train, y_train)

y_pred = Hc_model.predict(X_test)
r2_score_test = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# output
print("mse", mse)
print("R2:", r2_score_test)
print("rmse:", rmse)
# # ------------------------------------------------

# save model
# joblib.dump(Hc_model, 'Hc_model.pkl')
