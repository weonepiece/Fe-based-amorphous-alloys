import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import xgboost as xgb
import joblib
from sklearn.model_selection import GridSearchCV

# Data loading
data = pd.read_csv('Dataset path/Bs.csv')

X = data.drop(['Zr', 'Al','V', 'Co', 'magnetic saturation (T)'],axis=1)
y = data.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

# Set parameters
Bs_model = xgb.XGBRegressor(colsample_bytree= 1,
                         learning_rate= 0.5,
                         max_depth= 4,
                         n_estimators= 200,
                         reg_alpha= 0,
                         reg_lambda= 1,
                         subsample= 1.0,
                         random_state=42)

# Model training and evaluation
Bs_model.fit(X_train, y_train)

y_pred = Bs_model.predict(X_test)
r2_score_test = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(r2_score_test)
print(mse)
print("rmse:", rmse)
# # ------------------------------------------------

# save model
# joblib.dump(Bs_model, 'Bs_model.pkl')

