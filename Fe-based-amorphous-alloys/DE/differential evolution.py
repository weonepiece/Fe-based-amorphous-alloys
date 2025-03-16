from scipy.optimize import differential_evolution, LinearConstraint
import numpy as np
import joblib
import pandas as pd

# load model
model1 = joblib.load('Storage path of the pre-trained model/GS_model.pkl')
model2 = joblib.load('Storage path of the pre-trained model/RT_model.pkl')
model3 = joblib.load('Storage path of the pre-trained model/Bs_model.pkl')
model4 = joblib.load('Storage path of the pre-trained model/Hc_model.pkl')
model5 = joblib.load('Storage path of the pre-trained model/Tx1_model.pkl')
model6 = joblib.load('Storage path of the pre-trained model/Tx2_model.pkl')

# Defining model features
model_features = {
    'model1': ['Fe', 'Si', 'Al', 'B', 'P', 'Cu', 'Ag', 'V', 'Nb', 'Mo', 'Gd', 'TA', 'tA'],
    'model2': ['Fe', 'Si', 'Al', 'B', 'P', 'Cu', 'Ag', 'Ti', 'V', 'Zr', 'Nb', 'Mo', 'TA', 'tA'],
    'model3': ['Fe', 'Si', 'B', 'Cu', 'Nb', 'Mo', 'C', 'P', 'TA', 'tA'],
    'model4': ['Fe', 'Si', 'B', 'Cu', 'Nb', 'Mo', 'C', 'P', 'V', 'TA', 'tA'],
    'model5': ['Fe', 'Si', 'B', 'Cu', 'Nb', 'Mo', 'Zr', 'Al', 'C', 'P', 'V', 'Hf', 'U', 'Ga', 'Ge', 'Gd', 'Ag', 'HR'],
    'model6': ['Fe', 'Si', 'B', 'Cu', 'Nb', 'Zr', 'Al', 'C', 'P', 'Hf', 'W', 'Ag', 'HR']
}


# Objective function
def objective_function(params):
    # The element composition is 0.5 in step
    rounded_params = np.round(params / 0.5) * 0.5

    # Create a complete feature dictionary
    feature_values = {
        'Fe': rounded_params[0], 'B': rounded_params[1], 'Cu': rounded_params[2], 'Nb': rounded_params[3],
        'P': rounded_params[4],
        'Si': 0, 'Al': 0, 'Ag': 0, 'V': 0, 'Mo': 0, 'Gd': 0, 'Ti': 0, 'Zr': 0, 'C': 0,
        'Hf': 0, 'W': 0, 'U': 0, 'Ga': 0, 'Ge': 0, 'TA': rounded_params[5], 'tA': rounded_params[6],
        'HR': rounded_params[7]
    }


    models = [model1, model2, model3, model4, model5, model6]
    predictions = []

    for i, model in enumerate(models):
        feature_list = model_features[f'model{i + 1}']
        input_data = pd.DataFrame([[feature_values[feat] for feat in feature_list]], columns=feature_list)
        pred = model.predict(input_data)[0]
        predictions.append(pred)

    pred1, pred2, pred3, pred4, pred5, pred6 = predictions

    # Objective function calculation (penalty term)
    penalty = 0
    if pred1 >= 3: penalty += (pred1 - 3) * 100
    if pred2 >= 3: penalty += (pred2 - 3) * 100
    if pred3 <= 1.7: penalty += (1.7 - pred3) * 100
    if pred4 >= 1.8: penalty += (pred4 - 1.8) * 100
    if pred5 >= 6.522: penalty += (pred5 - 6.5) * 100
    if pred6 <= 6.734: penalty += (6.7 - pred6) * 100

    return penalty + sum(predictions)


# Constraint condition (make sure the sum of the ingredients is 100)
linear_constraint = LinearConstraint(
    [[1, 1, 1, 1, 1, 0, 0, 0]],  # The first five elements feature
    [100],  # lower bound
    [100]  # Upper bound
)

# Search area
bounds = [
    (82, 86),  # Fe
    (9, 13),  # B
    (0.5, 1),  # Cu
    (1, 1.6),  # Nb
    (1, 5),  # P
    (673, 673),  # TA
    (150, 150),  # tA
    (30, 30)  # HR
]

# optimize
result = differential_evolution(
    objective_function, bounds, constraints=(linear_constraint,), disp=True, seed=42, popsize=50
)

# output
final_params = np.round(result.x / 0.5) * 0.5
best_score = result.fun
print('Optimal feature combination:', final_params)
print('The corresponding penalty value:', best_score)
