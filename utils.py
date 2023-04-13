import pandas as pd
from keras.models import load_model
import pickle
import numpy as np

# load the models
# hm = pickle.load(open('models/heuristic_model.pkl', 'rb'))
dt = pickle.load(open('models/decision_tree.pkl', 'rb'))
rm = pickle.load(open('models/random_forest.pkl', 'rb'))
# nn = load_model('models/neural_network1.h5')

# load models comparison results/models_comparison.csv
models_comparison = pd.read_csv('results/models_comparison.csv', index_col=0)


# define get models function
def get_models():
    return {'heuristic_model': 1, 'decision_tree': 2, 'random_forest': 3, 'neural_network': 4,
            'models_comparison': models_comparison}


# define get model function
def get_model(model_id):
    if model_id == 1:
        return hm
    elif model_id == 2:
        return dt
    elif model_id == 3:
        return rm
    elif model_id == 4:
        return nn
    else:
        return None


# define a function that will make predictions on new data
def predict_new_data(model, new_data):
    return model.predict(new_data)
