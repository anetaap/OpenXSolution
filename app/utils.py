from keras.models import load_model
from train_model_utils import *

# load the models
hm = pickle.load(open('models/heuristic_model.pkl', 'rb'))
dt = pickle.load(open('models/decision_tree.pkl', 'rb'))
rm = pickle.load(open('models/random_forest.pkl', 'rb'))
nn = load_model('models/neural_network.h5')
feature_names = pickle.load(open('models/feature_names.pkl', 'rb'))

# load models comparison results/models_comparison.csv
models_comparison = pd.read_csv('results/models_comparison.csv', index_col=0)

# load results for each model
hm_results = pd.read_csv('results/Heuristic-Model.csv', index_col=0)
dt_results = pd.read_csv('results/Decision-Tree.csv', index_col=0)
rm_results = pd.read_csv('results/Random-Forest.csv', index_col=0)
nn_results = pd.read_csv('results/Neural-Network.csv', index_col=0)


# define get models function
def get_models():
    return {'heuristic_model': 1, 'decision_tree': 2, 'random_forest': 3, 'neural_network': 4,
            'models_comparison': models_comparison}


# define get model function
def get_model(model_id):
    if model_id == 1:
        return hm_results
    elif model_id == 2:
        return dt_results
    elif model_id == 3:
        return rm_results
    elif model_id == 4:
        return nn_results
    else:
        return {"The model does not exist"}


# define a function that will make predictions on new data
def predict_new_data(model_id, new_data):
    try:
        if model_id == 1:
            return hm(new_data)
        elif model_id == 2:
            model = dt
        elif model_id == 3:
            model = rm
        elif model_id == 4:
            model = nn
        else:
            return {"The model does not exist"}
        # Extract the values from the dictionary and store them in a list
        values = list(new_data.values())
        # Convert the list of values into a 2D array with a single row
        input_array = np.array(values).reshape(1, -1)

        return model.predict(input_array)

    except Exception as e:
        return {"There was an error, sorry"}
