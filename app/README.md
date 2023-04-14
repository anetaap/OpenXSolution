## Open X task Solution

This is a solution for the Open X task. It is a simple api to serve machine learning models. 

In the `app` folder you cen find:
- `train_models.py`: script to train a model
- `train_model_utils.py`: utils for the training script, contains all models code, plots and metrics
- `main.py`: main script to run the api
- `utils.py`: utils for the main script, contains all api code
- `requirements.txt`: requirements for the api
- `schems.py`: pydantic schemas for the api
- `Dockerfile`: dockerfile to build the api image

All learning models you can find in the `models` folder.

In folder `images` you can find images of confusion matrix for each model.

In folder `results` you can find results of the models and their comparison.
