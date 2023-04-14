import uvicorn

from app.utils import *
from app.schemas import *
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def read_root():
    return {"Hello": "Welcome in my task solution!"}


# allow user to choose a model
@app.get("/models")
def read_model():
    return get_models()


# allow user to choose a model
@app.get("/models/{model_id}")
def read_model(model_id: int):
    return get_model(model_id)


# allow user to post all input features and receive a prediction
@app.post("/models/predict")
def predict(data: Data, model_id: int):
    prediction = predict_new_data(model_id, data)
    return {"predicted value": prediction}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
