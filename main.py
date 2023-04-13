import uvicorn

from utils import *
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
    return {"Hello": "World"}


# allow user to choose a model
@app.get("/models")
def read_model():
    return get_models()


# allow user to choose a model
@app.get("/models/{model_id}")
def read_model(model_id: int):
    return get_model(model_id)


# allow user to post all input features and receive a prediction
@app.post("/models/{model_name}/predict")
def predict():
    return {"prediction": "prediction"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
