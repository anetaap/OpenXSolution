import uvicorn

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
    return {"models": "list of models"}


# allow user to choose a model
@app.get("/models/{model_name}")
def read_model(model_name: str):
    return {"model": model_name}


# allow user to post all input features and receive a prediction
@app.post("/models/{model_name}/predict")
def predict():
    return {"prediction": "prediction"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
