from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from src.pipline.predict_pipline import PredictPipline, CarData
from src.exception import CustomException
import sys
from src.logger import logging
import models
from database import SessionLocal, engine




app = FastAPI()

models.Base.metadata.create_all(bind=engine)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Car Price Prediction API!"}

@app.post("/predict")
def predict(data: CarData):
    try:
        # Load the preprocessor and model

        logging.info("Starting prediction pipeline")
        data_df = data.get_data_as_dataframe()
        logging.info("Data received for prediction")
        predict_pipeline = PredictPipline()
        prediction = predict_pipeline.predict(data_df)
        logging.info(f"Prediction successful: {prediction}")
        return {"predicted_price": float(prediction)}
    except Exception as e:
        raise CustomException(e, sys)