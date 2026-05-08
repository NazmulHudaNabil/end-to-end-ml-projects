from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
import numpy as np
import pandas as pd
import time
from src.pipline.predict_pipline import PredictPipline, CarData
from src.exception import CustomException
import sys
from src.logger import logging
import models
from database import SessionLocal, engine
from sqlalchemy.orm import Session
from typing import Generator, Annotated





app = FastAPI()


@app.on_event("startup")
def create_tables_with_retry():
    for attempt in range(10):
        try:
            models.Base.metadata.create_all(bind=engine)
            return
        except Exception as error:
            if attempt == 9:
                raise error
            time.sleep(2)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Car Price Prediction API!"}

@app.post("/predict")
def predict(data: CarData, db: Session = Depends(get_db)):
    try:

        logging.info("Starting prediction pipeline")
        data_df = data.get_data_as_dataframe()
        logging.info("Data received for prediction")
        predict_pipeline = PredictPipline()
        prediction = predict_pipeline.predict(data_df)
        logging.info(f"Prediction successful: {prediction}")

        car_data = data.dict()
        car_data["Manufacturer"] = car_data.pop("Manufacture")
        car_data.pop("Prod")
        car_data["Prod_year"] = car_data.pop("Year")
        car_data["Leather_interior"] = car_data.pop("Leather")
        car_data["Fuel_type"] = car_data.pop("Fuel")
        car_data["Gear_box_type"] = car_data.pop("Gear")
        car_data["Drive_wheels"] = car_data.pop("Drive")
        car_data["Engine_volume"] = car_data.pop("Engine")
        car_data["Wheel_position"] = car_data.pop("Wheel")

        db_car = models.Car(
            **car_data,
            predicted_price=float(prediction)
        )
        db.add(db_car)
        db.commit()
        logging.info("Prediction saved to database")                
        return {"predicted_price": float(prediction)}


    except Exception as e:
        raise CustomException(e, sys)
    

@app.get("/cars/{car_id}", status_code=status.HTTP_200_OK)
def get_car(car_id: int, db: Session = Depends(get_db)):
    try:
        car = db.query(models.Car).filter(models.Car.id == car_id).first()
        if not car:
            raise HTTPException(status_code=404, detail="Car not found")
        return car
    except Exception as e:
        raise CustomException(e, sys)