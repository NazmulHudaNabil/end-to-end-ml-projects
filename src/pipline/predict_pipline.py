import os
import sys
from src.logger import logging
from src.exception import CustomException
from pydantic import BaseModel, Field
from typing import List, Optional, Annotated
from src.utils import load_object
import pandas as pd
from src.component.data_transformation import DataTransformation

class PredictPipline:
    def __init__(self):
        pass

    def predict(self, input_data):
        try:
            logging.info("Starting prediction pipeline")
            # Load the preprocessor and model
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Clean data the same way training did (strip "Turbo", convert Engine volume to float, etc.)
            data_transformation = DataTransformation()
            input_data = data_transformation.clean_data(input_data)

            # Transform the input data
            logging.info("Transforming input data")
            data_scaled = preprocessor.transform(input_data)

            # Make prediction
            logging.info("Making prediction")
            prediction = model.predict(data_scaled)

            logging.info(f"Prediction completed: {prediction[0]}")
            return prediction[0]

        except Exception as e:
            logging.error(f"Error occurred during prediction: {str(e)}")
            raise CustomException(e, sys)
        

class CarData(BaseModel):
    Levy:Annotated[float, Field(description="Levy amount for the car", example=500.0)]
    Manufacture:Annotated[str, Field(description="Car manufacturer", example="Toyota")]
    Prod:Annotated[str, Field(description="Production model of the car", example="Corolla")]
    Year:Annotated[int, Field(description="Manufacturing year of the car", example=2015)]
    Category:Annotated[str, Field(description="Category of the car", example="Sedan")]
    Leather:Annotated[str, Field(description="Leather interior (Yes/No)", example="Yes")]
    Fuel:Annotated[str, Field(description="Fuel type of the car", example="Petrol")]
    Gear:Annotated[str, Field(description="Gear box type", example="Automatic")]
    Drive:Annotated[str, Field(description="Drive wheels", example="Front")]
    Engine:Annotated[str, Field(description="Engine volume (e.g., '1.6 Turbo')", example="1.6 Turbo")]
    Cylinders:Annotated[int, Field(description="Number of cylinders in the engine", example=4)]
    Airbags:Annotated[int, Field(description="Number of airbags in the car", example=6)]
    Doors:Annotated[int, Field(description="Number of doors in the car", example=4)]
    Wheel:Annotated[str, Field(description="Wheel position (Left/Right)", example="Left")]
    Color:Annotated[str, Field(description="Color of the car", example="Red")]


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = self.dict()
            df = pd.DataFrame([custom_data_input_dict], index=[0])

            # Rename columns to match the names used during model training
            df = df.rename(columns={
                "Manufacture": "Manufacturer",
                "Year":        "Prod. year",
                "Leather":     "Leather interior",
                "Fuel":        "Fuel type",
                "Gear":        "Gear box type",
                "Drive":       "Drive wheels",
                "Engine":      "Engine volume",
            })

            return df
        except Exception as e:
            logging.error(f"Error occurred while converting input data to DataFrame: {str(e)}")
            raise CustomException(e, sys)
