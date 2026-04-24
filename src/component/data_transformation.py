import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from src.component.data_ingestion import DataIngestion


class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['Manufacturer', 'Category', 'Leather interior', 'Fuel type', 'Gear box type', 'Drive wheels', 'Doors', 'Wheel', 'Color']
            categorical_columns = ['Levy', 'Prod. year', 'Engine volume', 'Cylinders', 'Airbags']

            num_transformer = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_transformer = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onhot', OneHotEncoder(drop='first', handle_unknown='ignore'))
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                transformers = [
                    ('num_transformer', num_transformer, numerical_columns),
                    ('cat_transformer', cat_transformer, categorical_columns)
                ]
            )

        except Exception as e:
            logging.error(f"Error occurred while creating data transformer object: {str(e)}")
            raise CustomException(e, sys)