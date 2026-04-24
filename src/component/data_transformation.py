import os
import sys
from src.logger import logging
import numpy as np
from src.exception import CustomException
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from src.utils import save_object

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw data to match the preprocessing done in the EDA notebook."""
        df = df.copy()
        # Levy: '-' represents missing, replace with NaN so imputer handles it
        if 'Levy' in df.columns:
            df['Levy'] = df['Levy'].replace({'-': None})
            df['Levy'] = pd.to_numeric(df['Levy'], errors='coerce')
        # Engine volume: strip 'Turbo' text, then convert to float
        if 'Engine volume' in df.columns:
            df['Engine volume'] = (
                df['Engine volume']
                .astype(str)
                .str.replace(r'\bTurbo\b', '', regex=True)
                .str.strip()
            )
            df['Engine volume'] = pd.to_numeric(df['Engine volume'], errors='coerce')
        return df

    def get_data_transformer_object(self):
        try:
            numerical_columns =['Levy', 'Prod. year', 'Engine volume', 'Cylinders', 'Airbags']
            categorical_columns = ['Manufacturer', 'Category', 'Leather interior', 
                       'Fuel type', 'Gear box type', 'Drive wheels', 
                       'Doors', 'Wheel', 'Color']

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
            return preprocessor

        except Exception as e:
            logging.error(f"Error occurred while creating data transformer object: {str(e)}")
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train DataFrame head: \n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame head: \n{test_df.head().to_string()}")

            logging.info("Obtaining preprocessing object")
            preprocessor_obj = self.get_data_transformer_object()
            logging.info("Preprocessing object obtained successfully")
            target_column_name = "Price"
            drop_columns = [target_column_name, "Model", "ID", "Mileage"]
            input_feature_train_df = train_df.drop(columns=drop_columns)
            target_feature_train_df = train_df[target_column_name]


            input_feature_test_df = test_df.drop(columns=drop_columns)
            target_feature_test_df = test_df[target_column_name]
            logging.info("Applying preprocessing object on training and testing datasets")

            # Apply EDA-style cleaning before the sklearn pipeline
            input_feature_train_df = self.clean_data(input_feature_train_df)
            input_feature_test_df = self.clean_data(input_feature_test_df)

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Convert sparse matrix to dense array if needed
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()
            logging.info("Preprocessing applied successfully on training and testing datasets")

            print("X shape:", input_feature_train_arr.shape)
            print("y shape:", np.array(target_feature_train_df).shape)

            target_feature_train_arr = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_arr = np.array(target_feature_test_df).reshape(-1, 1)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error(f"Error occurred while initiating data transformation: {str(e)}")
            raise CustomException(e, sys)


