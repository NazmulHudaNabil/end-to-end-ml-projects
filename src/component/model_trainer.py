import os
import sys
from src.logger import logging
from src.exception import CustomException
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.utils import save_object


class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            model = XGBRegressor(
                max_depth=8,
                learning_rate=0.035551653,
                n_estimators=528,
                random_state=42
            )
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_r2_score = r2_score(y_train, y_train_pred)
            test_r2_score = r2_score(y_test, y_test_pred)

            logging.info(f"Training R2 Score: {train_r2_score}")
            logging.info(f"Testing R2 Score: {test_r2_score}")

            if test_r2_score < 0.6:
                raise ValueError(f"Model performance is not satisfactory. Test R2 Score is {test_r2_score:.4f}, which is below 0.6.")

            logging.info("Saving the trained model")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logging.info(f"Model saved successfully at {self.model_trainer_config.trained_model_file_path}")
            print(f"Train R2 Score: {train_r2_score:.4f}")
            print(f"Test  R2 Score: {test_r2_score:.4f}")

            return train_r2_score, test_r2_score

        except Exception as e:
            logging.error(f"Error occurred during model training: {str(e)}")
            raise CustomException(e, sys)