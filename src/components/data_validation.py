from src.logger import logging
from src.exception import CustomException
import pandas as pd
import sys


class DataValidation:
    def __init__(self,train_data,test_data):
        self.train_data = train_data
        self.test_data = test_data
        
    def initiate_data_validation(self):
        try:
            train_df = pd.read_csv(self.train_data)
            test_df = pd.read_csv(self.test_data)
            
            # checking shape of dataset
            logging.info(f"Train data shape: \n{train_df.shape}")
            logging.info(f"test data shape: \n{test_df.shape}")
            
            # checking null value
            logging.info(f"Train data null value: \n{train_df.isnull().sum()}")
            logging.info(f"test data null value: \n{test_df.isnull().sum()}")
            
            # checking duplicate value
            logging.info(f"train data duplicate value: \n{train_df.duplicated().sum()}")
            logging.info(f"test data duplicate value: \n{test_df.duplicated().sum()}")
            
            # checking Statistics summary
            logging.info(f"train data Statistics summary: \n{train_df.describe()}")
            logging.info(f"test data Statistics summary: \n{test_df.describe()}")
            
            expected_columns = ['total_bill','sex','smoker','day','time','size','tips_label']
            missing_columns = [col for col in expected_columns if col not in train_df.columns]
            
            if missing_columns:
                raise CustomException(f"Missing column in train data:{missing_columns}",sys)
            
            return train_df, test_df
        except Exception as e:
            raise CustomException(e,sys)
    