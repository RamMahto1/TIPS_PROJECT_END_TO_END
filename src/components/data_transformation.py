from src.logger import logging
from src.exception import CustomException
import sys
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.utils import saved_obj
import os


@dataclass

class DataTransformationConfig:
    preprocessor_file_path_obj:str = os.path.join("artifacts","preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()
        
        
    def get_data_tranformer_obj(self):
        logging.info("starting data transformation")
        try:
            numerical_feature = ['total_bill','size']
            categorical_feature = ['sex','smoker','day','time']
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("One_Hot_Encoder",OneHotEncoder(handle_unknown="ignore"))
                ]
            )
            
            preprocessor = ColumnTransformer(
                [
                    ("num",num_pipeline,numerical_feature),
                    ("cat",cat_pipeline,categorical_feature)
                ]
            )
            
            logging.info(f"Numerical features: {numerical_feature}")
            logging.info(f"Categorical feature:{categorical_feature}")
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
            
    def initiate_data_transformation(self,train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("read data as Dataframe")
            logging.info(f"Obtaining preprocessor object")
            preprocessor_obj=self.get_data_tranformer_obj()
            
            target_column = ['tips_label']
            
            input_feature_train_df = train_df.drop(columns=target_column)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns=target_column)
            target_feature_test_df = test_df[target_column]
            
            logging.info("applying preprocessor object on training and testing")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            train_arr= np.c_[input_feature_train_arr,target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr,target_feature_test_df]
            
            logging.info(f"saved preprocessor object")
            
            saved_obj(
                file_path=self.data_tranformation_config.preprocessor_file_path_obj,
                obj=preprocessor_obj
            )
            
            return(
                train_arr,test_arr,
                self.data_tranformation_config.preprocessor_file_path_obj
            )
            
            
        except Exception as e:
            raise CustomException(e,sys)