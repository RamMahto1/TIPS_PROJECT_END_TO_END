from src.logger import logging
from src.exception import CustomException
import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd

@dataclass

class DataIngestionConfig:
    train_data_path:str= os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","data.csv")
    
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        
        
    def initiate_data_ingestion(self):
        try:
            data = os.path.join("notebooks","df.csv")
            df = pd.read_csv(data)
            logging.info("read data as Dataframe")
            
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            logging.info(f"Ensured directory exists: {os.path.dirname(self.data_ingestion_config.train_data_path)}")

            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)
            
            train_set, test_set =train_test_split(df,test_size=0.20,random_state=42)
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            
            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)