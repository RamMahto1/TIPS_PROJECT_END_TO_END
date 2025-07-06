from src.logger import logging
from src.exception import CustomException
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.utils import saved_obj





def main():
    print("Hello project")  # confirmation
    logging.info("Pipeline started")
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    print(f"Train data path: {train_path}")
    print(f"Test data path: {test_path}")
    logging.info("Data Ingestion Completed")
    
    ## step:2 - Data Tranformation
    Data_transformer=DataTransformation()
    train_arr, test_arr,_=Data_transformer.initiate_data_transformation(train_path,test_path)
    logging.info("Data Transformation Completed")
    
    print(f"Data Transformation train object:{train_arr}")
    print(f"Data tranformation test object:{test_arr}")
    

    
    
    
    

if __name__ == "__main__":
    main()
    