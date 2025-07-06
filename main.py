from src.logger import logging
from src.exception import CustomException
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.utils import saved_obj
from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer





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
    
    
    # Data validation
    validation=DataValidation(train_path,test_path)
    train_df, test_df = validation.initiate_data_validation()
    logging.info("Data validation completed")
    #return train_df,test_df

    # Model Training
    model_trainer = ModelTrainer()
    report, best_model_name, best_model, best_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    logging.info("Model training completed successfully")
    
if __name__ == "__main__":
    main()
    