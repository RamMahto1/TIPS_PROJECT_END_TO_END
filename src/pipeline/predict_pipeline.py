from src.utils import load_obj
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
import sys


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"
            
            model = load_obj(model_path)
            preprocessor = load_obj(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
