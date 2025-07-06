import os
from src.logger import logging
from src.exception import CustomException
import pickle
import sys

try:
    def saved_obj(file_path,obj):
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
except Exception as e:
    raise CustomException(e,sys)
    