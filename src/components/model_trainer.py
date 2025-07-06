from src.logger import logging
from src.exception import CustomException
from src.utils import saved_obj,evaluate_metric
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import(
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from dataclasses import dataclass
import sys
import os
@dataclass

class ModelTrainerConfig:
    model_obj_file_path: str = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
        
    def initiate_model_trainer(self, train_array,test_array):
        logging.info("spliting data into train test split")
        try:
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]
            
            ## initialize the model
            models = {
                "LogisticRegression":LogisticRegression(),
                "RandomForestClassifier":RandomForestClassifier(),
                "AdaBoostClassifier":AdaBoostClassifier(),
                "GradientBoostingClassifier":GradientBoostingClassifier(),
                "SVC":SVC(),
                "DecisionTreeClassifier":DecisionTreeClassifier()
            }
            params = {
                "LogisticRegression":{
                    'penalty':['l2'],
                    'C':[1.0],
                    'intercept_scaling':[1]
                },
                "RandomForestClassifier":{
                    'n_estimators':[20,50,100],
                    'min_samples_split':[2],
                    'min_samples_leaf':[1]
                    },
                'AdaBoostClassifier':{
                    'n_estimators':[50,100,200],
                    'learning_rate':[0.01,0.02]
                    },
                'GradientBoostingClassifier':{
                    'n_estimators':[20,50,100],
                    'learning_rate':[0.01,0.02],
                    'max_depth':[3,5,7]
                    },
                'SVC':{
                    'C':[0.1],
                    'degree':[3],
                    'kernel':['rbf']
                    },
                'DecisionTreeClassifier':{
                    'criterion':['gini'],
                    'splitter':['best'],
                    'max_depth':[3,5,None],
                    'min_samples_split':[2]
                    }}
            
            ## evaluate all the model
            report,best_model_name,best_model,best_score = evaluate_metric(X_train,y_train,X_test,y_test,params,models)
            
            logging.info(f"Best model found:{best_model_name} with score: {best_score}")
            
            # saving best model 
            saved_obj(
                file_path=self.model_trainer_config.model_obj_file_path,obj=best_model
            )
            return report,best_model_name,best_model,best_score
            
            
            
           
        except Exception as e:
            raise CustomException(e,sys)