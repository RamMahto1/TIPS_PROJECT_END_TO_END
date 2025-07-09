import os
import pickle
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    recall_score,
    precision_score,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def saved_obj(file_path, obj):
    """
    Save a Python object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_metric(X_train, y_train, X_test, y_test, params, models):
    """
    Evaluate multiple classification models using GridSearchCV.
    Returns evaluation report and best model details.
    """
    report = []
    best_score = float('-inf')
    best_model_name = None
    best_model = None

    for model_name, model in models.items():
        logging.info(f"Training model: {model_name}")
        param = params.get(model_name, {})
        gs = GridSearchCV(model, param_grid=param, cv=3, n_jobs=-1, scoring="f1_weighted")
        gs.fit(X_train, y_train)

        # Prediction
        y_pred = gs.predict(X_test)

        # Evaluation
        accuracy_sc = accuracy_score(y_test, y_pred)
        f1_sc = f1_score(y_test, y_pred, average='weighted')
        precision_sc = precision_score(y_test, y_pred, average='weighted')
        recall_sc = recall_score(y_test, y_pred, average='weighted')
        classification_report_sc = classification_report(y_test, y_pred)
        confusion_matrix_sc = confusion_matrix(y_test, y_pred)

        # Logging
        logging.info(f"Model: {model_name}")
        logging.info(f"Accuracy: {accuracy_sc}")
        logging.info(f"F1 Score: {f1_sc}")
        logging.info(f"Precision: {precision_sc}")
        logging.info(f"Recall: {recall_sc}")
        logging.info(f"Confusion Matrix:\n{confusion_matrix_sc}")
        logging.info(f"Classification Report:\n{classification_report_sc}")

        # Save best model
        if f1_sc > best_score:
            best_score = f1_sc
            best_model_name = model_name
            best_model = gs.best_estimator_

        # Append metrics to report list
        report.append({
            "Model": model_name,
            "Accuracy": accuracy_sc,
            "F1 Score": f1_sc,
            "Precision": precision_sc,
            "Recall": recall_sc
        })

    return report, best_model_name, best_model, best_score

def load_obj(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)