from src.logger import logging
from src.exception import CustomException
import sys



def main():
    try:
        0/1
    except Exception as e:
        raise CustomException(e,sys)

if __name__ == "__main__":
    logging.info("logging has started successfully")
    logging.info("zero divided by 1")
    