from src.logger import logging
import sys

def error_message_details(error,error_details:sys):
    _,_,exc_tb=error_details.exc_info()
    
    filename = exc_tb.tb_frame.f_code.co_filename
    
    
    error_message = "Error message occured in script name [{}] line number [{}] error message [{}]".format(
        filename,exc_tb.tb_lineno,str(error)
    )
    return error_message
    
class CustomException(Exception):
        def __init__(self,error,error_details:sys):
            super().__init__(error)
            self.error_message = error_message_details(error,error_details)
            
        