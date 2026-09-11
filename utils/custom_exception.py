import sys


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """Extracts file name, line number, and error message from the traceback."""
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = (
            f"Error occurred in python script: [{file_name}] "
            f"at line number: [{line_number}] "
            f"error message: [{str(error)}]"
        )
    else:
        error_message = f"Error: [{str(error)}]"
    return error_message


class CustomException(Exception):
    """Custom exception class with formatted error detail."""

    def __init__(self, error_message: str | Exception, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message
