import logging
import sys

def setup_logger():
    # Create a logger
    logger = logging.getLogger('Training')
    logger.setLevel(logging.DEBUG)  # Set the logging level

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a console handler and set the formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Create a file handler and set the formatter
    file_handler = logging.FileHandler('training.log')
    file_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger

# Initialize the logger
logger = setup_logger()