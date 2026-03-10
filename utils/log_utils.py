import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_logger(name: str = "KAI_Backend", level=logging.INFO):
    """Set up and return a logger with structured JSON formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate logs if logger is already configured
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s',
        rename_fields={"asctime": "timestamp", "levelname": "level", "message": "message"}
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger
