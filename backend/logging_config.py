import logging
from logging.handlers import RotatingFileHandler
import os

LOG_DIR = "/app/logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "api.log")

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=5_000_000,  # 5MB
    backupCount=5
)
handler.setFormatter(formatter)

logger = logging.getLogger("churn_api")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

def get_logger():
    return logger
