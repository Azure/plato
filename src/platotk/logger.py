import logging
import os

logFormatter = "[%(asctime)s][%(module)s][%(levelname)s] %(message)s"
logging.basicConfig(format=logFormatter, datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

log_level = os.getenv("LOG_LEVEL")

if log_level is None:
    log.setLevel(level=logging.INFO)
else:
    log.setLevel(level=log_level.upper())
