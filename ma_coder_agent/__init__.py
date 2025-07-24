import logging

logging.addLevelName(
    logging.DEBUG, f"\033[1;36m{logging.getLevelName(logging.DEBUG)}\033[1;0m"
)
logging.addLevelName(
    logging.INFO, f"\033[1;32m{logging.getLevelName(logging.INFO)}\033[1;0m"
)
logging.addLevelName(
    logging.WARNING, f"\033[1;31m{logging.getLevelName(logging.WARNING)}\033[1;0m"
)
logging.addLevelName(
    logging.ERROR, f"\033[1;41m{logging.getLevelName(logging.ERROR)}\033[1;0m"
)
