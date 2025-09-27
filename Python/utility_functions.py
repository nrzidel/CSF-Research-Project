import logging
import os
from datetime import datetime
from configparser import ConfigParser

def get_config() -> ConfigParser:
    config = ConfigParser()
    config.read('Python/config.ini')
    return config

class Logger:
    def __init__(
            self,
            level=logging.INFO,

        ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)

        if not self.logger.handlers:
            now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            os.makedirs(f'logs/{now}.log')

            formatter = logging.Formatter(
                fmt='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            file_handler = logging.FileHandler(f'Python/logs/{now}.log')
            file_handler.level = logging.DEBUG
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)

            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

            
    def info(self, message): self.logger.info(message)
    def debug(self, message): self.logger.debug(message)
    def warning(self, message): self.logger.warning(message)
    def error(self, message): self.logger.error(message)
    def critical(self, message): self.logger.critical(message)

