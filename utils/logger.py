import os
import logging


log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'log.txt')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=log_path)
logger = logging.getLogger(__name__)
# print(log_path)