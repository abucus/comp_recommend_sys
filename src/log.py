'''
Created on May 25, 2015

@author: tengmf
'''
import logging,os,json
from logging.config import dictConfig

def prepare_logger_config():
    default_path='E:\local_repo\comp_recommend_sys\src\log_configuration.json'
    default_level=logging.DEBUG
    if os.path.exists(default_path):
        with open(default_path, 'rt') as f:
            config = json.load(f)
        dictConfig(config)
    else:
        logging.basicConfig(filename='E:\local_repo\comp_recommend_sys\log\log.log',level=default_level, 
                            format='%(asctime)s %(message)s')
        
    
def get_logger(name):
    return logging.getLogger(name)
    
prepare_logger_config()