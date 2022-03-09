import os
import sys 
config_path = '/'.join(os.getcwd().split('/')[0:-1]) + '/' + 'config'
sys.path.insert(1, config_path)
from parameters import DATA_PATH, DATA_URL
from utils import download_data

print(DATA_PATH)
download_data(DATA_PATH, DATA_URL)
    
# #TODO add url to config
# #TODO process data using pip install farasapy