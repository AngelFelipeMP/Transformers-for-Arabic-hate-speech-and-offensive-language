import config 
from utils import download_data, process_data

if __name__ == "__main__":
    download_data(config.DATA_PATH,
                    config.DATA_URL)
    
    
    process_data(config.DATA_PATH, 
                    config.DATASET_COLUMNS, 
                    config.DATASET_TEXT, 
                    config.DATASET_CLASSES, 
                    config.DATASET_INDEX, 
                    DATASET_TEXT_PROCESSED
                    )

#TODO testar data.py
#TODO testar utils.py