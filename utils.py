import os
import shutil
import gdown

def download_data(data_path, data_urls):
    # create a data folder
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)
    
    for url in data_urls:
        #download data folders to current directory
        gdown.download_folder(url, quiet=True)
        sorce_folder = os.getcwd() + '/' + 'data'
        
        # move datasets to the data folder
        file_names = os.listdir(sorce_folder)
        for file_name in file_names:
            shutil.move(os.path.join(sorce_folder, file_name), data_path)
            
        # delete data folders from current directory
        shutil.rmtree(sorce_folder)