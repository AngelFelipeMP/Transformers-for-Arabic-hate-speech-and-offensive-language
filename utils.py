import os
import shutil
import pandas as pd
import gdown
from arabert.preprocess import ArabertPreprocessor
import config

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
        


def process_data(data_path, header, text_col, labels_col, index_col):
    arabic_prep = ArabertPreprocessor("aubmindlab/bert-base-arabertv2")
    files = os.listdir(data_path)
    
    for file in files:
        df = pd.read_csv(file, sep='\t', header=header)
        
        text_col_processed = text_col + '_processed'
        pass_value_config('DATASET_TEXT_PROCESSED', text_col_processed)
        df[text_col_processed] = df.loc[:, text_col].apply(lambda x: arabic_prep(x))
        
        for col, labels in labels_col.items():
            df[col] = df.replace(labels, inplace=True)
    
        dataset_name = '"/' + file[:-4] + '_processed' + '.txt"'
        value = 'DATA_PATH' + ' + ' + dataset_name
        variable = 'DATASET' + ['_TRAIN' if 'train' in file else '_DEV' if 'dev' in file else '_TEST'][0]
        pass_value_config(variable, dataset_name)
        
        df.to_csv(data_path + dataset_name, index=False, sep='\t',  index_label=index_col)


def map_labels(df, labels_col):
    for col, labels in labels_col.items():
        df[col] = df.replace({v: k for k, v in labels.items()}, inplace=True)
    return df
    


def pass_value_config(variable, value):
    with open(config.CODE_PATH + 'config.py', 'r') as conf:
        content = conf.read()
        new = content.replace(variable + ' = ', variable + ' = ' + value)
        
    with open(config.CODE_PATH + 'config.py', 'w') as conf_new:
        conf_new.write(new)