import os
import shutil
import pandas as pd
import gdown
from preprocess import ArabertPreprocessor
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
        


def process_OSACT2022_data(data_path, header, text_col, labels_col, index_col, columns_to_read):
    arabic_prep = ArabertPreprocessor("aubmindlab/bert-base-arabertv2", keep_emojis = True)
    files = [f for f in os.listdir(data_path) if 'processed' not in f]
    
    for file in files:
        if 'test' not in file:
            df = pd.read_csv(data_path + '/' + file, sep='\t', header=None, usecols=columns_to_read)
        else:
            df = pd.read_csv(data_path + '/' + file, sep='\t', header=None, usecols=[0,1])
        
        print(df)
        
        if 'train' in file or 'dev' in file:
            df[df.shape[1]+1] = df.iloc[:,-1].apply(lambda x: x if x == 'NOT_HS' else 'HS')
            df = df[df.columns.tolist()[:-2] + df.columns.tolist()[-1:] + df.columns.tolist()[-2:-1]]
            df.columns = header
            df.replace(labels_col, inplace=True)
            print(df.head())
        else:
            df.columns = header[:-3]
            print('@'*20)
            print(header[:-3])

        text_col_processed = text_col + '_processed'
        pass_value_config('DATASET_TEXT_PROCESSED', '\'' +  text_col_processed + '\'')
        df[text_col_processed] = df.loc[:, text_col].apply(lambda x: arabic_prep.preprocess(x))
        print(df.head())
        
        dataset_name =  file[:-4] + '_processed' + '.txt'
        variable = 'DATASET' + ['_TRAIN' if 'train' in file else '_DEV' if 'dev' in file else '_TEST'][0]
        pass_value_config(variable, '\'' + dataset_name + '\'')
        
        df.to_csv(data_path + '/' + dataset_name, index=False, sep='\t',  index_label=index_col)


def pass_value_config(variable, value):
    with open(config.CODE_PATH + '/' + 'config.py', 'r') as conf:
        content = conf.read()
        new = content.replace(variable + ' = ' + "''", variable + ' = ' +  value )
        
    with open(config.CODE_PATH + '/' + 'config.py', 'w') as conf_new:
        conf_new.write(new)


def map_labels(df, labels_col):
    for col, labels in labels_col.items():
        df.replace({col:{number: string for string, number in labels.items()}}, inplace=True)
    return df