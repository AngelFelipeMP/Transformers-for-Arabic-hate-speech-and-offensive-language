import os
import shutil
import pandas as pd
import gdown
from arabert.preprocess import ArabertPreprocessor


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
        


def process_data(data_path, header, text_col, labels_col, index_col, text_col_pro):
    arabic_prep = ArabertPreprocessor("aubmindlab/bert-base-arabertv2")
    files = os.listdir(data_path)
    
    for file in files:
        df = pd.read_csv(file, sep='\t', header=header)
        df[text_col_pro] = df.loc[:, text_col].apply(lambda x: arabic_prep(x))
        for col, labels in labels_col.items():
            df[col] = df.replace(labels, inplace=True)

    df.to_csv(data_path + '/' + file + '_processed' + '.txt', index=False, sep='\t',  index_label=index_col)



def map_labels(df, labels_col):
    for col, labels in labels_col.items():
        df[col] = df.replace({v: k for k, v in labels.items()}, inplace=True)
    return df
    
# TODO return labels and predictions to nouns
