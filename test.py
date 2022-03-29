import os
import dataset
import engine
import torch
import pandas as pd
import numpy as np
import random
import config
from tqdm import tqdm

from scipy.special import softmax
from model import TransforomerModel
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()
from validation import validation, higher, majority, transformer_parameters


if __name__ == "__main__":
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    df_test = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_TEST, sep='\t')
    
    df_val_metrics = pd.read_csv(config.DATA_PATH + '/' + 'validation_metrics' + '.csv')
    models_dict = {}
    for task in config.LABELS:
        models_dict[task] = df.loc[df['model'].str.contains(task)].sort_values(by=['f1_macro'], ascending=False).reset_index(inplace=True).loc[[0,1],'model'].tolist()
    
    for task in tqdm(config.LABELS, desc='TEST', position=0):
        
        if task == 'C':
            df_val = df_test.loc[df_test[models_dict['C'][0]] == 1]
        else:
            df_val = df_test.copy()
                
        for transformer in config.TRANSFORMERS:
            
            tqdm.write(f'Task {task} - {transformer.split("/")[-1]}')
            
            predictions, targets = validation(df_val, 
                                                task,
                                                transformer
            )
            
            df_val[task + '_' + transformer.split("/")[-1] + '_outputs'] = [softmax(pred).tolist() for pred in predictions]
            df_val[task + '_' + transformer.split("/")[-1] + '_prediction'] = [pred.index(max(pred)) for pred in predictions]

        columns_higher_sum = [col for col in df_val if all(item in col for item in [task, '_outputs'])]
        columns_majority_vote = [col for col in df_val if all(item in col for item in [task, '_prediction'])]
        
        df_val[task + '_higher_sum'] = df_val.loc[:,columns_higher_sum].apply(lambda x: higher(x), axis=1)
        df_val[task + '_majority_vote'] = df_val.loc[:,columns_majority_vote].apply(lambda x: majority(x), axis=1)
        
        df_test = pd.merge(df_test, df_val.loc[:, df_val.columns.difference(df_test.columns[1:])], how='left', on='index')
        
    df_test = df_test.fillna(-1)
    
    df_test.to_csv(config.LOGS_PATH + '/' + config.DOMAIN_TEST + '_all_model' +'.csv', index=False)

    for task in config.LABELS:
        for j, model enumerate(models_dict[task]):
            df_test.loc[:, ['index'] + models_dict[task][j]].to_csv(config.LOGS_PATH + '/' + config.DOMAIN_TEST + '_task_' + task + '_rank_' + j + '_model_' + model + '_.csv', index=False)
    
    


# update env,ylm "add" conda install tabulate













# models_dict = {task:[] for task in config.LABELS}
#     models_dict = dict()
    
#     for task in config.LABELS:
#         for model in df_val_metrics['model'].unique().tolist():
#             if task in model:
#                 models_dict[task].append(model)
    
    
#     for task in config.LABELS:
#         df_val_metrics.
    
    
    
#     # models_dict = {task:{} for task in config.LABELS}
#     for task in config.LABELS:
#         for model in df_val_metrics['model'].unique().tolist():
#             if task in model:
#                 models_dict[task][model] = df_val_metrics.loc[df_val_metrics['model'] == model, 'f1_macro'].values[0]
    
#     ranked_models = dict()
    
#     for task in config.LABELS:
#         for m,r in models_dict[task].items():
#             for top in sort(list(models_dict[task].values()))[-2]
            
#             ranked_models[task]['first'] = sort(list(models_dict[task].values()))[-1]
#             ranked_models[task]['second'] = sort(list(models_dict[task].values()))[-2]
    
    
    
#     # models_dict = {task:{'first':'', 'second':''} for task in config.LABELS}
    
#     for task in config.LABELS:
#         for 
        
    
#     for models in df_val_metrics
    