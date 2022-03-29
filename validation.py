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


def higher(series):
    outputs = series.tolist()
    sum_outputs = [sum(x) for x in zip(*outputs)]
    return sum_outputs.index(max(sum_outputs))


def majority(series):
    outputs = series.tolist()
    vote = {item : 0 for item in set(outputs)}
    for uni in set(outputs):
        for out in outputs:
            if out == uni:
                vote[uni] += 1
        
    max_val = max(vote.values())
    keys = [k for k,v in vote.items() if v == max_val]
    choice = random.choice(keys)
    return choice


def transformer_parameters(task, transformer, domain):
    for file in os.listdir(config.LOGS_PATH):
        if all(item in file for item in [task, transformer.split("/")[-1], domain]):

            return {'weights': config.LOGS_PATH + '/' + file,
                    'batch_size':int(file.split(']')[4].split('[')[1]),
                    'max_len': int(file.split(']')[3].split('[')[1]),
                    'dropout': float(file.split(']')[5].split('[')[1])}


def validation(df_val, task, transformer):
    parameters = transformer_parameters(task, transformer, config.DOMAIN_TRAIN)
    
    val_dataset = dataset.TransformerDataset(
        text=df_val[config.DATASET_TEXT_PROCESSED].values,
        target=df_val[task].values,
        max_len=parameters['max_len'],
        transformer=transformer
    )

    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=parameters['batch_size'], 
        num_workers=config.VAL_WORKERS
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransforomerModel(transformer, parameters['dropout'], number_of_classes=df_val[task].max()+1)
    model.load_state_dict(torch.load(parameters['weights']))
    model.to(device)
    
    pred_val, targ_val = engine.predict_fn(val_data_loader, model, device)
    
    return pred_val, targ_val


if __name__ == "__main__":
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    dfx = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_DEV, sep='\t', nrows=config.N_ROWS).fillna("none")
    
    for task in tqdm(config.LABELS, desc='VALIDATION', position=0):
        
        df_val = dfx.loc[dfx[task]>=0]
        
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
        
        dfx = pd.merge(dfx, df_val.loc[:, df_val.columns.difference(dfx.columns[1:])], how='left', on='index')
        
    dfx = dfx.fillna(-1)
    
    dfx.to_csv(config.LOGS_PATH + '/' + config.DOMAIN_VALIDATION + '.csv', index=False)

    
    metric_col = [col for col in dfx.columns if any(item in col for item in config.LABELS) ]
    metric_col = [col for col in metric_col if '_outputs' not in col]
    
    metric_dic = {'model':[], 'accuracy':[], 'f1_macro':[]}
    
    for task in config.LABELS:
        for col in metric_col:
            if task in col and task != col:
                metric_dic['model'].append(col)
                metric_dic['f1_macro'].append(metrics.f1_score(dfx[task], dfx[col], average='macro'))
                metric_dic['accuracy'].append(metrics.accuracy_score(dfx[task], dfx[col]))
            
    
    df_metrics = pd.DataFrame.from_dict(metric_dic)
    df_metrics.to_csv(config.LOGS_PATH + '/' + config.DOMAIN_VALIDATION + '_metrics' + '.csv', index=False)

    print('\n')
    print(df_metrics.to_markdown())