import os
import dataset
import engine
import torch
import pandas as pd
import numpy as np
import random
import config
from tqdm import tqdm

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
    sum_outputs = [sum(x) for x in zip(outputs)]
    return sum_outputs.index(max(sum_outputs))


def majority(series):
    outputs = series.tolist()
    vote = {item : 0 for item in set(outputs)}
    for uni in set(outputs):
        for out in outputs:
            if out == uni:
                vote[uni] += 1
        
    max_val = max(vote.values())
    keys = [key for k,v in vote.items() if v == max_val]
    choice = random.choice(keys)
    return choice


def transformer_parameters(task, transformer):
    for file in os.listdir(config.LOGS_PATH):
        if all(item in file for item in [task, transformer.split("/")[-1], config.DOMAIN_TRAIN]):

            return {'weights': config.LOGS_PATH + '/' + file,
                    'batch_size':int(file.split(']')[4].split('[')[1]),
                    'max_len': int(file.split(']')[3].split('[')[1]),
                    'dropout': int(file.split(']')[5].split('[')[1])}


def validation(df_val, task, transformer):
    parameters = transformer_parameters(task, transformer)
    
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
    model = TransforomerModel(transformer, parameters['dropout'], number_of_classes=df_train[task].max()+1)
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
            
            tqdm.write(f'\nValidation = Task: {task} Transfomer: {transformer.split("/")[-1]}')
            
            predictions, targets = validation(df_val, 
                                                task,
                                                transformer
            )
            
            df_val[task + '_' + transformer.split("/")[-1] + '_outputs'] = predictions
            df_val[task + '_' + transformer.split("/")[-1] + '_prediction'] = [pred.index(max(pred)) for pred in predictions]

        columns_higher_sum = [col if all(item in col for item in [task, '_outputs']) for col in df_val]
        columns_majority_vote = [col if all(item in col for item in [task, '_prediction']) for col in df_val]
        
        df_val[task + '_higher_sum'] = df_val.loc[:,columns_higher_sum].apply(lambda x: higher(x))
        df_val[task + '_majority_vote'] = df_val.loc[:,columns_majority_vote].apply(lambda x: majority(x))
        
        
        dfx = pd.merge(dfx, df_val, left_index)
        
    dfx = dfx.fillna(-1)
    dfx.to_csv(config.LOGS_PATH + '/' + config.DOMAIN_VALIDATION + '.csv')

    
    metric_col = [col if any(item in col for item in config.LABELS) for col in dfx.columns]
    metric_col = [col if '_outputs' not in col for col in metric_col]
    
    metric_dic = {'model':[], 'accuracy':[], 'f1_macro':[]}
    
    for task in config.LABELS:
        for col in metric_col:
            if task in col and task != col:
                metric_dic['model'].append(col)
                metric_dic['f1_macro'].append(metrics.f1_score(dfx[task], dfx[col], average='macro'))
                metric_dic['accuracy'].append(metrics.accuracy_score(dfx[task], dfx[col]))
            
    
    df_metrics = DataFrame(metric_dic)
    df_metrics.to_csv(config.LOGS_PATH + '/' + config.DOMAIN_VALIDATION + 'metrics' + '.csv')

    
    

        
            
            

            
    
    
    

    
            

# #TODO check if predictions and labels are in the right order with dfx dataset
# #TODO try train.py  again - I did some changes in the save domain
# #TODO write Validation.py
# #TODO write Train_all_data.py
# #TODO IMPORTANT predictions task C must the last 
# #TODO write test.py



















# if __name__ == "__main__":
#     random.seed(config.SEED)
#     np.random.seed(config.SEED)
#     torch.manual_seed(config.SEED)
#     torch.cuda.manual_seed_all(config.SEED)

#     dfx = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_DEV, sep='\t', nrows=config.N_ROWS).fillna("none")
    
#     df_results = pd.DataFrame(columns=['task',
#                                     'epoch',
#                                     'transformer',
#                                     'max_len',
#                                     'batch_size',
#                                     'lr',
#                                     'accuracy_train',
#                                     'f1-macro_train',
#                                     'loss_train'
#         ]
#     )
#     for task in tqdm(config.LABELS, desc='TASKS', position=0):
        
#         df_train = dfx.loc[dfx[task]>=0]
        
#         for transformer in tqdm(config.TRANSFORMERS, desc='TRANSFOMERS', position=1):
            
#             best_epoch, max_len, batch_size, drop_out, lr = best_parameters(task, transformer)
            
#             df_predictions = train(df_results,
#                                     df_train,
#                                     task,
#                                     transformer,
#                                     config.epochs,
#                                     best_epoch,
#                                     max_len,
#                                     batch_size,
#                                     drop_out,
#                                     lr
#             ) 
#             ## save model
            
#             df_preds['majority_vote'] = df_preds
#             df_preds['higher_sun'] = 
#             df_preds_&_labels = pd.merge(dfx, df_predictions, left_index=True)
            
#             ## I must live the task C as the last because it depend on task A model and task B model!!!