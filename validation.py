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


def Transformer_parameters(task, transformer):
    for file in os.listdir(config.LOGS_PATH):
        if all(item in file for item in [task, transformer.split("/")[-1], config.DOMAIN_TRAIN]):

            return {'weights': config.LOGS_PATH + '/' + file,
                    'batch_size':int(file.split(']')[4].split('[')[1]),
                    'max_len': int(file.split(']')[3].split('[')[1])}


def validation(df_val, task, transformer):
    parameters = Transformer_parameters(task, transformer)
    
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
    model = TransforomerModel(transformer, drop_out, number_of_classes=df_train[task].max()+1)
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
    
    for task in tqdm(config.LABELS, desc='TRAIN', position=0):
        
        df_val = dfx.loc[dfx[task]>=0]
        
        for transformer in config.TRANSFORMERS:
            
            tqdm.write(f'\nValidation = Task: {task} Transfomer: {transformer.split("/")[-1]} Max_len: {max_len}')
            
            predictions, targets = validation(df_val, 
                                                task,
                                                transformer
            )
            
            df_val[task + '_' + transformer.split("/")[-1]] = predictions
            
        dfx = pd.merge(dfx, df_val, left_index)
    # dfx = dfx.fillna(-1)

    

    #caculate prediction for each model
    #majority volte - tie get a random choice between the ones choice by the models
    #sum of the outputs
        
            
            

            
    
    
    
    f1_val = metrics.f1_score(targ_val, pred_val, average='macro')
    acc_val = metrics.accuracy_score(targ_val, pred_val)
    
    
    df_preds['majority_vote'] = df_preds
    df_preds['higher_sun'] = 
    df_preds_&_labels = pd.merge(dfx, df_predictions, left_index=True)
    
    ## I must live the task C as the last because it depend on task A model and task B model!!!
    
    df_results.to_csv(config.LOGS_PATH + '/' + 'train' + '.csv', index=False)
            

# #TODO check if predictions and labels are in the right order with dfx dataset
# #TODO try train.py  again - I did some changes in the save domain
# #TODO write Validation.py
# #TODO write Train_all_data.py
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