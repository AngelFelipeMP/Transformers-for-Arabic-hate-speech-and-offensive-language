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

def map_pred(pred, task):  
    for label, num in config.DATASET_CLASSES[task].items():
        if num == pred:
            return label
        
def test(df_test, task, transformer):
    parameters = transformer_parameters(task, transformer, config.DOMAIN_TRAIN_ALL_DATA)
    
    test_dataset = dataset.TransformerDataset_Test(
        text=df_test[config.DATASET_TEXT_PROCESSED].values,
        max_len=parameters['max_len'],
        transformer=transformer
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=parameters['batch_size'], 
        num_workers=config.VAL_WORKERS
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransforomerModel(transformer, parameters['dropout'], number_of_classes=max(list(config.DATASET_CLASSES[task].values()))+1)
    
    max(list(config.DATASET_CLASSES[task].values()))+1
    
    model.load_state_dict(torch.load(parameters['weights']))
    model.to(device)
    
    pred_test = engine.test_fn(test_data_loader, model, device)
    
    return pred_test

if __name__ == "__main__":
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    df_test = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_TEST, sep='\t')
    
    df_val_metrics = pd.read_csv(config.LOGS_PATH + '/' + config.DOMAIN_VALIDATION + '_metrics' + '.csv')
    models_dict = {}
    for task in config.LABELS:
        models_dict[task] = df_val_metrics.loc[df_val_metrics['model'].str.contains(task)].sort_values(by=['f1_macro'], ascending=False).reset_index(drop=True).loc[[0,1],'model'].tolist()
    
    for task in tqdm(config.LABELS, desc='TEST', position=0):
        
        if task == 'C':
            # I MUSR CHNAGE THE LINE BELOW from '>=0' to '>=1'
            df_val = df_test.loc[df_test[models_dict['B'][0]] >= 1]
        else:
            df_val = df_test
                
        for transformer in config.TRANSFORMERS:
            
            tqdm.write(f'Task {task} - {transformer.split("/")[-1]}')
            
            predictions = test(df_val, 
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
        for j, model in enumerate(models_dict[task]):
            
            df_test.loc[:, ['index'] + [models_dict[task][j]]].to_csv(config.LOGS_PATH + '/' + config.DOMAIN_TEST + '_task_' + task + '_rank_' + str(j+1) + '_model_' + model + '_.csv', index=False)

            df_original_labels = df_test.loc[:, ['index'] + [models_dict[task][j]]].copy()
            df_original_labels['maped_predictions'] = df_original_labels.loc[:, models_dict[task][j]].apply(lambda x: map_pred(x, task))
            df_original_labels.loc[:, ['index','maped_predictions']].to_csv(config.LOGS_PATH + '/' + config.DOMAIN_TEST + '_task_' + task + '_rank_' + str(j+1) + '_model_' + model + '_maped_labels_' + '_.csv', index=False)
