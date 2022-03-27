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
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()


def run(df_train, df_val, max_len, task, transformer, batch_size, drop_out, lr, df_results):
    
    train_dataset = dataset.TransformerDataset(
        text=df_train[config.DATASET_TEXT_PROCESSED].values,
        target=df_train[task].values,
        max_len=max_len,
        transformer=transformer
    )

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        num_workers = config.TRAIN_WORKERS
    )

    val_dataset = dataset.TransformerDataset(
        text=df_val[config.DATASET_TEXT_PROCESSED].values,
        target=df_val[task].values,
        max_len=max_len,
        transformer=transformer
    )

    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        num_workers=config.VAL_WORKERS
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransforomerModel(transformer, drop_out, number_of_classes=df_train[task].max()+1)
    model.to(device)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / batch_size * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    
    for epoch in range(1, config.EPOCHS+1):
        pred_train, targ_train, loss_train = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        f1_train = metrics.f1_score(targ_train, pred_train, average='macro')
        acc_train = metrics.accuracy_score(targ_train, pred_train)
        
        pred_val, targ_val, loss_val = engine.eval_fn(val_data_loader, model, device)
        f1_val = metrics.f1_score(targ_val, pred_val, average='macro')
        acc_val = metrics.accuracy_score(targ_val, pred_val)
        
        df_new_results = pd.DataFrame({'task':task,
                            'epoch':epoch,
                            'transformer':transformer,
                            'max_len':max_len,
                            'batch_size':batch_size,
                            'lr':lr,
                            'dropout':drop_out,
                            'accuracy_train':acc_train,
                            'f1-macro_train':f1_train,
                            'loss_train':loss_train,
                            'accuracy_val':acc_val,
                            'f1-macro_val':f1_val,
                            'loss_val':loss_val
                        }, index=[0]
        ) 
        df_results = pd.concat([df_results, df_new_results], ignore_index=True)
        
        tqdm.write("Epoch {}/{} f1-macro_training = {:.3f}  accuracy_training = {:.3f}  loss_training = {:.3f} f1-macro_val = {:.3f}  accuracy_val = {:.3f}  loss_val = {:.3f}".format(epoch, config.EPOCHS, f1_train, acc_train, loss_train, f1_val, acc_val, loss_val))

    return df_results

if __name__ == "__main__":
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    dfx = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_TRAIN, sep='\t', nrows=config.N_ROWS).fillna("none")
    skf = StratifiedKFold(n_splits=config.SPLITS, shuffle=True, random_state=config.SEED)

    df_results = pd.DataFrame(columns=['task',
                                        'epoch',
                                        'transformer',
                                        'max_len',
                                        'batch_size',
                                        'lr',
                                        'dropout',
                                        'accuracy_train',
                                        'f1-macro_train',
                                        'loss_train',
                                        'accuracy_val',
                                        'f1-macro_val',
                                        'loss_val'
            ]
    )
    
    inter = len(config.LABELS) * len(config.TRANSFORMERS) * len(config.MAX_LEN) * len(config.BATCH_SIZE) * len(config.DROPOUT) * len(config.LR) * config.SPLITS
    grid_search_bar = tqdm(total=inter, desc='GRID SEARCH', position=2)
    
    for task in tqdm(config.LABELS, desc='TASKS', position=1):
        df_grid_search = dfx.loc[dfx[task]>=0].reset_index(drop=True)
        for transformer in tqdm(config.TRANSFORMERS, desc='TRANSFOMERS', position=0):
            for max_len in config.MAX_LEN:
                for batch_size in config.BATCH_SIZE:
                    for drop_out in config.DROPOUT:
                        for lr in config.LR:
                            
                            for fold, (train_index, val_index) in enumerate(skf.split(df_grid_search[config.DATASET_TEXT_PROCESSED], df_grid_search[task])):
                                df_train = df_grid_search.loc[train_index]
                                df_val = df_grid_search.loc[val_index]
                                
                                tqdm.write(f'\nTask: {task} Transfomer: {transformer.split("/")[-1]} Max_len: {max_len} Batch_size: {batch_size} Dropout: {drop_out} lr: {lr} Fold: {fold+1}/{config.SPLITS}')
                                
                                df_results = run(df_train,
                                                    df_val, 
                                                    max_len, 
                                                    task, 
                                                    transformer, 
                                                    batch_size, 
                                                    drop_out,
                                                    lr,
                                                    df_results
                                )
                            
                                grid_search_bar.update(1)
                            
                            df_results = df_results.groupby(['task',
                                                            'epoch',
                                                            'transformer',
                                                            'max_len',
                                                            'batch_size',
                                                            'lr',
                                                            'dropout'], as_index=False, sort=False)['accuracy_train',
                                                                                                'f1-macro_train',
                                                                                                'loss_train',
                                                                                                'accuracy_val',
                                                                                                'f1-macro_val',
                                                                                                'loss_val'].mean()
                            
                            df_results.to_csv(config.LOGS_PATH + '/' + DOMAIN_GRID_SEARCH + '.csv', index=False)
                            
            
    
#TODO may remove save models
    
    
    
    
    
    
    
    
    

# pre_trained_model = "bert-base-uncased"
# transformer = AutoModel.from_pretrained(pre_trained_model)
# tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)

# max_len = 15
# Example1 = "Angel table home car"
# Example2 = "bhabha char roofing house get"
# Example3 = "I wan to go to the beach for surfing"

# pt_batch = tokenizer(
#     [Example1, Example2, Example3],
#     padding=True,
#     truncation=True,
#     add_special_tokens=True,
#     max_length=max_len,
#     return_tensors="pt")

# print(pt_batch)