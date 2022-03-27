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

def join_results():
    list_of_results = []
    for file in os.listdir(config.LOGS_PATH):
        if '.csv' in file and 'results.csv' not in file:
            list_of_results.append(pd.read_csv(config.LOGS_PATH + '/' + file))
            
    if len(list_of_results) > 1:
        df = pd.concat(list_of_results, ignore_index=True)
        df.to_csv(config.LOGS_PATH + '/' + 'results' + '.csv', index=False)
            

def best_parameters(task, transformer):
    join_results()
    
    for file in os.listdir(config.LOGS_PATH):
        if 'results.csv' in file:
            df = pd.read_csv(config.LOGS_PATH + '/' + file)
            parameters = df.loc[(df['transformer'] == transformer) & (df['task'] == task)].sort_values(by=['f1-macro_val'], ascending=False).iloc[0,:]

            
    return int(parameters['epoch']), int(parameters['max_len']), int(parameters['batch_size']), float(parameters['lr']), float(parameters['dropout'])

def train(df_results, df_train, task, transformer, epochs, best_epoch, max_len, batch_size, lr, drop_out):
    
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

    num_train_steps = int(len(df_train) / batch_size * epochs)
    optimizer = AdamW(optimizer_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    
    for epoch in range(1, best_epoch+1):
        pred_train, targ_train, loss_train = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        f1_train = metrics.f1_score(targ_train, pred_train, average='macro')
        acc_train = metrics.accuracy_score(targ_train, pred_train)
        
        df_new_results = pd.DataFrame({'task':task,
                            'epoch':epoch,
                            'transformer':transformer,
                            'max_len':max_len,
                            'batch_size':batch_size,
                            'lr':lr,
                            'dropout':drop_out,
                            'accuracy_train':acc_train,
                            'f1-macro_train':f1_train,
                            'loss_train':loss_train
                        }, index=[0]
        )
        
        df_results = pd.concat([df_results, df_new_results], ignore_index=True)
        
        tqdm.write("Epoch {}/{} f1-macro_training = {:.3f}  accuracy_training = {:.3f}  loss_training = {:.3f}".format(epoch, config.EPOCHS, f1_train, acc_train, loss_train))
    
    torch.save(model.state_dict(), f'{config.LOGS_PATH}/task[{task}]_transformer[{transformer.split("/")[-1]}]_epoch[{epoch}]_maxlen[{max_len}]_batchsize[{batch_size}]_dropout[{drop_out}]_lr[{lr}].model')

    return df_results


if __name__ == "__main__":
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    dfx = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_TRAIN, sep='\t', nrows=config.N_ROWS).fillna("none")
    
    df_results = pd.DataFrame(columns=['task',
                                    'epoch',
                                    'transformer',
                                    'max_len',
                                    'batch_size',
                                    'lr',
                                    'dropout',
                                    'accuracy_train',
                                    'f1-macro_train',
                                    'loss_train'
        ]
    )
    
    for task in tqdm(config.LABELS, desc='TRAIN', position=0):
        
        df_train = dfx.loc[dfx[task]>=0]
        
        for transformer in config.TRANSFORMERS:
            
            best_epoch, max_len, batch_size, lr, drop_out = best_parameters(task, transformer)
            best_epoch = 1
            # print(best_epoch)
            # print(max_len)
            # print(batch_size)
            # print(lr)
            # print(drop_out)
            # quit()
            
            df_results = train(df_results,
                                df_train,
                                task,
                                transformer,
                                config.EPOCHS,
                                best_epoch,
                                max_len,
                                batch_size,
                                lr,
                                drop_out
            ) 
            ## save table
            ## save model
            
            df_results.to_csv(config.LOGS_PATH + '/' + 'train' + '.csv', index=False)
            

#TODO double check train.py
#TODO set train.py and train
#TODO write Validation.py
#TODO write Train_all_data.py
#TODO write test.py
#DEBUG when I get new datasets from grid search UNCOMMENT "# parameters['drop_out']"