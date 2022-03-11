import dataset
import engine
import torch
import pandas as pd
import numpy as np
import random
import config

from model import TransforomerModel
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def run(df_train, df_val, max_len, task, transformer, batch_size, drop_out, embedding_size, number_of_classes, lr, best_f1, df_results):
        
        train_dataset = dataset.TransformerDataset(
            review=df_train[config.DATASET_TEXT_PROCESSED].values,
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
            review=df_val[config.DATASET_TEXT_PROCESSED].values,
            target=df_val[task].values,
            max_len=max_len,
            transformer=transformer

        val_data_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, 
            batch_size=batch_size, 
            num_workers=config.VAL_WORKERS
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TransforomerModel(transformer, drop_out, embedding_size, number_of_classes)
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

        for epoch in tqdm(range(1, config.EPOCHS+1)):
            pred_train, targ_train, loss_train = engine.train_fn(train_data_loader, model, optimizer, device, scheduler, epoch)
            f1_train = metrics.f1_score(targ_train, pred_train, average=macro)
            acc_train = metrics.accuracy(targ_train, pred_train)
            
            pred_val, targ_val, loss_val = engine.val_fn(val_data_loader, model, device)
            f1_val = metrics.f1_score(targ_val, pred_val, average=macro)
            acc_val = metrics.accuracy(targ_val, pred_val)
            
            df_results.append({'task':task,
                                'epoch':epoch,
                                'transformer':transformer,
                                'max_len':max_len,
                                'batch_size':batch_size,
                                'lr':lr,
                                'accuracy_train':acc_train,
                                'f1-macro_train':f1_train
                                'accuracy_val':acc_val,
                                'f1-macro_val':f1_val,
                                ignore_index=True
            }
            
            print(f"f1-macro_training = {:.3f}  accuracy_training = {:.3f}  loss_training = {:.3f}".format(f1_train, acc_train, loss_train))
            print(f"f1-macro_val = {:.3f}  accuracy_val = {:.3f}  loss_val = {:.3f}".format(f1_val, acc_val, loss_val))
            if f1_val > best_f1:
                torch.save(model.state_dict(), f'{config.LOGS_PATH}/task[{task}]_transformer[{transformer}]_epoch[{epoch}]_maxlen[{max_len}]_batchsize[{batch_size}]_dropout[{drop_out}]_lr[{lr}].model')
                best_f1 = f1_val

if __name__ == "__main__":
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    dfx = pd.read_csv(config.DATASET_TRAIN, sep='\t', index_col=config.DATASET_INDEX).fillna("none")
    skf = StratifiedKFold(n_splits=config.SPLITS, random_state=seed_val)

    df_results = pd.DataFrame(columns=['task',
                                        'epoch',
                                        'transformer',
                                        'max_len',
                                        'batch_size',
                                        'lr',
                                        'accuracy_train','
                                        'f1-macro_train',
                                        'accuracy_val',
                                        'f1-macro_val']
    )


    for task in config.LABELS:
        for transformer, embedding_size in config.TRANSFORMERS.item():
            best_f1 = 0
            for max_len in config.MAX_LEN:
                for batch_size in config.BATCH_SIZE:
                    for lr in config.LR:
                
        
                        for train_index, val_index in skf.split(dfx[config.DATASET_TEXT_PROCESSED], dfx[task]):
                            df_train = dfx.loc[train_index]
                            df_val = dfx.loc[val_index]
                            
                            
                            run(df_train, 
                                df_val, 
                                max_len, 
                                task, 
                                transformer, 
                                batch_size, 
                                drop_out, 
                                embedding_size, 
                                number_of_classes, 
                                lr, best_f1, 
                                df_results)
        

    

    #TODO testar data.py
    #TODO testar utils.py
    #TODO testar tqdm
    #TODO test code with one aranic transformer
    #TODO algument green search tring to improve the optimazer
    
    
    
    
    
    
    
    
    
    
    

pre_trained_model = "bert-base-uncased"
transformer = AutoModel.from_pretrained(pre_trained_model)
tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)

max_len = 15
Example1 = "Angel table home car"
Example2 = "bhabha char roofing house get"
Example3 = "I wan to go to the beach for surfing"

pt_batch = tokenizer(
    [Example1, Example2, Example3],
    padding=True,
    truncation=True,
    add_special_tokens=True,
    max_length=max_len,
    return_tensors="pt")

print(pt_batch)