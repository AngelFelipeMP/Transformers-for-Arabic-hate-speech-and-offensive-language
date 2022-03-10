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


seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

    
dfx = pd.read_csv(config.DATASET_TRAIN, sep='\t', index_col=config.DATASET_INDEX).fillna("none")
skf = StratifiedKFold(n_splits=10, random_state=seed_val)

# greed search come here
## for epoch in tqdm(range(1, epochs+1)):

for train_index, val_index in skf.split(dfx[config.DATASET_TEXT_PROCESSED], dfx[config.LABEL]):
    df_train = dfx.loc[train_index]
    df_val = dfx.loc[val_index]
    

def run(df_train, df_val, max_len, transformer, batch_size, drop_out, embedding_size, number_of_classes):
    
    train_dataset = dataset.TransformerDataset(
        review=df_train[config.DATASET_TEXT_PROCESSED].values,
        target=df_train[config.LABEL].values,
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
        target=df_val[config.LABEL].values,
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
    
    #TODO check if keeo or remove the part below from run()

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

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5 #TODO change
        accuracy = metrics.accuracy_score(targets, outputs) #TODO macro-averaged F1-score
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

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