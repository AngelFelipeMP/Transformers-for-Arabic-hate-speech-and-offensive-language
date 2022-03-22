import torch
import torch.nn as nn
from tqdm import tqdm


def loss_fn(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)


def train_fn(data_loader, model, optimizer, device, scheduler, epoch):
    model.train()
    fin_targets = []
    fin_predictions = []
    total_loss = 0
    
    # progress_bar = tqdm(data_loader, desc='Epoch {:1d}'.format(epoch))

    # for d in progress_bar:
    #     ids = d["ids"]
    #     token_type_ids = d["token_type_ids"]
    #     mask = d["mask"]
    #     targets = d["targets"]

    epoch_batch = tqdm(total=len(data_loader), desc='Batch', position=3)
    
    for d in data_loader:
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs, targets)
        
        total_loss += loss.cpu().detach().numpy().tolist()
        _, predictions = torch.max(outputs, 1)
        fin_targets.extend(targets.cpu().detach().numpy().tolist())
        fin_predictions.extend(predictions.cpu().detach().numpy().tolist())
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        epoch_batch.update(1)
        # progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.cpu().detach().numpy().tolist()/len(d))})
        
    return fin_predictions, fin_targets, total_loss/len(data_loader)
        
        


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_predictions = []
    total_loss = 0
    
    with torch.no_grad():
        for d in tqdm(data_loader):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fn(outputs, targets)
            total_loss += loss.cpu().detach().numpy().tolist()
            
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            _, predictions = torch.max(outputs, 1)
            fin_predictions.extend(predictions.cpu().detach().numpy().tolist())
    
    return fin_predictions, fin_targets, total_loss/len(data_loader)
