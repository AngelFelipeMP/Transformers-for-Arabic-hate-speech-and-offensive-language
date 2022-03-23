import torch
import torch.nn as nn


def loss_fn(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    fin_targets = []
    fin_predictions = []
    total_loss = 0
    
    
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
        
    return fin_predictions, fin_targets, total_loss/len(data_loader)
        
        


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_predictions = []
    total_loss = 0
    
    with torch.no_grad():
        for d in data_loader:
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
