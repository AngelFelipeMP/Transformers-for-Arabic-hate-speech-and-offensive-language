import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TransforomerModel(nn.Module):
    def __init__(self, transformer, drop_out, number_of_classes):
        super(TransforomerModel, self).__init__()
        self.number_of_classes = number_of_classes
        self.embedding_size = AutoConfig.from_pretrained(transformer).hidden_size
        self.transformer = AutoModel.from_pretrained(transformer)
        self.dropout = nn.Dropout(drop_out)
        self.classifier = nn.Linear(self.embedding_size * 2, self.number_of_classes)
        
    def forward(self, iputs):
        transformer_output  = self.transformer(**iputs)
        mean_pool = torch.mean(transformer_output['last_hidden_state'], 1)
        max_pool, _ = torch.max(transformer_output['last_hidden_state'], 1)
        cat = torch.cat((mean_pool,max_pool), 1)
        drop = self.dropout(cat)

        return self.classifier(drop)
    
    
    
    
# class TransforomerModel(nn.Module):
#     def __init__(self, transformer, drop_out, number_of_classes):
#         super(TransforomerModel, self).__init__()
#         self.number_of_classes = number_of_classes
#         self.embedding_size = AutoConfig.from_pretrained(transformer).hidden_size
#         self.transformer = AutoModel.from_pretrained(transformer)
#         self.dropout = nn.Dropout(drop_out)
#         self.classifier = nn.Linear(self.embedding_size * 2, self.number_of_classes)
        
#     def forward(self, ids, mask, token_type_ids):
#         transformer_output  = self.transformer(ids, attention_mask=mask, token_type_ids=token_type_ids)
#         mean_pool = torch.mean(transformer_output['last_hidden_state'], 1)
#         max_pool, _ = torch.max(transformer_output['last_hidden_state'], 1)
#         cat = torch.cat((mean_pool,max_pool), 1)
#         drop = self.dropout(cat)

#         return self.classifier(drop)