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
        
    def classifier(self):
        return nn.Linear(self.embedding_size * 2, self.number_of_classes)
        
    def forward(self, ids, mask, token_type_ids):
        last_hidden_state, _ = self.transformer(ids, attention_mask=mask, token_type_ids=token_type_ids)
        mean_pool = torch.mean(last_hidden_state, 1)
        max_pool = torch.max(last_hidden_state, 1)
        cat = torch.cat((mean_pool,max_pool), 1)
        drop = self.dropout(cat)
        
        return self.classifier(drop)







# class TransforomerModel(nn.Module):
#     def __init__(self, pre_trained_model=str(), drop_out=float(), embedding_size=int(), number_of_classes=int(), transformer_output=str()):
#         super(TransforomerModel, self).__init__()
#         self.number_of_classes = number_of_classes
#         self.embedding_size = embedding_size
#         self.transformer_output = transformer_output
#         self.transformer = AutoModel.from_pretrained(pre_trained_model, pooler_output=True)
#         self.dropout = nn.Dropout(drop_out)
#         self.classifier = classifier_()
        
        
#     def classifier_(self):
#         if self.transformer_output == 'last_hidden_state':
#             return nn.Linear(self.embedding_size * 2, self.number_of_classes)
#         elif self.transformer_output == 'pooler':
#             return nn.Linear(self.embedding_size, self.number_of_classes)
#         else:
#             print('The tramsformer output must be: /n (A) *last_hidden_state* or (B) *pooller* .')
#             print('Use one of the two options!')
#             exit()
        

#     def forward(self, ids=int(), mask=int(), token_type_ids=int()):
#         if self.transformer_output == 'last_hidden_state':
#             last_hidden_state, _ = self.transformer(ids, attention_mask=mask, token_type_ids=token_type_ids)
#             mean_pool = torch.mean(last_hidden_state, 1)
#             max_pool = torch.max(last_hidden_state, 1)
#             cat = torch.cat((mean_pool,max_pool), 1)
#             drop = self.dropout(cat)
#         elif self.transformer_output == 'pooler':
#             _ , poller_output = self.transformer(ids, attention_mask=mask, token_type_ids=token_type_ids) 
#             drop = self.dropout(poller_output)
    
#         return self.classifier(drop)