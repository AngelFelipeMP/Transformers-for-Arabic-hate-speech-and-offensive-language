
# import config
# import transformers
from tokenize import tokenize
import torch.nn as nn


# class TransforomerModel(nn.Module):
#     def __init__(self):
#         super(TransforomerModel, self).__init__()
#         self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
#         self.bert_drop = nn.Dropout(0.3)
#         self.out = nn.Linear(768, 1)

#     def forward(self, ids, mask, token_type_ids):
#         _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
#         bo = self.bert_drop(o2)
#         output = self.out(bo)
#         return output
    

from transformers import AutoModel, AutoTokenizer

pre_trained_model = "bert-base-uncased"
transformer = AutoModel.from_pretrained(pre_trained_model)
tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)

max_len = 256
Example1 = "Angel table home car"
Example2 = "bhabha char roofing house get"

pt_batch = tokenizer(
    [Example1, Example2],
    padding=True,
    truncation=True,
    add_special_tokens=True,
    max_length=max_len,
    return_tensors="pt")

print(pt_batch)

#TODO discover if have max token as a input is importat or I can send the frase with its own size