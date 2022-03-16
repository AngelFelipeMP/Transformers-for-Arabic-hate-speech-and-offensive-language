import torch
import torch.nn as nn
from transformers import AutoTokenizer

class TransformerDataset:
    def __init__(self, text, target, max_len, transformer):
        self.text = text
        self.target = target
        self.tokenizer = AutoTokenizer.from_pretrained(transformer)
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        # text = str(self.text[item])
        # text = " ".join(text.split())

        inputs = self.tokenizer(
            self.text,
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_tensors="pt")

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        
        # print('@'*100)
        # print("ids", len(ids))
        # print("mask", len(mask))
        # print( "token_type_ids", len(token_type_ids))
        # # print("targets", len(self.target[item]))
        
        print('@'*100)
        print("ids",torch.tensor(ids, dtype=torch.long).size())
        print("mask",torch.tensor(mask, dtype=torch.long).size())
        print( "token_type_ids",torch.tensor(token_type_ids, dtype=torch.long).size())
        print("targets",torch.tensor(self.target[item], dtype=torch.float).size())

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }