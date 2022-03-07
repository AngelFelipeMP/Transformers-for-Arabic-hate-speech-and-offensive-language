import torch
import torch.nn as nn
from transformers import AutoTokenizer

class TransformerDataset:
    def __init__(self, text, target, max_len):
        self.text = text
        self.target = target
        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = " ".join(text.split())

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_tensors="pt")

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }