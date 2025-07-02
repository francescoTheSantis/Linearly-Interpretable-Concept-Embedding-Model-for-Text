from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
from src.loaders.datasets.utilities import MAX_LEN

class CLINC_OOS_Dataset(Dataset):
    def __init__(self, dataset, tokenizer='bert-base-uncased'):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        label = item['intent']
        
        encoded = self.tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')

        return encoded, None, torch.tensor(label, dtype=torch.long)
