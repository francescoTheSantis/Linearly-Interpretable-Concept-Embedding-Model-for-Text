from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
from src.loaders.datasets.utilities import MAX_LEN

class FewRelDataset(Dataset):
    def __init__(self, dataset, tokenizer='bert-base-uncased'):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        label_names = self.dataset.features['label'].names
        self.label_map = {label: idx for idx, label in enumerate(label_names)}

        self.dataset = self.dataset.map(lambda x: {'label_id': x['label']})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # FewRel stores 'tokens' as a list of words; join into string
        text = " ".join(item['tokens'])
        label = item['label_id']

        encoded = self.tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded, None, torch.tensor(label, dtype=torch.long)
