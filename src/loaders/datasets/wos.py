import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.loaders.datasets.utilities import MAX_LEN

class WOSDataset(Dataset):
    def __init__(self, dataset, tokenizer='bert-base-uncased'):

        # Load from Hugging Face
        self.dataset = dataset

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if tokenizer else AutoTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx >= len(self.dataset):
            raise IndexError(f"Index {idx} out of bounds for dataset length {len(self.dataset)}")

        item = self.dataset[idx]
        text = item['input_data'] 
        label = item['label']  # WOS uses 'label' as class index (0–34)

        # Tokenize text
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )

        label_tensor = torch.tensor(label, dtype=torch.float)

        return encoding, None, label_tensor