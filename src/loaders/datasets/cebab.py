from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from src.loaders.datasets.utilities import process, process2, process4, MAX_LEN

class CEBABDataset(Dataset):
    def __init__(self, root, split, tokenizer='bert-base-uncased'):
        path = os.path.join(root, f'cebab_{split}.csv')
        self.data = pd.read_csv(path)
        self.data['food'] = self.data.apply(lambda row: process(row['food']), axis=1)
        self.data['ambiance'] = self.data.apply(lambda row: process(row['ambiance']), axis=1)
        self.data['service'] = self.data.apply(lambda row: process(row['service']), axis=1)
        self.data['noise'] = self.data.apply(lambda row: process(row['noise']), axis=1)
        #self.data['bin_rating'] = self.data.apply(lambda row: process4(row['average_rating']), axis=1) # bin_rating
        self.data['bin_rating'] = self.data.apply(lambda row: process2(row['bin_rating']), axis=1)

        # Use the provided tokenizer or default to AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if tokenizer else AutoTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} is out of bounds for DataFrame with length {len(self.data)}")

        # Extract review, concept annotations, and label
        review = self.tokenizer(
                self.data.loc[idx, 'review'],
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=MAX_LEN
        )  # Column name for the review text
        concepts = self.data.loc[idx, ['food', 'service', 'ambiance', 'noise']].values

        label = self.data.loc[idx, 'bin_rating']  # Column name for the label

        # Convert concepts to numeric and handle invalid entries
        concepts = np.array(concepts, dtype=np.float32)  # Ensure all are numeric
        label = int(label)  # Ensure label is an integer

        # Convert to tensors
        concepts_tensor = torch.tensor(concepts, dtype=torch.float)
        label_tensor = torch.tensor(label, dtype=torch.float)

        return review, concepts_tensor, label_tensor
