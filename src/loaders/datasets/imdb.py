import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from src.loaders.datasets.utilities import process, process2, MAX_LEN

class IMDBDataset(Dataset):
    def __init__(self, root, split, selected_concepts=None, tokenizer_name='bert-base-uncased'):
        """"""
        self.folder = root
        self.data = pd.concat([pd.read_csv(f'{self.folder}/IMDB-{split}-generated.csv'), pd.read_csv(f'{self.folder}/IMDB-{split}-manual.csv')]).reset_index()
        self.data['acting'] = self.data.apply(lambda row: process(row['acting']), axis=1)
        self.data['storyline'] = self.data.apply(lambda row: process(row['storyline']), axis=1)
        self.data['emotional arousal'] = self.data.apply(lambda row: process(row['emotional arousal']), axis=1)
        self.data['cinematography'] = self.data.apply(lambda row: process(row['cinematography']), axis=1)
        self.data['soundtrack'] = self.data.apply(lambda row: process(row['soundtrack']), axis=1)
        self.data['directing'] = self.data.apply(lambda row: process(row['directing']), axis=1)
        self.data['background setting'] = self.data.apply(lambda row: process(row['background setting']), axis=1)
        self.data['editing'] = self.data.apply(lambda row: process(row['editing']), axis=1)
        self.data['sentiment'] = self.data.apply(lambda row: process2(row['sentiment']), axis=1)
        
        # Use AutoTokenizer to dynamically load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if selected_concepts is None:
            self.selected_concepts = ['acting', 'storyline', 'emotional arousal', 'cinematography', 'soundtrack', 'directing', 'background setting', 'editing']
        else:
            self.selected_concepts = selected_concepts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} is out of bounds for DataFrame with length {len(self.data)}")

        # Extract review, concept annotations, and label
        # Extract review, concept annotations, and label
        review = self.tokenizer(
                self.data.loc[idx, 'review'],
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=MAX_LEN
        )  # Column name for the review text
        concepts = self.data.loc[idx, self.selected_concepts].values

        label = self.data.loc[idx, 'sentiment']  # Column name for the label

        # Convert concepts to numeric and handle invalid entries
        concepts = np.array(concepts, dtype=np.float32)  # Ensure all are numeric
        label = int(label)  # Ensure label is an integer

        # Convert to tensors
        concepts_tensor = torch.tensor(concepts, dtype=torch.float)
        label_tensor = torch.tensor(label, dtype=torch.float)

        return review, concepts_tensor, label_tensor