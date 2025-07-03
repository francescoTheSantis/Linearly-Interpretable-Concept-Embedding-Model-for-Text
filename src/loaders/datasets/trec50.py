import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.loaders.datasets.utilities import MAX_LEN

class TREC50Dataset(Dataset):
    def __init__(self, dataset, tokenizer='bert-base-uncased'):

        # Load from Hugging Face
        self.dataset = dataset

        # Use provided tokenizer or default to BERT
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if tokenizer else AutoTokenizer.from_pretrained('bert-base-uncased')

        '''
        # Map fine_label to int if needed
        self.label_map = {label: idx for idx, label in enumerate(set(self.dataset['fine_label']))}
        self.dataset = self.dataset.map(lambda example: {
            'label_id': self.label_map[example['fine_label']]
        })
        '''

        # Store the y names
        self.y_names = list(self.dataset.features['fine_label'].names)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx >= len(self.dataset):
            raise IndexError(f"Index {idx} is out of bounds for dataset with length {len(self.dataset)}")

        example = self.dataset[idx]
        question = example['text']
        #label = example['label_id']
        label = example['fine_label']

        # Tokenize the question
        encoded = self.tokenizer(
            question,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN
        )

        label_tensor = torch.tensor(label, dtype=torch.float)

        return encoded, None, label_tensor
