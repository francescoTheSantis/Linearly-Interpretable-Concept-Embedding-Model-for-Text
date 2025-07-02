import torch
import os
from env import DATA_PATH
from datasets import load_dataset

# Datasets
from src.loaders.datasets.cebab import CEBABDataset
from src.loaders.datasets.imdb import IMDBDataset
from src.loaders.datasets.trec50 import TREC50Dataset
from src.loaders.datasets.wos import WOSDataset
from src.loaders.datasets.clinc import CLINC_OOS_Dataset
from src.loaders.datasets.banking import Banking77Dataset

# Embedding Extractor
from src.loaders.text_preprocessing import EmbeddingExtractor_text

# Prompts
from src.loaders.prompts import CEBAB_LABELING_PROMPT, \
    IMDB_LABELING_PROMPT, TREC_LABELING_PROMPT, WOS_LABELING_PROMPT, \
    CLINC_LABELING_PROMPT, BANK_LABELING_PROMPT

class TextDataLoader:
    def __init__(self, 
                 dataset, 
                 batch_size, 
                 model_name = 'all-distilroberta-v1', 
                 tokenizer = "bert-base-uncased",   
                 selected_concepts=None,
                 llm_client=None,
                 concept_annotations=False,
                 seed=42
                ):
        
        self.dataset = dataset
        self.root = DATA_PATH
        self.batch_size = batch_size
        self.selected_concepts = selected_concepts
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.llm_client = llm_client
        self.llm_client.n_concepts = len(self.get_info()[0])
        self.concept_annotations = concept_annotations
        self.seed = seed

    def get_info(self):
        if self.dataset == 'cebab':
            c_names = ['good food', 'good ambiance', 'good service', 'good noise']
            y_names = ['Negative', 'Positive']
            c_groups = None
        elif self.dataset == 'imdb':
            c_names = self.selected_concepts if self.selected_concepts else [
                'acting', 'storyline', 'emotional arousal', 'cinematography', 'soundtrack', 'directing', 'background setting', 'editing'
            ]
            y_names = ['Negative', 'Positive']
            c_groups = None
        elif self.dataset == 'trec50':
            if self.selected_concepts!=None:
                c_names = self.selected_concepts 
            else:
                raise ValueError("selected_concepts must be provided for TREC50 dataset")
            test_dataset = load_dataset("cogcomp/trec", split='test')
            self.loaded_test = TREC50Dataset(test_dataset, self.tokenizer)
            y_names = self.loaded_test.y_names
            del self.loaded_test
            del test_dataset
            c_groups = None
        elif self.dataset == 'wos':
            if self.selected_concepts!=None:
                c_names = self.selected_concepts 
            else:
                raise ValueError("selected_concepts must be provided for WOS dataset")
            y_names = [f'{i}' for i in range(35)]  # WOS has 35 classes, no specific names provided.
            c_groups = None
        elif self.dataset == 'clinc':
            if self.selected_concepts!=None:
                c_names = self.selected_concepts 
            else:
                raise ValueError("selected_concepts must be provided for CLINC dataset")
            dataset = load_dataset("clinc_oos", "small", trust_remote_code=True)
            y_names = list(dataset['train'].features['intent'].names)
            del dataset
            c_groups = None
        elif self.dataset == 'banking':
            if self.selected_concepts!=None:
                c_names = self.selected_concepts 
            else:
                raise ValueError("selected_concepts must be provided for Banking77 dataset")
            dataset = load_dataset("banking77", split='train')
            y_names = list(dataset.features['label'].names)
            del dataset
            c_groups = None
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        return c_names, y_names, c_groups

    def load_data(self):
        if self.dataset == 'cebab':
            data_root = os.path.join(self.root, 'cebab')
            self.loaded_train = CEBABDataset(data_root, 'train', self.tokenizer)
            self.loaded_val = CEBABDataset(data_root, 'validation', self.tokenizer)
            self.loaded_test = CEBABDataset(data_root, 'test', self.tokenizer)
            self.istruction_prompt = CEBAB_LABELING_PROMPT
        elif self.dataset == 'imdb':
            data_root = os.path.join(self.root, 'imdb')
            self.loaded_train = IMDBDataset(data_root, 'train', self.selected_concepts, self.tokenizer)
            self.loaded_val = IMDBDataset(data_root, 'validation', self.selected_concepts, self.tokenizer)
            self.loaded_test = IMDBDataset(data_root, 'test', self.selected_concepts, self.tokenizer)
            self.istruction_prompt = IMDB_LABELING_PROMPT
        elif self.dataset == 'trec50':
            full_train_dataset = load_dataset("cogcomp/trec", split='train')
            split_datasets = full_train_dataset.train_test_split(test_size=0.1, seed=self.seed, shuffle=False)
            train_dataset = split_datasets['train']
            # Shuffle the train dataset to ensure randomness
            train_dataset = train_dataset.shuffle(seed=self.seed)
            self.loaded_train = TREC50Dataset(train_dataset, self.tokenizer)
            val_dataset = split_datasets['test']
            self.loaded_val = TREC50Dataset(val_dataset, self.tokenizer)
            test_dataset = load_dataset("cogcomp/trec", split='test')
            self.loaded_test = TREC50Dataset(test_dataset, self.tokenizer)
            self.istruction_prompt = TREC_LABELING_PROMPT
        elif self.dataset == 'wos':
            full_train_dataset = load_dataset("web_of_science", "WOS11967", split='train', trust_remote_code=True)
            split_datasets = full_train_dataset.train_test_split(test_size=0.2, seed=self.seed, shuffle=False)
            train_val_dataset = split_datasets['train'].train_test_split(test_size=1/8, seed=self.seed, shuffle=False)
            train_dataset = train_val_dataset['train']
            # Shuffle the train dataset to ensure randomness
            train_dataset = train_dataset.shuffle(seed=self.seed)
            self.loaded_train = WOSDataset(train_dataset, self.tokenizer)
            val_dataset = train_val_dataset['test']
            self.loaded_val = WOSDataset(val_dataset, self.tokenizer)
            test_dataset = split_datasets['test']
            self.loaded_test = WOSDataset(test_dataset, self.tokenizer)
            self.istruction_prompt = WOS_LABELING_PROMPT
        elif self.dataset == 'clinc':
            dataset = load_dataset("clinc_oos", "small", trust_remote_code=True)
            # Shuffle the train dataset to ensure randomness
            train_dataset = dataset['train'].shuffle(seed=self.seed)
            self.loaded_train = CLINC_OOS_Dataset(train_dataset, self.tokenizer)
            self.loaded_val = CLINC_OOS_Dataset(dataset['validation'], self.tokenizer)
            self.loaded_test = CLINC_OOS_Dataset(dataset['test'], self.tokenizer)
            self.istruction_prompt = CLINC_LABELING_PROMPT
        elif self.dataset == 'banking':
            dataset = load_dataset("banking77", split='train')
            train_val_dataset = dataset.train_test_split(test_size=0.2, seed=self.seed, shuffle=False)
            train_dataset = train_val_dataset['train']
            # Shuffle the train dataset to ensure randomness
            train_dataset = train_dataset.shuffle(seed=self.seed)
            self.loaded_train = Banking77Dataset(train_dataset, self.tokenizer)
            val_dataset = train_val_dataset['test']
            self.loaded_val = Banking77Dataset(val_dataset, self.tokenizer)
            test_dataset = load_dataset("banking77", split='test')
            self.loaded_test = Banking77Dataset(test_dataset, self.tokenizer)
            self.istruction_prompt = BANK_LABELING_PROMPT
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        extractor = EmbeddingExtractor_text(
            self.loaded_train, 
            self.loaded_val, 
            self.loaded_test,
            self.batch_size, 
            self.model_name, 
            self.tokenizer,
            self.device,
            self.llm_client,
            self.istruction_prompt,
            self.concept_annotations
        )

        loaded_train, loaded_val, loaded_test = extractor.produce_loaders()

        return loaded_train, loaded_val, loaded_test