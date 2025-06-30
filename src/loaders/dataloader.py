import torch
import os
from src.loaders.datasets.cebab import CEBABDataset
from src.loaders.datasets.imdb import IMDBDataset
from src.loaders.text_preprocessing import EmbeddingExtractor_text
from env import DATA_PATH
from src.loaders.prompts import CEBAB_LABELING_PROMPT, IMDB_LABELING_PROMPT

class TextDataLoader:
    def __init__(self, 
                 dataset, 
                 batch_size, 
                 model_name = 'all-distilroberta-v1', 
                 tokenizer = "bert-base-uncased",   
                 selected_concepts=None,
                 llm_client=None,
                 concept_annotations=False,
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
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        
        return c_names, y_names, c_groups

    def load_data(self):
        if self.dataset == 'cebab':
            data_root = os.path.join(self.root, 'cebab')
            self.loaded_train = CEBABDataset(data_root, 'train')
            self.loaded_val = CEBABDataset(data_root, 'validation')
            self.loaded_test = CEBABDataset(data_root, 'test')
            self.istruction_prompt = CEBAB_LABELING_PROMPT
        elif self.dataset == 'imdb':
            data_root = os.path.join(self.root, 'imdb')
            self.loaded_train = IMDBDataset(data_root, 'train', self.selected_concepts)
            self.loaded_val = IMDBDataset(data_root, 'validation', self.selected_concepts)
            self.loaded_test = IMDBDataset(data_root, 'test', self.selected_concepts)
            self.istruction_prompt = IMDB_LABELING_PROMPT

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