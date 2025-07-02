import torch
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, AutoTokenizer
from tqdm import tqdm

class EmbeddingExtractor_text:
    def __init__(self, 
                 train_loader, 
                 val_loader, 
                 test_loader, 
                 batch_size, 
                 model_name, 
                 tokenizer_name=None, 
                 device='cuda',
                 llm_client=None,
                 istruction_prompt=None,
                 concept_annotations=False
                 ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) if tokenizer_name!=None else BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = SentenceTransformer(model_name).to(device)
        self.llm_client = llm_client
        self.istruction_prompt = istruction_prompt
        self.concept_annotations = concept_annotations

    def _extract_embeddings(self, loader):
        """Helper function to extract embeddings for a given DataLoader."""
        input_ids = []
        token_type_ids = []
        attention_mask = []
        embeddings = []
        concepts_list = []
        generated_concepts = []
        labels = []

        with torch.no_grad():
            for review, concepts, targets in tqdm(loader):
                # decode the reviews
                ids = review['input_ids']
                decoded_review = self.tokenizer.decode(ids.squeeze(0), skip_special_tokens=True) 
                embs = self.model.encode(decoded_review, convert_to_tensor=True, show_progress_bar=False)
                # If llm_client is provided, use it to get concepts
                if self.llm_client is not None and self.istruction_prompt is not None:
                    # Ask the LLM for concepts
                    gen_concepts = self.llm_client.ask(decoded_review, self.istruction_prompt, return_tensor=True)
                else:
                    raise ValueError("LLM client or instruction prompt not provided.")

                if not self.concept_annotations:
                    # If no concept annotations, create a tensor of -1 with the same shape as the embeddings
                    concepts = -1 * torch.ones_like(gen_concepts)
                
                embs = embs.unsqueeze(0)
                concepts = concepts.unsqueeze(0)
                targets = targets.unsqueeze(0)
                gen_concepts = gen_concepts.unsqueeze(0)

                embeddings.append(embs.cpu())
                concepts_list.append(concepts.cpu())
                labels.append(targets.cpu())
                input_ids.append(ids.cpu())
                token_type_ids.append(review['token_type_ids'].cpu())
                attention_mask.append(review['attention_mask'].cpu())
                generated_concepts.append(gen_concepts.cpu())

        # Concatenate all embeddings and labels
        embeddings = torch.cat(embeddings, dim=0)
        concepts = torch.cat(concepts_list, dim=0)
        labels = torch.cat(labels, dim=0)
        input_ids = torch.cat(input_ids, dim=0)
        token_type_ids = torch.cat(token_type_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        generated_concepts = torch.cat(generated_concepts, dim=0)

        return input_ids, token_type_ids, attention_mask, embeddings, concepts.float(), labels.long(), generated_concepts.float()

    def _create_loader(self, 
                       input_ids, 
                       token_type_ids, 
                       attention_mask, 
                       embeddings, 
                       concepts, 
                       labels, 
                       gen_c,
                       batch_size, 
                       shuffle=False):
        """Helper function to create a DataLoader from embeddings and labels."""
        dataset = TensorDataset(input_ids, 
                                token_type_ids, 
                                attention_mask, 
                                embeddings, 
                                concepts, 
                                labels,
                                gen_c)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def produce_loaders(self):
        """Produces new DataLoaders with embeddings."""
        train_input_ids, train_token_type_ids, train_attention_mask, train_embeddings, train_concepts, train_labels, train_gen_c = self._extract_embeddings(self.train_loader)
        val_input_ids, val_token_type_ids, val_attention_mask, val_embeddings, val_concepts, val_labels, val_gen_c = self._extract_embeddings(self.val_loader)
        test_input_ids, test_token_type_ids, test_attention_mask, test_embeddings, test_concepts, test_labels, test_gen_c = self._extract_embeddings(self.test_loader)
        train_loader = self._create_loader(train_input_ids, train_token_type_ids, train_attention_mask, train_embeddings, train_concepts, train_labels, train_gen_c, self.batch_size, True)
        val_loader = self._create_loader(val_input_ids, val_token_type_ids, val_attention_mask, val_embeddings, val_concepts, val_labels, val_gen_c, self.batch_size)
        test_loader = self._create_loader(test_input_ids, test_token_type_ids, test_attention_mask, test_embeddings, test_concepts, test_labels, test_gen_c, self.batch_size)
        return train_loader, val_loader, test_loader
