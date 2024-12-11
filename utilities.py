import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
import math
import re
import string
import nltk
import gensim.downloader
import pandas as pd
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import os
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from mistralai.client import MistralClient
#from mistralai.models.chat_completion import ChatMessage
#from hqq.core.quantize import *
from sklearn.model_selection import train_test_split
from collections import Counter

def softselect(values, temperature):
    softmax_scores = torch.log_softmax(values, dim=1)
    softscores = torch.sigmoid(softmax_scores - temperature * softmax_scores.mean(dim=1, keepdim=True))
    return softscores


class ConceptLinearLayer(torch.nn.Module):
    """
    This layer implements a linear layer working over concept embedding and outputting the task prediction.
    Similarly to the ConceptReasoningLayer, it also makes an interpretable prediction. This time, however, the
    prediction is a linear combination of the concepts, where the weights are predicted for each sample by the layer.
    """
    def __init__(self, emb_size, n_classes, bias=True, attention=False, modality='unbound'):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.modality = modality
        self.bias = bias
        
        if self.modality=='standard':
            self.pos_weight_nn = torch.nn.Sequential(
                torch.nn.Linear(emb_size, emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(emb_size, n_classes),
                torch.nn.Sigmoid(),
            )
            self.neg_weight_nn = torch.nn.Sequential(
                torch.nn.Linear(emb_size, emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(emb_size, n_classes),
                torch.nn.Sigmoid(),
            )
            if self.bias:
                self.pos_bias_nn = torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size),
                                                   torch.nn.LeakyReLU(),
                                                   torch.nn.Linear(emb_size, n_classes),
                                                   torch.nn.Sigmoid()
                                                   )
                self.neg_bias_nn = torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size),
                                                      torch.nn.LeakyReLU(),
                                                      torch.nn.Linear(emb_size, n_classes),
                                                      torch.nn.Sigmoid()
                                                      )
        elif self.modality=='unbound':
            self.weight_nn = torch.nn.Sequential(
                torch.nn.Linear(emb_size, emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(emb_size, n_classes)
            )
            if self.bias:
                self.bias_nn = torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size),
                                                   torch.nn.LeakyReLU(),
                                                   torch.nn.Linear(emb_size, n_classes),
                                                   )    


        self.attention = attention
        if attention:
            self.attention_nn = torch.nn.MultiheadAttention(emb_size, 1, batch_first=True)

    def forward(self, x, c, return_attn=False, weight_attn=None):

        if self.attention:
            x, _ = self.attention_nn(x, x, x)

        if weight_attn is None:
            if self.modality=='standard':
                weight_attn = self.pos_weight_nn(x) - self.neg_weight_nn(x)
            elif self.modality=='unbound':
                weight_attn = self.weight_nn(x)

        logits = (c.unsqueeze(-1) * weight_attn).sum(dim=1).float()
        
        if self.bias:
            if self.modality=='standard':
                bias_attn = self.pos_bias_nn(x.mean(dim=1)) - self.neg_bias_nn(x.mean(dim=1))
            elif self.modality=='unbound':
                bias_attn = self.bias_nn(x.mean(dim=1))

            logits += bias_attn
            
        preds = logits
        if return_attn:
            if self.bias:
                return preds, weight_attn, bias_attn
            return preds, weight_attn, torch.zeros_like(preds)
        else:
            return preds

    def explain(self, x, c, mode, concept_names=None, class_names=None, weight_attn=None):
        assert mode in ['local', 'global', 'exact']

        if concept_names is None:
            concept_names = [f'c_{i}' for i in range(c.shape[1])]
        if class_names is None:
            class_names = [f'y_{i}' for i in range(self.n_classes)]

        # make a forward pass to get predictions and attention weights
        if self.bias:
            y_preds, weight_attn_mask, _ = self.forward(x, c, return_attn=True, weight_attn=weight_attn)
        else:
            y_preds, weight_attn_mask = self.forward(x, c, return_attn=True, weight_attn=weight_attn)

        explanations = []
        all_class_explanations = {cn: [] for cn in class_names}
        for sample_idx in range(len(x)):
            prediction = y_preds[sample_idx] > 0.5
            active_classes = torch.argwhere(prediction).ravel()

            if len(active_classes) == 0:
                # if no class is active for this sample, then we cannot extract any explanation
                explanations.append({
                    'class': -1,
                    'explanation': '',
                    'attention': [],
                })
            else:
                # else we can extract an explanation for each active class!
                for target_class in active_classes:
                    attentions = []
                    minterm = {}
                    for concept_idx in range(len(concept_names)):
                        c_pred = c[sample_idx, concept_idx]
                        weight_attn = weight_attn_mask[sample_idx, concept_idx, target_class]

                        # we first check if the concept was relevant
                        # a concept is relevant <-> the absolute value of the weight attention score is higher than 0.5 and the concept is active
                        if torch.abs(c_pred) > 0.5:
                            minterm[concept_names[concept_idx]] = weight_attn
                        attentions.append(weight_attn.item())

                    # add explanation to list
                    target_class_name = class_names[target_class]
                    all_class_explanations[target_class_name].append(minterm)
                    explanations.append({
                        'sample-id': sample_idx,
                        'class': target_class_name,
                        'explanation': minterm,
                        'attention': attentions
                    })

        if mode == 'global':
            # count most frequent explanations for each class
            explanations = []
            for class_id, class_explanations in all_class_explanations.items():
                explanation_count = Counter(class_explanations)
                for explanation, count in explanation_count.most_common():
                    if count > 5:
                        explanations.append({
                            'class': class_id,
                            'explanation': explanation,
                            'count': count,
                        })

        return explanations

    @staticmethod
    def entropy_reg(t: torch.Tensor):
        abs_t = torch.abs(t) + 1e-10
        entropy = abs_t * torch.log(abs_t)
        entropy_sum = - torch.sum(entropy, dim=1)
        if entropy_sum.isnan().any():
            print(t)
            raise ValueError
        return entropy_sum.mean()

    
class Modified_cem(nn.Module):
    def __init__(self, in_features, n_concepts, emb_size):
        super(Modified_cem, self).__init__()   
        self.in_features = in_features
        self.n_concepts = n_concepts
        self.emb_size = emb_size
        self.phi = nn.Sequential(nn.Linear(in_features, 2 * emb_size * n_concepts),
                                 nn.LeakyReLU(),
        )

    def forward(self, x, c_pred):
        concept_embeddings = self.phi(x).view(-1, self.n_concepts * 2, self.emb_size)
        pos_embs = concept_embeddings[:,0:self.n_concepts, :]
        neg_embs = concept_embeddings[:,self.n_concepts:, :]
        c_emb = pos_embs * c_pred.unsqueeze(-1).expand(-1, -1, self.emb_size) + \
                neg_embs * (1-c_pred).unsqueeze(-1).expand(-1, -1, self.emb_size)
        return c_emb

    
class ConceptEmbedding_cace(torch.nn.Module):
    def __init__(
            self,
            in_features,
            n_concepts,
            emb_size,
            active_intervention_values=None,
            inactive_intervention_values=None,
            intervention_idxs=None,
            training_intervention_prob=0.25,
    ):
        super(ConceptEmbedding_cace, self).__init__()
        self.emb_size = emb_size
        self.intervention_idxs = intervention_idxs
        self.training_intervention_prob = training_intervention_prob
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        self.concept_context_generators = torch.nn.ModuleList()
        for i in range(n_concepts):
            self.concept_context_generators.append(torch.nn.Sequential(
                torch.nn.Linear(in_features, 2 * emb_size),
                torch.nn.LeakyReLU(),
            ))
        self.concept_prob_predictor = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, 1),
            torch.nn.Sigmoid(),
        )

        # And default values for interventions here
        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.zeros(n_concepts)

    def _after_interventions(
            self,
            prob,
            concept_idx,
            intervention_idxs=None,
            c_true=None,
            train=False,
    ):
        if train and (self.training_intervention_prob != 0) and (intervention_idxs is None):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(self.ones * self.training_intervention_prob)
            intervention_idxs = torch.nonzero(mask).reshape(-1)
        if (c_true is None) or (intervention_idxs is None):
            return prob
        if concept_idx not in intervention_idxs:
            return prob
        return (c_true[:, concept_idx:concept_idx + 1] * self.active_intervention_values[concept_idx]) + \
            ((c_true[:, concept_idx:concept_idx + 1] - 1) * -self.inactive_intervention_values[concept_idx])

    def forward(self, x, intervention_idxs=None, c=None, train=False, on=None, concept_id=0):
        c_emb_list, c_pred_list = [], []
        # We give precendence to inference time interventions arguments
        used_int_idxs = intervention_idxs
        if used_int_idxs is None:
            used_int_idxs = self.intervention_idxs
        for i, context_gen in enumerate(self.concept_context_generators):
            context = context_gen(x)
            c_pred = self.concept_prob_predictor(context)
            c_pred_list.append(c_pred)
            # Time to check for interventions
            c_pred = self._after_interventions(
                prob=c_pred,
                concept_idx=i,
                intervention_idxs=used_int_idxs,
                c_true=c,
                train=train,
            )
            context_pos = context[:, :self.emb_size]
            context_neg = context[:, self.emb_size:]
            if on is True:
                if i == concept_id:
                    c_pred[:,0] = 1
            elif on is False:
                if i == concept_id:
                    c_pred[:,0] = 0
            c_emb = context_pos * c_pred + context_neg * (1 - c_pred)
            c_emb_list.append(c_emb.unsqueeze(1))
        return torch.cat(c_emb_list, axis=1), torch.cat(c_pred_list, axis=1)
    

class ConceptEmbedding_intervention(torch.nn.Module):
    def __init__(
            self,
            in_features,
            n_concepts,
            emb_size,
            active_intervention_values=None,
            inactive_intervention_values=None,
            intervention_idxs=None,
            training_intervention_prob=0.25,
    ):
        super(ConceptEmbedding_intervention, self).__init__()
        self.emb_size = emb_size
        self.intervention_idxs = intervention_idxs
        self.training_intervention_prob = training_intervention_prob
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        self.concept_context_generators = torch.nn.ModuleList()
        for i in range(n_concepts):
            self.concept_context_generators.append(torch.nn.Sequential(
                torch.nn.Linear(in_features, 2 * emb_size),
                torch.nn.LeakyReLU(),
            ))
        self.concept_prob_predictor = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, 1),
            torch.nn.Sigmoid(),
        )

        # And default values for interventions here
        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.zeros(n_concepts)

    def _after_interventions(
            self,
            prob,
            concept_idx,
            intervention_idxs=None,
            c_true=None,
            train=False,
    ):
        if train and (self.training_intervention_prob != 0) and (intervention_idxs is None):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(self.ones * self.training_intervention_prob)
            intervention_idxs = torch.nonzero(mask).reshape(-1)
        if (c_true is None) or (intervention_idxs is None):
            return prob
        if concept_idx not in intervention_idxs:
            return prob
        return (c_true[:, concept_idx:concept_idx + 1] * self.active_intervention_values[concept_idx]) + \
            ((c_true[:, concept_idx:concept_idx + 1] - 1) * -self.inactive_intervention_values[concept_idx])

    def forward(self, x, intervention_idxs=None, c=None, train=False, p=None):
        c_emb_list, c_pred_list = [], []
        # We give precendence to inference time interventions arguments
        used_int_idxs = intervention_idxs
        if used_int_idxs is None:
            used_int_idxs = self.intervention_idxs
        for i, context_gen in enumerate(self.concept_context_generators):
            context = context_gen(x)
            c_pred = self.concept_prob_predictor(context)
            
            bernoulli_dist = torch.distributions.Bernoulli(probs=p)
            samples = bernoulli_dist.sample((c_pred.shape[0],1)).to(torch.bool).to('cuda')
            c_true = c[:,i].view(-1, 1)
            c_pred = torch.where(samples==True, c_true, c_pred)
            c_pred_list.append(c_pred)
            
            # Time to check for interventions
            context_pos = context[:, :self.emb_size]
            context_neg = context[:, self.emb_size:]
            
            c_emb = context_pos * c_pred + context_neg * (1 - c_pred)
            c_emb_list.append(c_emb.unsqueeze(1))
        return torch.cat(c_emb_list, axis=1), torch.cat(c_pred_list, axis=1)
    
    
class Cebab_loader:
    def __init__(self, folder, concepts, sentences_per_concept, chuncks, train, val, test, 
                       max_length, tokenizer, pseudo_labeling, model_name, labeler_tokenizer):
        self.folder = folder
        self.train = train
        self.val = val
        self.test = test 
        self.chuncks = chuncks
        self.tokenized_train = None
        self.tokenized_val = None
        self.tokenized_test = None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.concepts = concepts
        self.sentences_per_concept = sentences_per_concept
        self.pseudo_labeling = pseudo_labeling
        self.labeler_tokenizer = labeler_tokenizer
        if self.pseudo_labeling=='sbert':
            self.model = SentenceTransformer(model_name)
        if self.pseudo_labeling=='sbert':
            self.model = SentenceTransformer(model_name)
        elif self.pseudo_labeling=='mistral':
            self.labeler_tokenizer.pad_token_id = self.labeler_tokenizer.eos_token_id
            self.labeler_tokenizer.padding_side = "left"
            # Local model to extract the embedding of the last token
            self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="cuda")
            # mistral api of the same model for text generation. 
            # We decided to use the API instead of running the model locally because it was significantly faster.
            # NASCONDERE LA CHIAVE PRIMA DI CONDIVIDERE
            os.environ['MISTRAL_API_KEY'] = "" # replace with the api key
            api_key = os.environ["MISTRAL_API_KEY"]
            self.api_model = "open-mistral-7b" 
            self.client = MistralClient(api_key=api_key)
        elif self.pseudo_labeling=='mixtral':
            
        
            self.labeler_tokenizer.pad_token_id = self.labeler_tokenizer.eos_token_id
            self.labeler_tokenizer.padding_side = "left"
            # Local model to extract the embedding of the last token
            self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="cuda")            
            
            
            # Local quantized model Mixtral-8x7B to extract the embedding of the last token
            #from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
            #self.labeler_tokenizer.pad_token_id = self.labeler_tokenizer.eos_token_id
            #self.model = HQQModelForCausalLM.from_quantized(model_name)
            #HQQLinear.set_backend(HQQBackend.PYTORCH_COMPILE) 
            #os.environ['TOKENIZERS_PARALLELISM'] = "false"
            
            # self.model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1', load_in_4bit=True, device_map="cuda")   
            # self.labeler_tokenizer.pad_token_id = self.labeler_tokenizer.eos_token_id
            # self.labeler_tokenizer.padding_side = "left"

            # NASCONDERE LA CHIAVE PRIMA DI CONDIVIDERE
            os.environ['MISTRAL_API_KEY'] = "" # replace with api key
            api_key = os.environ["MISTRAL_API_KEY"]
            self.api_model = "open-mixtral-8x7b" 
            self.client = MistralClient(api_key=api_key)
        else:
            raise ValueError(f'{self.pseudo_labeling} is not a supported labeling strategy.')

    def process(self, elem):
        if elem in ['Positive']:
            return 1
        else:
            return 0

    def process2(self, elem):
        if elem == 'Positive':
            return 1
        else:
            return 0 
    
    def find_first_number_after_substring(self, input_string, substring):
        # Create a regular expression pattern to match the substring followed by a number
        pattern = re.compile(r'{}.*?(\d+)'.format(re.escape(substring)))

        # Search for the pattern in the input string
        match = pattern.search(input_string)

        if match:
            # Extract the number from the matched group
            number = match.group(1)
            if int(number) in [0,1]:
                return int(number)
            return 0
        else:
            return 0  # If no match is found   
    
    def preprocess_function(self, examples):
        model_inputs = self.tokenizer(
            examples["review"],
            max_length=self.max_length,
            truncation=True,
            padding = 'max_length'
        )
        model_inputs["labels"] = examples["bin_rating"]
        
        input_ids = torch.Tensor(model_inputs['input_ids'])
        cos_scores = torch.zeros(input_ids.shape[0], len(self.concepts), 2)
        
        if self.pseudo_labeling=='sbert':
            # iterate over samples
            for i in tqdm(range(input_ids.shape[0]), position=0, leave=True):
                if self.chuncks:
                    sentences = sent_tokenize(self.tokenizer.decode(input_ids[i,:], skip_special_tokens=True))
                else:
                    sentences = [self.tokenizer.decode(input_ids[i,:], skip_special_tokens=True)]
                # iterate over concepts
                for j, concept_sentences in enumerate(self.concepts):
                    neg_score = -1 # minimum value for cosine similarity
                    # iterate over negative samples
                    for concept in concept_sentences[:self.sentences_per_concept]:
                        embedded_concept = self.model.encode(concept, convert_to_tensor=True)
                        for sentence in sentences:
                            value = util.cos_sim(self.model.encode(sentence, convert_to_tensor=True), 
                                                 embedded_concept).squeeze().cpu()
                            if value>neg_score:
                                neg_score=value
                    pos_score = -1 # minimum value for cosine similarity  
                    # iterate over positive samples
                    for concept in concept_sentences[self.sentences_per_concept:]:
                        embedded_concept = self.model.encode(concept, convert_to_tensor=True)
                        for sentence in sentences:
                            value = util.cos_sim(self.model.encode(sentence, convert_to_tensor=True), 
                                                 embedded_concept).squeeze().cpu()
                            if value>pos_score:
                                pos_score=value
                    # score for sample i and concept j (negative in position 0 and positive in position 1)
                    cos_scores[i,j,0] = neg_score
                    cos_scores[i,j,1] = pos_score
            model_inputs['cos_scores'] = cos_scores
        elif self.pseudo_labeling in ['mistral', 'mixtral']:
            predictions = []
            for review in tqdm(examples["review"], position=0, leave=True):
                prompt = f"""In a dataset of restaurant reviews there are 4 possible concepts: good food, good ambiance, good service and good noise. 
                Given a certain review, you have to detect if those concepts are present or not in the review. 
                Answer format: good food:score, good ambiance:score, good service:score, good noise:score. 
                Do not add any text other than that specified by the answer format. 
                The score should be equal to 1 if the concept is present or zero otherwise, no other values are accepted.
                The following are examples:
                review: 'The food was delicious but the service fantastic'.
                answer: good food:1, good ambiance:0, good service:1, good noise:0

                review: 'The staff was very rough but the restaurant decorations were great. Other than that there was very relaxing background music'.
                answer: good food:0, good ambiance:1, good service:0, good noise:1

                Now it's your turn:
                review: {review}
                answer:
                """
                messages = [
                    ChatMessage(role="user", content=prompt, temperature=0)
                ]
                chat_response = self.client.chat(
                    model=self.api_model,
                    messages=messages,
                )
                predictions.append(chat_response.choices[0].message.content)
            processed_predictions = torch.zeros(input_ids.shape[0], 4)
            for i, answer in enumerate(predictions):
                #split = answer.split(':')
                food = self.find_first_number_after_substring(answer, 'good food') #int(split[1][0])
                ambiance = self.find_first_number_after_substring(answer, 'good ambiance') #int(split[2][0])
                service = self.find_first_number_after_substring(answer, 'good service') #int(split[3][0])
                noise = self.find_first_number_after_substring(answer, 'good noise') #int(split[4])
                processed_predictions[i,:] = torch.Tensor([food, ambiance, service, noise])
            model_inputs['cos_scores'] = processed_predictions

        model_inputs['food'] = examples['food']
        model_inputs['ambiance'] = examples['ambiance']
        model_inputs['service'] = examples['service']
        model_inputs['noise'] = examples['noise']
        if self.pseudo_labeling == 'sbert':
            embeddings = self.model.encode(examples['review'], convert_to_tensor=True)
        elif self.pseudo_labeling in ['mistral', 'mixtral']:
            template = 'This_sentence_:_"*sent_0*"_means_in_one_word:"'
            embeddings = torch.zeros(input_ids.shape[0], 4096).to('cuda')
            for idx, review in tqdm(enumerate(examples["review"]), position=0, leave=True):
                inputs = self.labeler_tokenizer([template.replace('*sent_0*', i).replace('_', ' ') for i in [review]], padding=True,  return_tensors="pt")
                with torch.no_grad():
                    embeddings[idx, :] = self.model(**inputs.to('cuda'), output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            
        model_inputs['embedded_reviews'] = embeddings
        return model_inputs     
    
    def load(self):
        folder = self.folder
        train = self.train
        val = self.val
        test = self.test
        train_set = pd.read_csv(folder+'/'+train)
        val_set = pd.read_csv(folder+'/'+val)
        test_set = pd.read_csv(folder+'/'+test)
        train_set['food'] = train_set.apply(lambda row: self.process(row['food']), axis=1)
        train_set['ambiance'] = train_set.apply(lambda row: self.process(row['ambiance']), axis=1)
        train_set['service'] = train_set.apply(lambda row: self.process(row['service']), axis=1)
        train_set['noise'] = train_set.apply(lambda row: self.process(row['noise']), axis=1)
        train_set['bin_rating'] = train_set.apply(lambda row: self.process2(row['bin_rating']), axis=1)

        val_set['food'] = val_set.apply(lambda row: self.process(row['food']), axis=1)
        val_set['ambiance'] = val_set.apply(lambda row: self.process(row['ambiance']), axis=1)
        val_set['service'] = val_set.apply(lambda row: self.process(row['service']), axis=1)
        val_set['noise'] = val_set.apply(lambda row: self.process(row['noise']), axis=1)
        val_set['bin_rating'] = val_set.apply(lambda row: self.process2(row['bin_rating']), axis=1)

        test_set['food'] = test_set.apply(lambda row: self.process(row['food']), axis=1)
        test_set['ambiance'] = test_set.apply(lambda row: self.process(row['ambiance']), axis=1)
        test_set['service'] = test_set.apply(lambda row: self.process(row['service']), axis=1)
        test_set['noise'] = test_set.apply(lambda row: self.process(row['noise']), axis=1)
        test_set['bin_rating'] = test_set.apply(lambda row: self.process2(row['bin_rating']), axis=1)     
        
        train_ds = Dataset.from_pandas(train_set)
        val_ds = Dataset.from_pandas(val_set)
        test_ds = Dataset.from_pandas(test_set)
        
        tokenized_train = train_ds.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_ds.column_names,
        )

        tokenized_val = val_ds.map(
            self.preprocess_function,
            batched=True,
            remove_columns=val_ds.column_names,
        )

        tokenized_test = test_ds.map(
            self.preprocess_function,
            batched=True,
            remove_columns=test_ds.column_names,
        )       
        
        self.tokenized_train = tokenized_train
        self.tokenized_val = tokenized_val
        self.tokenized_test = tokenized_test
        
    def collator(self, batch_train=32, batch_val_test=16):
        data_collator = CustomDataCollator(self.tokenizer, self.concepts)
        loaded_train = DataLoader(
            self.tokenized_train, collate_fn=data_collator, batch_size=batch_train, 
            shuffle=False) 
        loaded_val = DataLoader(
            self.tokenized_val, collate_fn=data_collator, batch_size=batch_val_test, 
            shuffle=False)
        loaded_test = DataLoader(
            self.tokenized_test, collate_fn=data_collator, batch_size=batch_val_test, 
            shuffle=False)
        return loaded_train, loaded_val, loaded_test

    
class CustomDataCollator:
    def __init__(self, tokenizer, concepts):
        self.tokenizer = tokenizer
        self.concepts = concepts
        
    def __call__(self, batch):
        input_ids = torch.Tensor([example['input_ids'] for example in batch])
        token_type_ids = torch.Tensor([example['token_type_ids'] for example in batch])
        attention_mask = torch.Tensor([example['attention_mask'] for example in batch])
        embedded_reviews = torch.Tensor([example['embedded_reviews'] for example in batch])
        labels = torch.Tensor([example['labels'] for example in batch])
        cos_scores = torch.Tensor([example['cos_scores'] for example in batch])
        food = torch.Tensor([example['food'] for example in batch])
        ambiance = torch.Tensor([example['ambiance'] for example in batch])
        service = torch.Tensor([example['service'] for example in batch])
        noise = torch.Tensor([example['noise'] for example in batch])
        
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'embedded_reviews': embedded_reviews,
            'labels': labels,
            'concept_score': cos_scores,
            'food': food,
            'ambiance': ambiance,
            'service': service,
            'noise': noise
        }
    

class Drug_loader:
    def __init__(self, train_folder, test_folder, concepts, sentences_per_concept, chuncks, max_length, 
                 tokenizer, pseudo_labeling, model_name, labeler_tokenizer, seed):
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.chuncks = chuncks
        self.tokenized_train = None
        self.tokenized_val = None
        self.tokenized_test = None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.concepts = concepts
        self.sentences_per_concept = sentences_per_concept
        self.pseudo_labeling = pseudo_labeling
        self.labeler_tokenizer = labeler_tokenizer
        self.seed = seed
        if self.pseudo_labeling=='sbert':
            self.model = SentenceTransformer(model_name)
        elif self.pseudo_labeling=='mistral':
            self.labeler_tokenizer.pad_token_id = self.labeler_tokenizer.eos_token_id
            self.labeler_tokenizer.padding_side = "left"
            # Local model to extract the embedding of the last token
            self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="cuda")
            # mistral api of the same model for text generation. 
            # We decided to use the API instead of running the model locally because it was significantly faster.
            # NASCONDERE LA CHIAVE PRIMA DI CONDIVIDERE
            os.environ['MISTRAL_API_KEY'] = "" # replace with the api key
            api_key = os.environ["MISTRAL_API_KEY"]
            self.api_model = "open-mistral-7b" 
            self.client = MistralClient(api_key=api_key)
        elif self.pseudo_labeling=='mixtral':
            
        
            self.labeler_tokenizer.pad_token_id = self.labeler_tokenizer.eos_token_id
            self.labeler_tokenizer.padding_side = "left"
            # Local model to extract the embedding of the last token
            self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="cuda")            
            
            
            # Local quantized model Mixtral-8x7B to extract the embedding of the last token
            #from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
            #self.labeler_tokenizer.pad_token_id = self.labeler_tokenizer.eos_token_id
            #self.model = HQQModelForCausalLM.from_quantized(model_name)
            #HQQLinear.set_backend(HQQBackend.PYTORCH_COMPILE) 
            #os.environ['TOKENIZERS_PARALLELISM'] = "false"
            
            # self.model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1', load_in_4bit=True, device_map="cuda")   
            # self.labeler_tokenizer.pad_token_id = self.labeler_tokenizer.eos_token_id
            # self.labeler_tokenizer.padding_side = "left"

            # NASCONDERE LA CHIAVE PRIMA DI CONDIVIDERE
            os.environ['MISTRAL_API_KEY'] = "" # replace with api key
            api_key = os.environ["MISTRAL_API_KEY"]
            self.api_model = "open-mixtral-8x7b" 
            self.client = MistralClient(api_key=api_key)
        else:
            raise ValueError(f'{self.pseudo_labeling} is not a supported labeling strategy.')
    
    def find_first_number_after_substring(self, input_string, substring):
        # Create a regular expression pattern to match the substring followed by a number
        pattern = re.compile(r'{}.*?(\d+)'.format(re.escape(substring)))
        # Search for the pattern in the input string
        match = pattern.search(input_string)
        if match:
            # Extract the number from the matched group
            number = match.group(1)
            return int(number)
        else:
            return 0    
    
    def preprocess_function(self, examples):
        model_inputs = self.tokenizer(
            examples["review"],
            max_length=self.max_length,
            truncation=True,
            padding = 'max_length'
        )
        model_inputs['effectiveness'] = examples['effectiveness']
        model_inputs['sideEffects'] = examples['sideEffects']
        model_inputs['labels'] = examples['rating']
        
        input_ids = torch.Tensor(model_inputs['input_ids'])
        cos_scores = torch.zeros(input_ids.shape[0], len(self.concepts), 2)
        
        if self.pseudo_labeling=='sbert':
            # iterate over samples
            for i in tqdm(range(input_ids.shape[0]), position=0, leave=True):
                if self.chuncks:
                    sentences = sent_tokenize(self.tokenizer.decode(input_ids[i,:], skip_special_tokens=True))
                else:
                    sentences = [self.tokenizer.decode(input_ids[i,:], skip_special_tokens=True)]
                # iterate over concepts
                for j, concept_sentences in enumerate(self.concepts):
                    neg_score = -1 # minimum value for cosine similarity
                    # iterate over negative samples
                    for concept in concept_sentences[:self.sentences_per_concept]:
                        embedded_concept = self.model.encode(concept, convert_to_tensor=True)
                        for sentence in sentences:
                            value = util.cos_sim(self.model.encode(sentence, convert_to_tensor=True), 
                                                 embedded_concept).squeeze().cpu()
                            if value>neg_score:
                                neg_score=value
                    pos_score = -1 # minimum value for cosine similarity  
                    # iterate over positive samples
                    for concept in concept_sentences[self.sentences_per_concept:]:
                        embedded_concept = self.model.encode(concept, convert_to_tensor=True)
                        for sentence in sentences:
                            value = util.cos_sim(self.model.encode(sentence, convert_to_tensor=True), 
                                                 embedded_concept).squeeze().cpu()
                            if value>pos_score:
                                pos_score=value
                    # score for sample i and concept j (negative in position 0 and positive in position 1)
                    cos_scores[i,j,0] = neg_score
                    cos_scores[i,j,1] = pos_score
            model_inputs['cos_scores'] = cos_scores
        elif self.pseudo_labeling in ['mistral', 'mixtral']:
            predictions = []
            for review in tqdm(examples["review"], position=0, leave=True):
                prompt = f"""In a dataset of drug reviews there are 2 possible concepts: 
                - effectiveness: 1 if the drug was highly effective and 0 if it was marginally or not effective, 
                - side effects: 1 if the drug gave side effects and 0 otherwise. 
                Given a certain review, you have to detect if those concepts are present or not in the review. 
                Answer format: effectveness: score, side effects: score. 
                Do not add any text other than that specified by the answer format. 
                The score should be equal to 1 if the concept is present or zero otherwise, no other values are accepted.
                The following are examples:
                review: 'The medicine worked wonders for me. However, I did experience some side effects. Despite this, I still found it easy to use and incredibly effective'.
                answer: effectiveness: 1, side effects: 1

                review: 'Not only did it fail to alleviate my symptoms, but it also led to unpleasant side effects'.
                answer: effectiveness: 0, side effects: 1

                Now it's your turn:
                review: {review}
                answer:
                """
                messages = [
                    ChatMessage(role="user", content=prompt, temperature=0)
                ]
                chat_response = self.client.chat(
                    model=self.api_model,
                    messages=messages,
                )
                predictions.append(chat_response.choices[0].message.content)
            processed_predictions = torch.zeros(input_ids.shape[0], len(self.concepts))
            for i, answer in enumerate(predictions):
                #split = answer.split(':')
                effectiveness = self.find_first_number_after_substring(answer, 'effectiveness') #int(split[1][0])
                sideEffects = self.find_first_number_after_substring(answer, 'side effects') #int(split[2][0])
                processed_predictions[i,:] = torch.Tensor([effectiveness, sideEffects])
            model_inputs['cos_scores'] = processed_predictions

        if self.pseudo_labeling == 'sbert':
            embeddings = self.model.encode(examples['review'], convert_to_tensor=True)
        elif self.pseudo_labeling in ['mistral', 'mixtral']:
            template = 'This_sentence_:_"*sent_0*"_means_in_one_word:"'
            embeddings = torch.zeros(input_ids.shape[0], 4096).to('cuda')
            for idx, review in tqdm(enumerate(examples["review"]), position=0, leave=True):
                inputs = self.labeler_tokenizer([template.replace('*sent_0*', i).replace('_', ' ') for i in [review]], padding=True,  return_tensors="pt")
                with torch.no_grad():
                    embeddings[idx, :] = self.model(**inputs.to('cuda'), output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            
        model_inputs['embedded_reviews'] = embeddings
        return model_inputs     
    
    def prepare_df(self, folder):
        df = pd.read_csv(folder, delimiter='\t')
        #pd.set_option('future.no_silent_downcasting', True)
        # Mapping the values
        replacement_map = {
            'Highly Effective': 1.0,
            'Considerably Effective': 1.0,
            'Moderately Effective': 1.0,
            'Ineffective': 0.0,
            'Marginally Effective': 0.0
        }
        # Apply the mapping to the 'effectiveness' column
        df['effectiveness'] = df['effectiveness'].replace(replacement_map)
        # Mapping the values
        replacement_map = {
            'Mild Side Effects': 1.0,
            'No Side Effects': 0.0,
            'Moderate Side Effects': 1.0,
            'Severe Side Effects': 1.0,
            'Extremely Severe Side Effects': 1.0
        }
        # Apply the mapping to the 'sideEffects' column
        df['sideEffects'] = df['sideEffects'].replace(replacement_map)#.infer_objects(copy=False)
        # Define the bins and labels for the classes
        bins = [0, 6, 8, 10]
        labels = ['0', '1', '2']
        # Bin the 'rating' column into the specified classes
        df['rating'] = pd.cut(df['rating'], bins=bins, labels=labels, include_lowest=True)
        #df['review'] = df['commentsReview'] + ' ' + df['benefitsReview'] + ' ' + df['sideEffectsReview']
        df['review'] = df['benefitsReview'] + ' ' + df['sideEffectsReview']
        # Selecting specific columns
        df = df[['rating', 'effectiveness', 'sideEffects', 'review']]
        df['rating'] = df['rating'].astype('int64')
        df['effectiveness'] = df['effectiveness'].astype('int64')
        df['sideEffects'] = df['sideEffects'].astype('int64')
        df['review'] = df['review'].astype('str')
        return df
    
    def load(self):
        train_df = self.prepare_df(self.train_folder)
        # Substitute 'Test' with 'train'
        test_df = self.prepare_df(self.test_folder)
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=self.seed)
        train_ds = Dataset.from_pandas(train_df)
        val_ds = Dataset.from_pandas(val_df)
        test_ds = Dataset.from_pandas(test_df)
        
        tokenized_train = train_ds.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_ds.column_names,
        )
        tokenized_val = val_ds.map(
            self.preprocess_function,
            batched=True,
            remove_columns=val_ds.column_names,
        )
        tokenized_test = test_ds.map(
            self.preprocess_function,
            batched=True,
            remove_columns=test_ds.column_names,
        )       
        self.tokenized_train = tokenized_train
        self.tokenized_val = tokenized_val
        self.tokenized_test = tokenized_test
        
    def collator(self, batch_train=32, batch_val_test=16):
        data_collator = DrugDataCollator(self.tokenizer, self.concepts)
        loaded_train = DataLoader(
            self.tokenized_train, collate_fn=data_collator, batch_size=batch_train, 
            shuffle=False) 
        loaded_val = DataLoader(
            self.tokenized_val, collate_fn=data_collator, batch_size=batch_val_test, 
            shuffle=False)
        loaded_test = DataLoader(
            self.tokenized_test, collate_fn=data_collator, batch_size=batch_val_test, 
            shuffle=False)
        return loaded_train, loaded_val, loaded_test

class DrugDataCollator:
    def __init__(self, tokenizer, concepts):
        self.tokenizer = tokenizer
        self.concepts = concepts
        
    def __call__(self, batch):
        input_ids = torch.Tensor([example['input_ids'] for example in batch])
        token_type_ids = torch.Tensor([example['token_type_ids'] for example in batch])
        attention_mask = torch.Tensor([example['attention_mask'] for example in batch])
        embedded_reviews = torch.Tensor([example['embedded_reviews'] for example in batch])
        labels = torch.Tensor([example['labels'] for example in batch])
        cos_scores = torch.Tensor([example['cos_scores'] for example in batch])
        effectiveness = torch.Tensor([example['effectiveness'] for example in batch])
        sideEffects = torch.Tensor([example['sideEffects'] for example in batch])
        
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'embedded_reviews': embedded_reviews,
            'labels': labels,
            'concept_score': cos_scores,
            'effectiveness': effectiveness,
            'sideEffects': sideEffects,
        }
    
    
class Emo_loader:
    def __init__(self, folder, concepts, sentences_per_concept, chuncks, max_length, 
                 tokenizer, pseudo_labeling, model_name, labeler_tokenizer, seed):
        self.folder = folder
        self.chuncks = chuncks
        self.tokenized_train = None
        self.tokenized_val = None
        self.tokenized_test = None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.concepts = concepts
        self.sentences_per_concept = sentences_per_concept
        self.pseudo_labeling = pseudo_labeling
        self.labeler_tokenizer = labeler_tokenizer
        self.seed = seed
        if self.pseudo_labeling=='sbert':
            self.model = SentenceTransformer(model_name)
        elif self.pseudo_labeling=='mistral':
            self.labeler_tokenizer.pad_token_id = self.labeler_tokenizer.eos_token_id
            self.labeler_tokenizer.padding_side = "left"
            # Local model to extract the embedding of the last token
            self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="cuda")
            # mistral api of the same model for text generation. 
            # We decided to use the API instead of running the model locally because it was significantly faster.
            # NASCONDERE LA CHIAVE PRIMA DI CONDIVIDERE
            os.environ['MISTRAL_API_KEY'] = "" # replace with the api key
            api_key = os.environ["MISTRAL_API_KEY"]
            self.api_model = "open-mistral-7b" 
            self.client = MistralClient(api_key=api_key)
        elif self.pseudo_labeling=='mixtral':
            
        
            self.labeler_tokenizer.pad_token_id = self.labeler_tokenizer.eos_token_id
            self.labeler_tokenizer.padding_side = "left"
            # Local model to extract the embedding of the last token
            self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="cuda")            
            
            
            # Local quantized model Mixtral-8x7B to extract the embedding of the last token
            #from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
            #self.labeler_tokenizer.pad_token_id = self.labeler_tokenizer.eos_token_id
            #self.model = HQQModelForCausalLM.from_quantized(model_name)
            #HQQLinear.set_backend(HQQBackend.PYTORCH_COMPILE) 
            #os.environ['TOKENIZERS_PARALLELISM'] = "false"
            
            # self.model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1', load_in_4bit=True, device_map="cuda")   
            # self.labeler_tokenizer.pad_token_id = self.labeler_tokenizer.eos_token_id
            # self.labeler_tokenizer.padding_side = "left"

            # NASCONDERE LA CHIAVE PRIMA DI CONDIVIDERE
            os.environ['MISTRAL_API_KEY'] = "" # replace with api key
            api_key = os.environ["MISTRAL_API_KEY"]
            self.api_model = "open-mixtral-8x7b" 
            self.client = MistralClient(api_key=api_key)
        else:
            raise ValueError(f'{self.pseudo_labeling} is not a supported labeling strategy.')
    
    def find_first_number_after_substring(self, input_string, substring):
        # Create a regular expression pattern to match the substring followed by a number
        pattern = re.compile(r'{}.*?(\d+)'.format(re.escape(substring)))
        # Search for the pattern in the input string
        match = pattern.search(input_string)
        if match:
            # Extract the number from the matched group
            number = match.group(1)
            return int(number)
        else:
            return 0    

    def preprocess_function(self, examples):
        model_inputs = self.tokenizer(
            examples["review"],
            max_length=self.max_length,
            truncation=True,
            padding = 'max_length'
        )
        model_inputs['joy'] = examples['GIOIA']
        model_inputs['trust'] = examples['FIDUCIA']
        model_inputs['sadness'] = examples['TRISTEZZA']
        model_inputs['surprise'] = examples['SORPRESA']
        model_inputs['pos'] = examples['POS']
        model_inputs['neg'] = examples['NEG']
        model_inputs['neut'] = examples['NEUT']
        
        input_ids = torch.Tensor(model_inputs['input_ids'])
        cos_scores = torch.zeros(input_ids.shape[0], len(self.concepts), 2)
        
        if self.pseudo_labeling=='sbert':
            # iterate over samples
            for i in tqdm(range(input_ids.shape[0]), position=0, leave=True):
                if self.chuncks:
                    sentences = sent_tokenize(self.tokenizer.decode(input_ids[i,:], skip_special_tokens=True))
                else:
                    sentences = [self.tokenizer.decode(input_ids[i,:], skip_special_tokens=True)]
                # iterate over concepts
                for j, concept_sentences in enumerate(self.concepts):
                    neg_score = -1 # minimum value for cosine similarity
                    # iterate over negative samples
                    #for concept in concept_sentences[:self.sentences_per_concept]:
                    #    embedded_concept = self.model.encode(concept, convert_to_tensor=True)
                    #    for sentence in sentences:
                    #        value = util.cos_sim(self.model.encode(sentence, convert_to_tensor=True), 
                    #                             embedded_concept).squeeze().cpu()
                    #        if value>neg_score:
                    #            neg_score=value
                    pos_score = -1 # minimum value for cosine similarity  
                    # iterate over positive samples
                    for concept in concept_sentences[self.sentences_per_concept:]:
                        embedded_concept = self.model.encode(concept, convert_to_tensor=True)
                        for sentence in sentences:
                            value = util.cos_sim(self.model.encode(sentence, convert_to_tensor=True), 
                                                 embedded_concept).squeeze().cpu()
                            if value>pos_score:
                                pos_score=value
                    # score for sample i and concept j (negative in position 0 and positive in position 1)
                    cos_scores[i,j,0] = neg_score
                    cos_scores[i,j,1] = pos_score
            model_inputs['cos_scores'] = cos_scores
        elif self.pseudo_labeling in ['mistral', 'mixtral']:
            predictions = []
            for review in tqdm(examples["review"], position=0, leave=True):
                prompt = f"""In un dataset che contiene commenti in lingua italiana devi identificare i seguenti concetti: 
                - gioia: 1 se il commento esprime gioia 0 altrimenti, 
                - fiducia: 1 se il commento esprime fiducia 0 altrimenti,
                - tristezza: 1 se il commento esprime tristezza 0 altrimenti,
                - sorpresa: 1 se il commento esprime sorpresa 0 altrimenti.
                
                Dato un determinato commento, devi individuare se questi concetti sono presenti o no nel commento.
                Formato della risposta: gioia: score, fiducia: score, tristezza: score, sorpresa: score.
                Non aggiungere altro testo al di fuori di quello specificato nel formato della risposta.
                Lo score deve essere uguale a 1 se il concetto è presente e 0 altrimenti, non sono accettati altri valori.
                I seguenti sono esempi:
                commento: 'Io non capisco come faccia ad essere fra le ultime questa canzone'.
                risposta: gioia: 0, fiducia: 0, tristezza: 1, sorpresa: 1

                commento: 'Mi piace questa versione di Elettra, più dolce, raffinata, elegante, brava!'
                risposta: gioia: 1, fiducia: 1, tristezza: 0, sorpresa: 1

                Ora è il tuo turno:
                commento: {review}
                risposta:
                """
                
                messages = [
                    ChatMessage(role="user", content=prompt, temperature=0)
                ]
                chat_response = self.client.chat(
                    model=self.api_model,
                    messages=messages,
                )
                predictions.append(chat_response.choices[0].message.content)
            processed_predictions = torch.zeros(input_ids.shape[0], len(self.concepts))
            for i, answer in enumerate(predictions):
                #split = answer.split(':')
                gioia = self.find_first_number_after_substring(answer, 'gioia') #int(split[1][0])
                fiducia = self.find_first_number_after_substring(answer, 'fiducia') #int(split[1][0])
                tristezza = self.find_first_number_after_substring(answer, 'tristezza') #int(split[1][0])
                sorpresa = self.find_first_number_after_substring(answer, 'sorpresa') #int(split[2][0])
                processed_predictions[i,:] = torch.Tensor([gioia, fiducia, tristezza, sorpresa])
            model_inputs['cos_scores'] = processed_predictions

        if self.pseudo_labeling == 'sbert':
            embeddings = self.model.encode(examples['review'], convert_to_tensor=True)
        elif self.pseudo_labeling in ['mistral', 'mixtral']:
            template = 'This_sentence_:_"*sent_0*"_means_in_one_word:"'
            embeddings = torch.zeros(input_ids.shape[0], 4096).to('cuda')
            for idx, review in tqdm(enumerate(examples["review"]), position=0, leave=True):
                inputs = self.labeler_tokenizer([template.replace('*sent_0*', i).replace('_', ' ') for i in [review]], padding=True,  return_tensors="pt")
                with torch.no_grad():
                    embeddings[idx, :] = self.model(**inputs.to('cuda'), output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            
        model_inputs['embedded_reviews'] = embeddings
        return model_inputs     
    
    def load(self):
        folder = self.folder
        df = pd.read_csv(self.folder, delimiter='\t')
        df = df.drop(columns=['type', 'title', 'URL', 'EMOTIONS'])
        df = df.drop(columns=['TREPIDAZIONE', 'SARCASM', 'DISGUSTO', 'PAURA', 'RABBIA'])
        # Drop rows where any value is None
        df.dropna(inplace=True)
        # Remove all rows where 'UNRELATED' is 1
        df = df[df['UNRELATED'] != 1]
        df = df.rename(columns={'comment': 'review'})
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=self.seed)
        val_df, test_df = train_test_split(temp_df, test_size=0.67, random_state=self.seed)
        train_ds = Dataset.from_pandas(train_df)
        val_ds = Dataset.from_pandas(val_df)
        test_ds = Dataset.from_pandas(test_df)
        
        tokenized_train = train_ds.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_ds.column_names,
        )
        tokenized_val = val_ds.map(
            self.preprocess_function,
            batched=True,
            remove_columns=val_ds.column_names,
        )
        tokenized_test = test_ds.map(
            self.preprocess_function,
            batched=True,
            remove_columns=test_ds.column_names,
        )       
        self.tokenized_train = tokenized_train
        self.tokenized_val = tokenized_val
        self.tokenized_test = tokenized_test
        
    def collator(self, batch_train=32, batch_val_test=16):
        data_collator = EmoDataCollator(self.tokenizer, self.concepts)
        loaded_train = DataLoader(
            self.tokenized_train, collate_fn=data_collator, batch_size=batch_train, 
            shuffle=False) 
        loaded_val = DataLoader(
            self.tokenized_val, collate_fn=data_collator, batch_size=batch_val_test, 
            shuffle=False)
        loaded_test = DataLoader(
            self.tokenized_test, collate_fn=data_collator, batch_size=batch_val_test, 
            shuffle=False)
        return loaded_train, loaded_val, loaded_test

    
class EmoDataCollator:
    def __init__(self, tokenizer, concepts):
        self.tokenizer = tokenizer
        self.concepts = concepts
        
    def __call__(self, batch):
        input_ids = torch.Tensor([example['input_ids'] for example in batch])
        token_type_ids = torch.Tensor([example['token_type_ids'] for example in batch])
        attention_mask = torch.Tensor([example['attention_mask'] for example in batch])
        embedded_reviews = torch.Tensor([example['embedded_reviews'] for example in batch])
        cos_scores = torch.Tensor([example['cos_scores'] for example in batch])

        surprise = torch.Tensor([example['surprise'] for example in batch])
        joy = torch.Tensor([example['joy'] for example in batch])
        trust = torch.Tensor([example['trust'] for example in batch])
        sadness = torch.Tensor([example['sadness'] for example in batch])
        
        one_hot_labels = torch.Tensor([[example['pos'], example['neut'], example['neg']] for example in batch])
        labels = torch.argmax(one_hot_labels, dim=1)
        
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'embedded_reviews': embedded_reviews,
            'labels': labels,
            'concept_score': cos_scores,
            'joy': joy,
            'trust': trust,
            'sadness': sadness,
            'surprise': surprise,
        }
    
    
    
    
    
    
class depressed_loader:
    def __init__(self, folder, concepts, sentences_per_concept, chuncks, max_length, 
                 tokenizer, pseudo_labeling, model_name, labeler_tokenizer, seed):
        self.folder = folder
        self.chuncks = chuncks
        self.tokenized_train = None
        self.tokenized_val = None
        self.tokenized_test = None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.concepts = concepts
        self.sentences_per_concept = sentences_per_concept
        self.pseudo_labeling = pseudo_labeling
        self.labeler_tokenizer = labeler_tokenizer
        self.seed = seed
        if self.pseudo_labeling=='sbert':
            self.model = SentenceTransformer(model_name)
        elif self.pseudo_labeling=='mistral':
            self.labeler_tokenizer.pad_token_id = self.labeler_tokenizer.eos_token_id
            self.labeler_tokenizer.padding_side = "left"
            # Local model to extract the embedding of the last token
            self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="cuda")
            # mistral api of the same model for text generation. 
            # We decided to use the API instead of running the model locally because it was significantly faster.
            # NASCONDERE LA CHIAVE PRIMA DI CONDIVIDERE
            os.environ['MISTRAL_API_KEY'] = "" # replace with the api key
            api_key = os.environ["MISTRAL_API_KEY"]
            self.api_model = "open-mistral-7b" 
            self.client = MistralClient(api_key=api_key)
        elif self.pseudo_labeling=='mixtral':
            
            self.labeler_tokenizer.pad_token_id = self.labeler_tokenizer.eos_token_id
            self.labeler_tokenizer.padding_side = "left"
            # Local model to extract the embedding of the last token
            self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="cuda")            
            
            # Local quantized model Mixtral-8x7B to extract the embedding of the last token
            #from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
            #self.labeler_tokenizer.pad_token_id = self.labeler_tokenizer.eos_token_id
            #self.model = HQQModelForCausalLM.from_quantized(model_name)
            #HQQLinear.set_backend(HQQBackend.PYTORCH_COMPILE) 
            #os.environ['TOKENIZERS_PARALLELISM'] = "false"
            
            # self.model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1', load_in_4bit=True, device_map="cuda")   
            # self.labeler_tokenizer.pad_token_id = self.labeler_tokenizer.eos_token_id
            # self.labeler_tokenizer.padding_side = "left"

            # NASCONDERE LA CHIAVE PRIMA DI CONDIVIDERE
            os.environ['MISTRAL_API_KEY'] = "" # replace with api key
            api_key = os.environ["MISTRAL_API_KEY"]
            self.api_model = "open-mixtral-8x7b" 
            self.client = MistralClient(api_key=api_key)
        else:
            raise ValueError(f'{self.pseudo_labeling} is not a supported labeling strategy.')
    
    def find_first_number_after_substring(self, input_string, substring):
        # Create a regular expression pattern to match the substring followed by a number
        pattern = re.compile(r'{}.*?(\d+)'.format(re.escape(substring)))
        # Search for the pattern in the input string
        match = pattern.search(input_string)
        if match:
            # Extract the number from the matched group
            number = match.group(1)
            return int(number)
        else:
            return 0    

    def preprocess_function(self, examples):
        model_inputs = self.tokenizer(
            examples["review"],
            max_length=self.max_length,
            truncation=True,
            padding = 'max_length'
        )       
        
        input_ids = torch.Tensor(model_inputs['input_ids'])
        cos_scores = torch.zeros(input_ids.shape[0], len(self.concepts), 2)
        
        if self.pseudo_labeling=='sbert':
            # iterate over samples
            for i in tqdm(range(input_ids.shape[0]), position=0, leave=True):
                if self.chuncks:
                    sentences = sent_tokenize(self.tokenizer.decode(input_ids[i,:], skip_special_tokens=True))
                else:
                    sentences = [self.tokenizer.decode(input_ids[i,:], skip_special_tokens=True)]
                # iterate over concepts
                for j, concept_sentences in enumerate(self.concepts):
                    neg_score = -1 # minimum value for cosine similarity
                    # iterate over negative samples
                    #for concept in concept_sentences[:self.sentences_per_concept]:
                    #    embedded_concept = self.model.encode(concept, convert_to_tensor=True)
                    #    for sentence in sentences:
                    #        value = util.cos_sim(self.model.encode(sentence, convert_to_tensor=True), 
                    #                             embedded_concept).squeeze().cpu()
                    #        if value>neg_score:
                    #            neg_score=value
                    pos_score = -1 # minimum value for cosine similarity  
                    # iterate over positive samples
                    for concept in concept_sentences[self.sentences_per_concept:]:
                        embedded_concept = self.model.encode(concept, convert_to_tensor=True)
                        for sentence in sentences:
                            value = util.cos_sim(self.model.encode(sentence, convert_to_tensor=True), 
                                                 embedded_concept).squeeze().cpu()
                            if value>pos_score:
                                pos_score=value
                    # score for sample i and concept j (negative in position 0 and positive in position 1)
                    cos_scores[i,j,0] = neg_score
                    cos_scores[i,j,1] = pos_score
            model_inputs['cos_scores'] = cos_scores
        elif self.pseudo_labeling in ['mistral', 'mixtral']:
            predictions = []
            for review in tqdm(examples["review"], position=0, leave=True):
                prompt = f"""You have to identify the presence or absence of a number of concepts in a certain text. 
                The concepts to be identified are:
                - Self-Deprecation: the text exhibits self-critical or self-deprecating language, expressing feelings of guilt, shame, or inadequacy. 
                - Loss of Interest: diminished pleasure or motivation in the writer's descriptions of hobbies or pursuits.
                - Hopelessness: the writer express feelings of futility or a lack of optimism about their prospects.
                - Sleep Disturbances: the writer mentions insomnia, oversleeping, or disrupted sleep as part of their experience.
                - Appetite Changes: there are references to changes in eating habits.
                - Fatigue: there are references to exhaustion or lethargy.
                Answer format: Self-Deprecation: score, Loss of Interest: score, Hopelessness: score, Sleep Disturbances: score, Appetite Changes: score, Fatigue: score.
                The score has to be 1 if the concept is detected and 0 otherwise. Do not add any other text besides the one specified in the answer format.

                article: {review}
                Answer:
                """
                
                messages = [
                    ChatMessage(role="user", content=prompt, temperature=0)
                ]
                chat_response = self.client.chat(
                    model=self.api_model,
                    messages=messages,
                )
                predictions.append(chat_response.choices[0].message.content)
            processed_predictions = torch.zeros(input_ids.shape[0], len(self.concepts))
            for i, answer in enumerate(predictions):
                #split = answer.split(':')
                Deprecation = self.find_first_number_after_substring(answer, 'Self-Deprecation') #int(split[1][0])
                Loss_of_Interest = self.find_first_number_after_substring(answer, 'Loss of Interest') #int(split[1][0])
                Hopelessness = self.find_first_number_after_substring(answer, 'Hopelessness') #int(split[2][0])
                Sleep_Disturbances = self.find_first_number_after_substring(answer, 'Sleep Disturbances') #int(split[2][0])
                Appetite_Changes = self.find_first_number_after_substring(answer, 'Appetite Changes') #int(split[2][0])
                Fatigue = self.find_first_number_after_substring(answer, 'Fatigue') #int(split[2][0])
                processed_predictions[i,:] = torch.Tensor([Deprecation, Loss_of_Interest, Hopelessness, Sleep_Disturbances, Appetite_Changes, Fatigue])
            model_inputs['cos_scores'] = processed_predictions

        if self.pseudo_labeling == 'sbert':
            embeddings = self.model.encode(examples['review'], convert_to_tensor=True)
        elif self.pseudo_labeling in ['mistral', 'mixtral']:
            template = 'This_sentence_:_"*sent_0*"_means_in_one_word:"'
            embeddings = torch.zeros(input_ids.shape[0], 4096)#.to('cuda')
            for idx, review in tqdm(enumerate(examples["review"]), position=0, leave=True):
                inputs = self.labeler_tokenizer([template.replace('*sent_0*', i).replace('_', ' ') for i in [review]], padding=True,  return_tensors="pt")
                with torch.no_grad():
                    embeddings[idx, :] = self.model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            
        model_inputs['embedded_reviews'] = embeddings
        model_inputs['labels'] = examples['labels']
        return model_inputs     
    
    def load(self):
        
        df = pd.read_csv(f'{self.folder}/depression_dataset_reddit_cleaned 2.csv')
        df.rename(columns={'clean_text': 'review'}, inplace=True)
        df.rename(columns={'is_depression': 'labels'}, inplace=True)

        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['labels'], random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['labels'], random_state=42)
        
        train_ds = Dataset.from_pandas(train_df)
        val_ds = Dataset.from_pandas(val_df)
        test_ds = Dataset.from_pandas(test_df)
        
        tokenized_train = train_ds.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_ds.column_names,
        )
        tokenized_val = val_ds.map(
            self.preprocess_function,
            batched=True,
            remove_columns=val_ds.column_names,
        )
        tokenized_test = test_ds.map(
            self.preprocess_function,
            batched=True,
            remove_columns=test_ds.column_names,
        )       
        self.tokenized_train = tokenized_train
        self.tokenized_val = tokenized_val
        self.tokenized_test = tokenized_test
        
    def collator(self, batch_train=32, batch_val_test=16):
        data_collator = DepressedDataCollator(self.tokenizer)
        loaded_train = DataLoader(
            self.tokenized_train, collate_fn=data_collator, batch_size=batch_train, 
            shuffle=False) 
        loaded_val = DataLoader(
            self.tokenized_val, collate_fn=data_collator, batch_size=batch_val_test, 
            shuffle=False)
        loaded_test = DataLoader(
            self.tokenized_test, collate_fn=data_collator, batch_size=batch_val_test, 
            shuffle=False)
        return loaded_train, loaded_val, loaded_test

    
class DepressedDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        input_ids = torch.Tensor([example['input_ids'] for example in batch])
        token_type_ids = torch.Tensor([example['token_type_ids'] for example in batch])
        attention_mask = torch.Tensor([example['attention_mask'] for example in batch])
        embedded_reviews = torch.Tensor([example['embedded_reviews'] for example in batch])
        cos_scores = torch.Tensor([example['cos_scores'] for example in batch])
        labels = torch.Tensor([example['labels'] for example in batch])
        
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'embedded_reviews': embedded_reviews,
            'labels': labels,
            'concept_score': cos_scores
        }