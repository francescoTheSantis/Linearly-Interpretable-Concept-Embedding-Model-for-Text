import torch
import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import re
import os
from mistralai import Mistral
import openai
from tqdm import tqdm
import pickle


cebab_template = """In a dataset of restaurant reviews there are 4 possible concepts: food, ambiance, service and noise. 
Given a certain review, you have to detect if those concepts are present or not in the review. 
Answer format: food:score, ambiance:score, service:score, noise:score. 
Do not add any text other than that specified by the answer format. 
The score should be equal to 1 if the concept is present or zero otherwise, no other values are accepted.
The following are examples:
review: 'The food was delicious and the service fantastic'.
answer: food:1, ambiance:0, service:1, noise:0

review: 'The staff was very rough but the restaurant decorations were great. Other than that there was very relaxing background music'.
answer: food:0, ambiance:1, service:1, noise:1

Now it's your turn:
review: {review}
answer:
"""

class CustomDataCollator: 
    def __call__(self, batch):
        input_ids = torch.Tensor([example['input_ids'] for example in batch])
        token_type_ids = torch.Tensor([example['token_type_ids'] for example in batch])
        attention_mask = torch.Tensor([example['attention_mask'] for example in batch])
        labels = torch.Tensor([example['labels'] for example in batch])
        food = torch.Tensor([example['food'] for example in batch])
        ambiance = torch.Tensor([example['ambiance'] for example in batch])
        service = torch.Tensor([example['service'] for example in batch])
        noise = torch.Tensor([example['noise'] for example in batch])
        embedded_text = torch.Tensor([example['embedded_text'] for example in batch])
        
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'food': food,
            'ambiance': ambiance,
            'service': service,
            'noise': noise,
            'embedded_text': embedded_text
        }

class Cebab_loader:
    def __init__(self, 
                 folder,  
                 max_length, 
                 encoder='bert-base-uncased', # provide the name of the HuggingFace model (mistralai/Mistral-7B-v0.1,bert-base-uncased, mistralai/Mixtral-8x7B-v0.1)
                 concept_labeler='supervised', # provide the name of the HuggingFace model or supervised to use labeled concepts (mistralai/Mistral-7B-v0.1, mistralai/Mixtral-8x7B-v0.1)
                 use_api_support=False,
                 api_key_mistral=None,
                 api_key_openai=None,
                 storing_folder=None):
        self.folder = folder
        self.train = 'cebab_train.csv'
        self.val = 'cebab_validation.csv'
        self.test = 'cebab_test.csv'
        self.tokenized_train = None
        self.tokenized_val = None
        self.tokenized_test = None
        self.encoder = encoder
        self.use_api_support = use_api_support
        self.concept_labeler = concept_labeler 
        self.max_length = max_length
        self.storing_folder = storing_folder

        if self.storing_folder!=None:
            self.create_folder_if_not_exists(storing_folder)

        if self.encoder in ['bert-base-uncased']:
            print(f'Loading Model: {self.encoder}')
            self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder) #, model_max_length = max_length)
            self.embedding_size = 768
        elif 'Mistral' in self.encoder or 'Mixtral' in self.encoder:
            print(f'Loading Model: {self.encoder}')
            self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder) #, model_max_length = max_length)
            self.encoder_tokenizer.pad_token = self.encoder_tokenizer.eos_token # setting the pad token to the eos token
            self.encoder_model = AutoModelForCausalLM.from_pretrained(encoder, torch_dtype=torch.bfloat16)  # the model is instantiated in bfloat16 to reduce memory occupation
            self.embedding_size = 4096
        else:
            raise ValueError('Encoder not supported')

        if use_api_support and concept_labeler=='supervised':
            raise ValueError('API support not available for supervised concept labeling')
        elif use_api_support:  
            if 'Mistral' in self.concept_labeler:
                self.api_model = 'open-mistral-7b'
                self.client = Mistral(api_key=api_key_mistral)
            elif 'Mixtral' in self.concept_labeler:
                self.api_model = 'open-mixtral-8x7b'
                self.client = Mistral(api_key=api_key_mistral)
            elif 'gpt' in self.concept_labeler:
                self.api_key = api_key_openai
                self.api_model = 'gpt-3.5-turbo'
            else:
                raise ValueError('API model not supported')
        elif ('Mistral' in self.concept_labeler or 'Mixtral' in self.concept_labeler) and not use_api_support:
            if self.concept_labeler == self.encoder:
                self.labeler_tokenizer = self.encoder_tokenizer
                self.labeler_model = self.encoder_model
            else:
                self.labeler_tokenizer = AutoTokenizer.from_pretrained(encoder, padding=True) 
                self.labeler_model = AutoModelForCausalLM.from_pretrained(encoder, torch_dtype=torch.bfloat16) # the model is instantiated in bfloat16 to reduce memory occupation
            self.embedding_size = 4096       
        elif concept_labeler == 'supervised':
            pass 
        else:
            raise ValueError(f'Combination of use_api_support ({use_api_support}) and concept_labeler ({concept_labeler}) not supported')

    def create_folder_if_not_exists(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f'Folder "{folder_path}" created.')
        else:
            print(f'Folder "{folder_path}" already exists.')

    def process(self, elem):
        if elem in ['Negative']:
            return 0
        if elem in ['unknown']:
            return 1
        else:
            return 2

    def process2(self, elem):
        if elem == 'Positive':
            return 1
        else:
            return 0 
     
    def find_first_number_after_substring(self, input_string, substring):
        pattern = re.compile(r'{}.*?(\d+)'.format(re.escape(substring)))
        match = pattern.search(input_string)
        if match:
            number = match.group(1)
            if int(number) in [0,1]:
                return int(number)
            return 0
        else:
            return 0   
        
    def run_query_on_local_LLM(self, query, max_new_tokens=100):
        inputs = self.labeler_tokenizer(query, return_tensors="pt")
        outputs = self.labeler_model.generate(**inputs, 
                                              eos_token_id=self.labeler_tokenizer.eos_token_id,
                                              pad_token_id=self.labeler_tokenizer.eos_token_id, 
                                              max_new_tokens=max_new_tokens)
        generated_text = self.labeler_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # eliminate the query from the response
        response = generated_text[len(query):].strip()
        return response

    def preprocess_function(self, examples):
        model_inputs = self.encoder_tokenizer(
            examples["review"],
            max_length=self.max_length,
            truncation=True,
            padding = 'max_length'
        )
        model_inputs["labels"] = examples["bin_rating"]

        if self.concept_labeler == 'supervised':
            model_inputs['food'] = examples['food']
            model_inputs['ambiance'] = examples['ambiance']
            model_inputs['service'] = examples['service']
            model_inputs['noise'] = examples['noise']
        elif 'Mistral' in self.concept_labeler or 'Mixtral' in self.concept_labeler:
            food, ambiance, service, noise = [], [], [], []
            print('Producing concept annotations')
            predictions = []
            if self.use_api_support:
                if 'Mistral' in self.concept_labeler or 'Mixtral':
                    for review in tqdm(examples["review"], position=0, leave=True):
                        prompt = cebab_template.format(review=review)
                        chat_response = self.client.chat.complete(
                            model= self.api_model,
                            messages = [
                                {
                                    "role": "user",
                                    "content": prompt,
                                },
                            ]
                        )
                        print(chat_response.choices[0].message.content)
                        predictions.append(chat_response.choices[0].message.content)  
                elif 'gpt' in self.concept_labeler:
                    for review in tqdm(examples["review"], position=0, leave=True):
                        prompt = cebab_template.format(review=review)
                        response = openai.Completion.create(
                            engine=self.api_model,
                            prompt=prompt,
                            max_tokens=100
                        )
                        predictions.append(response.choices[0].text)   
            else:
                for review in tqdm(examples["review"], total=len(examples["review"]), position=0, leave=True):
                    response = self.run_query_on_local_LLM(cebab_template.format(review=examples["review"][0]))
                    predictions.append(response)   
            for _, answer in enumerate(predictions):
                food.append(self.find_first_number_after_substring(answer, 'food'))
                ambiance.append(self.find_first_number_after_substring(answer, 'ambiance'))
                service.append(self.find_first_number_after_substring(answer, 'service'))
                noise.append(self.find_first_number_after_substring(answer, 'noise'))
            model_inputs['food'] = food
            model_inputs['ambiance'] = ambiance
            model_inputs['service'] = service
            model_inputs['noise'] = noise
        else:
            raise ValueError('Combination of concept_labeler and use_api_support not supported')

        # get text embedding using the encoder model only if the encoder used is either mistral or mixtral
        input_ids = torch.Tensor(model_inputs['input_ids'])
        if 'Mistral' in self.encoder or 'Mixtral' in self.encoder:
            with torch.no_grad():
                template = 'This sentence: "*sent_0*" means in one word:"'
                embeddings = torch.zeros(input_ids.shape[0], self.embedding_size)
                print('Producing text embeddings')
                total_reviews = len(examples["review"])
                for idx, review in tqdm(enumerate(examples["review"]), total=total_reviews, position=0, leave=True):
                    inputs = self.encoder_tokenizer([template.replace('*sent_0*', i) for i in [review]], padding=True,  return_tensors="pt")
                    with torch.no_grad():
                        embeddings[idx, :] = self.encoder_model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
                model_inputs['embedded_text'] = embeddings
        else:
            embeddings = torch.zeros(input_ids.shape[0], self.embedding_size)
            model_inputs['embedded_text'] = embeddings
        return model_inputs

    def load(self):
        train_set = pd.read_csv(self.folder+'/'+self.train)
        val_set = pd.read_csv(self.folder+'/'+self.val)
        test_set = pd.read_csv(self.folder+'/'+self.test)

        split_list = [train_set, val_set, test_set]
        for split in split_list:
            split['food'] = split.apply(lambda row: self.process(row['food']), axis=1)
            split['ambiance'] = split.apply(lambda row: self.process(row['ambiance']), axis=1)
            split['service'] = split.apply(lambda row: self.process(row['service']), axis=1)
            split['noise'] = split.apply(lambda row: self.process(row['noise']), axis=1)
            split['bin_rating'] = split.apply(lambda row: self.process2(row['bin_rating']), axis=1)

        '''
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
        '''

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

    def collator(self, batch_train=64, batch_val_test=32):
        data_collator = CustomDataCollator() 
        loaded_train = DataLoader(self.tokenized_train, collate_fn=data_collator, batch_size=batch_train, shuffle=False) 
        loaded_val = DataLoader(self.tokenized_val, collate_fn=data_collator, batch_size=batch_val_test, shuffle=False)
        loaded_test = DataLoader(self.tokenized_test, collate_fn=data_collator, batch_size=batch_val_test, shuffle=False)
        
        if self.storing_folder!=None:
            with open(f'{self.storing_folder}/loaded_train.pkl', 'wb') as f:
                pickle.dump(loaded_train, f)
            with open(f'{self.storing_folder}/loaded_val.pkl', 'wb') as f:
                pickle.dump(loaded_val, f)
            with open(f'{self.storing_folder}/loaded_test.pkl', 'wb') as f:    
                pickle.dump(loaded_test, f)
        return loaded_train, loaded_val, loaded_test