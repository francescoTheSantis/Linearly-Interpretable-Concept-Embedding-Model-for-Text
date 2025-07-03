from src.loaders.llm_client import llm_client 
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from src.metrics import f1_acc_metrics
import pandas as pd

class LLMClassifier(llm_client):
    def __init__(self, 
                 class_dict, 
                 LLM='gpt-4o', 
                 temperature=0, 
                 max_tries=1, 
                 examples=None,
                 metadata=None,
                 tokenizer=None,
                 storing_path=None,
                 use_examples=False,
                 istruction_prompt=None
                 ):
        super().__init__(LLM, temperature, max_tries)
        self.class_dict = class_dict
        self.examples = examples
        self.use_examples = use_examples
        if tokenizer != None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) 
        else:
            raise ValueError("Tokenizer must be provided if not using default.")
        self.storing_path = storing_path
        self.metadata = metadata
        self.istruction_prompt = istruction_prompt

    def _default_answer(self):
        """
        To avoid errors when the LLM does not return a valid answer,
        we define a default answer.
        This should be the index of the class that is considered the default or fallback.
        In this case, we assume the default class index is 0.
        """
        return 0
    
    def _parse_answer(self, answer, return_tensor=False):
        """
        Parse the answer from the LLM response.
        """
        answer_part = answer.strip()
        # Convert it to an integer
        try:
            llm_prediction = int(answer_part)
        except ValueError:
            # If conversion fails, return the default answer
            llm_prediction = self._default_answer()
        if return_tensor:
            # Convert the class index to a tensor
            llm_prediction = torch.tensor(llm_prediction, dtype=torch.int64)
        return llm_prediction

    def ask(self, question, return_tensor=False):
        """
        Ask the llm to classify a sample
        """
        # replace the <question> in the istruction_prompt with the actual question
        # and the <class_dict> with the actual class_dict
        updated_prompt = self.istruction_prompt.replace('<query>', question)
        updated_prompt = updated_prompt.replace('{class_dict}', str(self.class_dict))
        if self.use_examples:
            updated_prompt = updated_prompt.replace('{examples}', self.examples)
        else:
            # remove the examples part from the prompt
            updated_prompt = updated_prompt.replace('{examples}', 'No examples provided.')

        # invoke the llm
        answer = self._invoke(updated_prompt, temperature=self.temperature)

        return self._parse_answer(answer, return_tensor)
    
    def test(self, data_loader):
        """
        Test the llm classifier on a dataset.
        """
        predictions = []
        labels = []
        for batch in data_loader:
            ids = batch[0]
            targets = batch[-2]
            for id, target in tqdm(zip(ids, targets)):
                decoded_sample = self.tokenizer.decode(
                    id.squeeze(0), 
                    skip_special_tokens=True
                ) 
                # Ask the LLM for the class index
                class_index = self.ask(
                    decoded_sample,  
                    return_tensor=False
                )
                predictions.append(class_index)
                labels.append(target)
        f1, acc = f1_acc_metrics(labels, predictions)
        print('############ Test Results ############')
        print(f"F1 Score: {f1}, Accuracy: {acc}")
        print('######################################')
        self._store_results(f1, acc)
        
    def _store_results(self, f1, acc):
        """
        Store the results in a file.
        """
        # create a pandas df with the results
        results_df = pd.DataFrame({
            'f1': [f1],
            'accuracy': [acc]
        })
        # store the results in a csv file
        results_df.to_csv(f"{self.storing_path}/results.csv", index=False)
        