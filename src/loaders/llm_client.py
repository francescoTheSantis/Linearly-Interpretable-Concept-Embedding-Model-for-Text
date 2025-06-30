from mistralai import Mistral
import openai
from env import OPENAI_API_KEY, MISTRALAIKEY
import torch
import logging

# Set the logging level for httpx to WARNING to suppress INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)

class llm_client:
    def __init__(self, LLM='gpt-4o', temperature=0, max_tries=1):
        
        self.LLM = LLM
        self.temperature = temperature
        self.max_tries = max_tries
        self.n_concepts = None
        
        if 'gpt' in self.LLM:     
            openai.api_key = OPENAI_API_KEY
        elif 'mistral' in self.LLM or 'mixtral' in self.LLM or 'ministral' in self.LLM:
            api_key = MISTRALAIKEY
            self.client = Mistral(api_key=api_key)
        else:                     
            raise ValueError(f"model ({self.LLM}) not supported.")
                                  
    def _invoke(self, question, temperature=0):
        if 'gpt' in self.LLM:
            response = openai.chat.completions.create(
                model=self.LLM,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": question},
                ],
                temperature=temperature
            )
            answer = response.choices[0].message.content
        elif 'mistral' in self.LLM or 'mixtral' in self.LLM or 'ministral' in self.LLM:
            chat_response = self.client.chat.complete(
                model = self.LLM,
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": question},
                ],
                temperature=temperature
            )
            answer = chat_response.choices[0].message.content
        else:
            raise ValueError("model still not implemented")
        return answer
    
    '''
    def parse_answer(self, answer):
        start_tag = "<answer>"
        end_tag = "</answer>"
        start_index = answer.find(start_tag) + len(start_tag)
        end_index = answer.find(end_tag)
        if start_index != -1 and end_index != -1:
            answer = answer[start_index:end_index].strip()
        return answer
    '''

    def _parse_answer(self, answer, return_tensor=False):
        # The answe will be structured in the following way:
        # concept_1: value_1, concept_2: value_2, ...
        # where value_i is either 0 or 1.
        # We will parse the answer to extract the concepts and their values.
        concepts = {}
        parts = answer.split(',')
        for part in parts:
            part = part.strip()
            if ':' in part:
                concept, value = part.split(':', 1)
                concept = concept.strip()
                value = value.strip()
                if int(value) in [0, 1]:
                    concepts[concept] = int(value)
        
        if return_tensor:
            # Convert the concepts dictionary to a tensor
            concept_names = list(concepts.keys())
            concept_values = list(concepts.values())
            concepts_tensor = torch.tensor(concept_values, dtype=torch.float32)
            return concepts_tensor
        return concepts
    
    def _default_answer(self):
        # Return a default answer in case of failure.
        if self.n_concepts is None:
            raise ValueError("Number of concepts not set. Please set n_concepts before calling _default_answer.")
        default_answer = {f"concept_{i+1}": 0 for i in range(self.n_concepts)}
        default_answer = ', '.join([f"{k}: {v}" for k, v in default_answer.items()])
        return default_answer

    def ask(self, question, istruction_prompt, return_tensor=False):
        """
        Ask a question to the LLM and return the parsed answer.
        """

        # Compose the istruction prompt and the question.
        # the istruction prompt has a specific location for the question.
        question = istruction_prompt.replace("<review>", question)

        try:
            answer = self._invoke(question, temperature=self.temperature)
            # answer = self.parse_answer(answer)
            answer = self._parse_answer(answer, return_tensor)
            return answer
        except Exception as e:
            print(f"Error: {e}. Returning default answer.")
            answer = self._default_answer() 
            answer = self._parse_answer(answer, return_tensor)
            return answer
            