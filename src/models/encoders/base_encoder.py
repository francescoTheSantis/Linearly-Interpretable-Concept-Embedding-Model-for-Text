import torch
from torch import nn
from transformers import AutoModel

class BaseEncoder(nn.Module):
    def __init__(self, model_name: str, fine_tune: bool = False):
        super(BaseEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        
        self._freeze_parameters()

        if fine_tune:
            self._unfreeze_last_layer()

    def _freeze_parameters(self):
        """
        Freeze all parameters by default
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def _unfreeze_last_layer(self):
        """
        Unfreeze the last layer of the model to allow training.
        """
        for name, param in self.model.named_parameters():
            if "layer." in name and name.endswith(".weight") or name.endswith(".bias"):
                param.requires_grad = True

    def forward(self, tokenized_input: dict) -> torch.Tensor:
        """
        Extract the [CLS] token embedding from the output of the HuggingFace model.

        Args:
            tokenized_input (dict): The dictionary output from the tokenizer containing
                                    input_ids, attention_mask, etc.

        Returns:
            torch.Tensor: The embedding of the [CLS] token.
        """
        outputs = self.model(**tokenized_input)
        # Extract the last hidden state
        last_hidden_state = outputs.last_hidden_state
        # The [CLS] token is typically the first token in the sequence
        cls_embedding = last_hidden_state[:, 0, :]
        return cls_embedding