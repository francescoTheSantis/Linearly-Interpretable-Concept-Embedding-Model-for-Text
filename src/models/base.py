import torch.nn as nn
import torch
import torch.nn.functional as F
from src.models.encoders.base_encoder import BaseEncoder

class BaseModel(nn.Module):
    """"""
    def __init__(self, 
                 output_size,
                 task='classification',
                 activation='ReLU',
                 latent_size=64,
                 c_groups=None,
                 encoder: BaseEncoder=None,
                 use_embeddings=False,
                 supervision=None
                 ):
        super().__init__()
        
        self.output_size = output_size
        self.task = task
        self.latent_size = latent_size
        self.int_idxs = None
        self.test_interventions = False
        self.c_groups = c_groups
        self.global_step = 0
        self.noise = None
        self.use_embeddings = use_embeddings
        self.supervision = supervision
        self.has_ml_cls = False

        if not self.use_embeddings:
            self.encoder = encoder
        else:
            self.encoder = None
            # Delete encoder
            del encoder

        self.task_loss_form = nn.CrossEntropyLoss()

        self.concept_loss_form = None
        self.task_penalty = None

    def _get_int_idxs(self, c):
        if self.supervision in ['supervised', 'generative']:
            if self.training or self.test_interventions:
                # intervene on the concepts according to the int_prob
                int_idxs = self.get_intervened_concepts_predictions(
                    c,
                    groups=self.c_groups
                )
            else:
                int_idxs = torch.zeros_like(c)
        elif self.supervision == 'self-generative':
            int_idxs = torch.ones_like(c, dtype=torch.bool)
        int_idxs = int_idxs.bool()
        return int_idxs
    
    def encode(self, input):
        ids = input['ids']
        type = input['type']
        attention = input['attention']
        embs = input['embedding']
        c_true = input['c']
        gen_c = input['gen_c'] 
    
        # If noise is provided, create a convex combination of the input and noise
        if self.noise!=None:
            eps = torch.randn_like(x)
            x = eps * self.noise + x * (1-self.noise)
         
        # Pass the input through the encoder
        if not self.use_embeddings:
            x = self.encoder({
                'ids': ids,
                'type': type,
                'attention': attention,
            })
        else:
            x = embs

        if self.has_concepts:
            if self.supervision == 'supervised':
                c = c_true
            elif self.supervision in ['generative','self-generative']:
                c = gen_c
            else:
                raise ValueError(f"Unknown supervision type: {self.supervision}")

            int_idxs = self._get_int_idxs(c)
            return x, c, int_idxs
        else:
            return x, None, None

    def _task_loss(self, y_hat, y):
        """
        Compute the task loss based on the task type.
        """
        y = y.flatten().long()
        # Task loss
        # check whether there are -1 in the predictions
        if torch.any(y_hat == -1) or self.classifier in ['dt', 'xg']:
            # If there are -1 in the predictions, set the task loss to 0
            task_loss = torch.tensor(0.0, device=y_hat.device, requires_grad=True)
        else:
            task_loss = self.task_loss_form(y_hat.squeeze(), y) 
        task_loss = self.task_penalty * task_loss
        return task_loss
    
    def _concept_loss(self, c_hat, c):
        """
        Compute the concept loss based on the concept loss form.
        """
        # If the the modality is supervised or generative,
        # and there are no -1 in the concepts (which is associated to no concept annotation), 
        # compute the concept loss.
        if self.supervision in ['supervised', 'generative'] and not torch.any(c == -1):
            concept_loss = 0
            for i in range(c.shape[1]):
                concept_loss += self.concept_loss_form(c_hat[:,i], c[:,i])
            concept_loss /= c.shape[1]
        else:
            concept_loss = torch.tensor(0.0, device=c_hat.device, requires_grad=True)
        return concept_loss

    def concept_based_loss(self, y_hat, y, c_hat=None, c=None):
        task_loss = self._task_loss(y_hat, y)
        concept_loss = self._concept_loss(c_hat, c) 
        loss = task_loss + concept_loss
        return loss
        
    def get_intervened_concepts_predictions(self, labels, groups=None):
        '''
        Function to generate a mask for the intervention process.
        The mask is generated based on the probability of intervention 
        and the mismatch between predictions and labels.
        '''
        if groups is not None:
            n_groups = len(groups)
            # Generate a mask of shape Batch x n_groups
            random_mask = torch.rand((labels.shape[0],n_groups), 
                                     dtype=torch.float,
                                     device=labels.device)
            mask = (random_mask < self.int_prob)
            mask = mask.int()
            # Apply group-based intervention
            group_mask = torch.zeros_like(labels, dtype=torch.int, device=labels.device)
            for idx, (_, group) in enumerate(groups.items()):
                group_mask[:, group] = mask[:, idx].unsqueeze(1).expand(-1, len(group))
            mask = group_mask.int()
        else:
            # Generate a probability mask of the same shape
            random_mask = torch.rand_like(labels, dtype=torch.float)
            # Apply probability threshold only on mismatched elements
            mask = (random_mask < self.int_prob)
            mask = mask.int()

        return mask
    
    def filter_output_for_loss(self, y_output, c_output=None):
        """
        Filter the output of the model for loss computation.
        This method can be overridden in subclasses to customize the output filtering.
        """
        return y_output, c_output
    
    def filter_output_for_metrics(self, y_output, c_output=None):
        """
        Filter the output of the model for metrics computation.
        This method can be overridden in subclasses to customize the output filtering.
        """
        return y_output, c_output
