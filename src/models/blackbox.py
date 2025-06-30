import torch
import torch.nn as nn
from src.models.base import BaseModel

class BlackBox(BaseModel):
    def __init__(self,
                 output_size,
                 c_names=None,
                 y_names=None,
                 activation='ReLU',
                 latent_size = 128,
                 c_groups = None,
                 encoder=None,
                 supervision=None,
                 use_embeddings=False,
                 encoder_output_size=None,
                 lm_embedding_size=None
                 ):
        
        super().__init__(
                 output_size = output_size,
                 activation = activation,
                 latent_size = latent_size,
                 c_groups = c_groups,
                 encoder = encoder,
                 use_embeddings = use_embeddings,
                 supervision = supervision
                 )

        self.has_concepts = False
        self.encoder_output_size = encoder_output_size
        self.lm_embedding_size = lm_embedding_size 
        
        input_size = lm_embedding_size if use_embeddings else encoder_output_size
        self.predictor = nn.Sequential(
            nn.Linear(input_size, latent_size),
            getattr(nn, activation)(),
            nn.Linear(latent_size, output_size)
        )

    def forward(self, input):
        x, _, _ = self.encode(input)
        y_hat = self.predictor(x)
        return y_hat, None
    
    def loss(self, y_hat, y, c_hat=None, c=None):
        y = y.flatten().long()
        # cross entropy
        loss = self.task_loss_form(y_hat.squeeze(), y)
        return loss