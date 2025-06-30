import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn

from src.models.base import BaseModel

class ConceptEmbeddingModel(BaseModel):
    def __init__(self, 
                 output_size,
                 c_names,
                 y_names,
                 task_penalty,
                 activation='ReLU',
                 int_prob=0.1,
                 int_idxs=None,
                 noise=None,
                 embedding_size = 16,
                 latent_size = 128,
                 c_groups=None,
                 encoder=None,
                 supervision='supervised',
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

        self.embedding_size = embedding_size
        self.task_penalty = task_penalty
        self.c_names = list(c_names)
        self.int_prob = int_prob
        self.int_idxs = int_idxs
        self.has_concepts = True
        self.noise = noise
        self.encoder_output_size = encoder_output_size
        self.lm_embedding_size = lm_embedding_size 

        input_size = lm_embedding_size if use_embeddings else encoder_output_size
        self.first_layer = nn.Sequential(
            nn.Linear(input_size, latent_size),
            getattr(nn, activation)(),
        )

        self.bottleneck = pyc_nn.ConceptEmbeddingBottleneck(
            latent_size,
            self.c_names,
            embedding_size,
        )

        self.y_predictor = nn.Sequential(
            nn.Linear(len(self.c_names) * embedding_size, latent_size),
            nn.LeakyReLU(),
            nn.Linear(latent_size, output_size),
        )

        self.concept_loss_form = nn.BCELoss()

    def forward(self, input):
        x, c_true, int_idxs = self.encode(input)
        
        x = self.first_layer(x)

        c_emb, c_dict = self.bottleneck(
            x,
            c_true=c_true,
            intervention_idxs=int_idxs,
            intervention_rate=1.,
        )
        c_pred = c_dict['c_int']

        y_pred = self.y_predictor(c_emb.flatten(-2))
        return y_pred, c_pred

    def loss(self, y_hat, y, c_hat=None, c=None):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        return loss


