import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from torch_concepts.semantic import ProductTNorm
from torch_concepts.nn import functional as CF
from src.models.base import BaseModel, LogicModel
from torch.nn import functional as F

class DeepConceptReasoner(BaseModel):
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
                 semantic = ProductTNorm(),
                 temperature = 100,
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

        self.n_roles = 3
        
        self.embedding_size = embedding_size
        self.latent_size = latent_size
        self.task_penalty = task_penalty
        self.c_names = list(c_names)
        self.int_prob = int_prob
        self.int_idxs = int_idxs
        self.has_concepts = True
        self.noise = noise
        self.semantic = semantic
        self.temperature = temperature
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
        self.concept_importance_predictor = nn.Sequential(
            nn.Linear(embedding_size, self.latent_size),
            getattr(nn, activation)(),
            nn.Linear(self.latent_size, output_size * self.n_roles),
            nn.Unflatten(-1, (output_size, self.n_roles)),
        )

        self.concept_loss_form = nn.BCELoss()
        self.task_loss_form = nn.BCELoss()

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
        c_weights = self.concept_importance_predictor(c_emb)
        # adding memory dimension
        c_weights = c_weights.unsqueeze(dim=1)
        # soft selecting concept relevance (last role) among concepts
        relevance = CF.soft_select(c_weights[:, :, :, :, -2:-1],
                                   self.temperature, -3)
        # softmax over positive/negative roles
        polarity = c_weights[:, :, :, :, :-1].softmax(-1)
        # batch_size x memory_size x n_concepts x n_tasks x n_roles
        c_weights = torch.cat([polarity, 1 - relevance], dim=-1)

        y_pred = CF.logic_rule_eval(c_weights, c_pred,
                                    semantic=self.semantic)
        # removing memory dimension
        y_pred = y_pred[:, :, 0]
        return y_pred, c_pred

    def loss(self, y_hat, y, c_hat=None, c=None):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        return loss