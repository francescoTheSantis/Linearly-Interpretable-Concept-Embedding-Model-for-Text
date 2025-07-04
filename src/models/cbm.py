import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from src.models.base import BaseModel
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

import numpy as np
import torch.nn.functional as F

class ConceptBottleneckModel(BaseModel):
    def __init__(self, 
                 output_size,
                 c_names,
                 y_names,
                 task_penalty,
                 classifier=None,
                 activation='ReLU',
                 int_prob=0.1,
                 int_idxs=None,
                 noise=None,
                 latent_size = 128,
                 c_groups=None,
                 weight_reg=1e-6,
                 encoder=None,
                 supervision='supervised',
                 use_embeddings=False,
                 encoder_output_size=None,
                 lm_embedding_size=None,
                 ml_params=None
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

        self.classifier = classifier
        self.task_penalty = task_penalty
        self.c_names = list(c_names)
        self.int_prob = int_prob
        self.int_idxs = int_idxs
        self.has_concepts = True
        self.noise = noise
        self.encoder_output_size = encoder_output_size
        self.lm_embedding_size = lm_embedding_size
        self.latent_size = latent_size 
        self.ml_params = ml_params 
        self.output_size = output_size
        self.weight_reg = weight_reg

        input_size = lm_embedding_size if use_embeddings else encoder_output_size
        self.first_layer = nn.Sequential(
            nn.Linear(input_size, latent_size),
            getattr(nn, activation)(),
        )

        self.bottleneck = pyc_nn.LinearConceptBottleneck(
            latent_size,
            self.c_names,
        )

        if self.classifier=='linear':
            self.y_predictor = nn.Linear(
                len(c_names), 
                output_size
            )
        elif self.classifier=='mlp':
            self.y_predictor = nn.Sequential(
                nn.Linear(len(c_names), 2 * len(c_names)),
                getattr(nn, activation)(),
                nn.Linear(2 * len(c_names), output_size),
            )
        elif self.classifier=='dt':
            self.y_predictor = DecisionTreeClassifier(**self.ml_params) if self.ml_params is not None else DecisionTreeClassifier()
        elif self.classifier=='xg':
            self.y_predictor = xgb.XGBClassifier(**self.ml_params) if self.ml_params is not None \
                else xgb.XGBClassifier(objective='multi:softprob', num_class=self.output_size, eval_metric='mlogloss', nthread=1)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier}")

        self.concept_loss_form = nn.BCELoss()

        if self.classifier in ['dt', 'xg']:
            self.ml_fit = False
            self.has_ml_cls = True
        else:
            self.ml_fit = None

    def _concept_prediction(self, input):
        x, c_true, int_idxs = self.encode(input)
        bsz = x.shape[0]

        x = self.first_layer(x)

        c_pred, _ = self.bottleneck(
            x,
            c_true=c_true,
            intervention_idxs=int_idxs,
            intervention_rate=1.,
        )
        
        if self.classifier in ['linear', 'mlp']:
            y_pred = self.y_predictor(c_pred)
        else:
            y_pred = None
        
        return y_pred, c_pred

    def forward(self, input):
        y_pred, c_pred = self._concept_prediction(input)
        if self.classifier in ['dt', 'xg']:
            y_pred = self.ml_predict(c_pred)
        return y_pred, c_pred

    def loss(self, y_hat, y, c_hat=None, c=None):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        # adding l1 regularization to the weights if neural classifier
        if self.classifier in ['linear', 'mlp']:
            w_loss = self.weight_reg * self.y_predictor.weight.flatten().norm(p=1)
            loss += w_loss
        return loss
    
    def unpack_batch(self, batch):
        ids = batch[0]
        type = batch[1]
        attention = batch[2]
        embedding = batch[3]
        c = batch[4]
        y = batch[5]
        gen_c = batch[6] 
        return ids, type, attention, embedding, c, y, gen_c

    def fit_ml_model(self, loaded_train):
        """
        Train decision tree / XGBoost to predict the task variable given the concept predictions
        """
        # Get the model's concept predictions by:
        # 1. Concatenating the concept predictions sin the loaded train dataset,
        # 2. get the concept predictions by using _concept_prediction.

        concat_c_preds = []
        concat_y_true = []
        with torch.no_grad():
            for batch in loaded_train:
                ids, type, attention, embedding, c, y, gen_c = self.unpack_batch(batch)
                inputs = {
                    'ids': ids,
                    'type': type,
                    'attention': attention,
                    'embedding': embedding,
                    'c': c,
                    'y': y, # y_true
                    'gen_c': gen_c
                }
                model_output = self.forward(inputs)
                c_preds = model_output[1]
                concat_c_preds.append(c_preds.cpu().numpy())
                concat_y_true.append(y.cpu().numpy())
        # Concatenate the concept predictions and true labels
        c_preds = np.concatenate(concat_c_preds, axis=0)
        y_true = np.concatenate(concat_y_true, axis=0)
        if self.y_predictor.__class__.__name__ == 'DecisionTreeClassifier':
            self.y_predictor.fit(c_preds, y_true)
        elif self.y_predictor.__class__.__name__ == 'XGBClassifier':
            self.y_predictor.fit(c_preds, y_true)
        self.ml_fit = True

    def ml_predict(self, c_preds):
        """
        Predict using the trained decision tree/ XGboost model.
        """
        device = c_preds.device
        if self.ml_fit:
            c_preds = c_preds.cpu().numpy()
            # hard concepts to facilitate the decision tree/ XGboost model
            hard_c_preds = np.where(c_preds > 0.5, 1, 0)
            # Predict using the decision tree/ XGboost model
            if isinstance(self.y_predictor, DecisionTreeClassifier):
                y_preds = self.y_predictor.predict(hard_c_preds)
            elif isinstance(self.y_predictor, xgb.XGBClassifier):
                y_preds = self.y_predictor.predict(hard_c_preds)
            # Transform to torch tensor
            return torch.tensor(y_preds, device=device, dtype=torch.float32)
        else:
            # If the model is not fitted, return a dummy output of shape (batch_size, output_size).
            # It is multiplied by -1 to indicate no prediction.
            return torch.ones(c_preds.shape[0], self.output_size, device=device, requires_grad=True) * -1
    



