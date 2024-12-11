import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_explain as te
from torch_explain.nn.concepts import ConceptReasoningLayer
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

class CEM(nn.Module):
     """
    Concept Embedding Module (CEM)
    
    This module maps input embeddings to a concept space using a fully connected 
    linear layer followed by an activation function and a concept embedding layer. 
    The concept embedding layer projects the features into a space where they 
    are associated with interpretable concepts.

    Attributes:
        linear (nn.Linear): A fully connected linear layer that transforms the input embeddings.
        activation (nn.LeakyReLU): A non-linear activation function applied after the linear layer.
        concept_embedding (te.nn.ConceptEmbedding): A concept embedding layer that maps 
            the output of the linear layer to a space defined by the number of concepts 
            and the size of the concept embeddings.
    """
     
     def __init__(self, embedding_size, n_concepts, embedding_size_cem, n_out_linear=400):
        """
        Initializes the CEM module.

        Args:
            embedding_size (int): The size of the input embeddings (e.g., from a BERT model).
            n_concepts (int): The number of interpretable concepts.
            embedding_size_cem (int): The size of the embedding for each concept.
            n_out_linear (int, optional): The output size of the linear layer. Defaults to 400.
        """
        super(CEM, self).__init__()
        self.linear = nn.Linear(embedding_size, n_out_linear)
        self.activation = nn.LeakyReLU()
        self.concept_embedding = te.nn.ConceptEmbedding(n_out_linear, n_concepts, embedding_size_cem)
    
     def forward(self, x):
        """
        Defines the forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embedding_size).

        Returns:
            torch.Tensor: Output tensor after processing through the linear, activation, 
                and concept embedding layers. The shape depends on the 
                `embedding_size_cem` and `n_concepts` of the `ConceptEmbedding` layer.
        """
        
        x = self.linear(x)
        x = self.activation(x)
        x = self.concept_embedding(x)
        return x
     
class CBM_DNN(nn.Module):
    """
    Concept Bottleneck Model with a Deep Neural Network (CBM_DNN).

    Attributes:
        concept_ff (nn.Sequential): The concept extraction module.
        prediction_ff (nn.Sequential): The prediction module based on extracted concepts.
    """
    def __init__(self, embedding_size, n_concepts, n_classes, mode='linear', expand_factor=2):
        """
        Initializes the CBM_DNN model.

        Args:
            embedding_size (int): Size of the input embedding.
            n_concepts (int): Number of concepts to be extracted.
            n_classes (int): Number of output classes.
            mode (str): The prediction mode, either 'linear' or 'mlp'.
            expand_factor (int): Expansion factor for hidden layer size in 'mlp' mode.
        """
        super(CBM_DNN, self).__init__()
        
        # Concept extraction layer
        self.concept_ff = nn.Sequential(
            nn.Linear(embedding_size, n_concepts),
            nn.ReLU(),
            nn.Linear(n_concepts, n_concepts),
            nn.Sigmoid()
        )
        
        # Prediction layer
        if mode == 'linear':
            self.prediction_ff = nn.Sequential(
                nn.Linear(n_concepts, n_classes),
                nn.Softmax(dim=-1)
            )
        elif mode == 'mlp':
            self.prediction_ff = nn.Sequential(
                nn.Linear(n_concepts, int(expand_factor * n_concepts)),
                nn.ReLU(),
                nn.Linear(int(expand_factor * n_concepts), n_classes),
                nn.Softmax(dim=-1)
            )
        else:
            raise ValueError(f"Mode {mode} not supported!")
    
    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embedding_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_classes).
        """
        concepts = self.concept_ff(x)
        predictions = self.prediction_ff(concepts)
        return predictions


class CBM_ML(nn.Module):
    def __init__(self, embedding_size, n_concepts, n_classes, ml_model="tree", **ml_params):
        """
        CBM_ML class combining a DNN for concept prediction and a tree/XGBoost model for classification.

        Args:
        - embedding_size (int): Size of the input features to the DNN.
        - n_concepts (int): Number of concepts output by the DNN.
        - n_classes (int): Number of prediction classes.
        - ml_model (str): 'tree' or 'xgboost'.
        - ml_params (dict): Parameters for the ML model.
        """
        super(CBM_ML, self).__init__()

        # Define DNN for concept prediction
        self.concept_ff = nn.Sequential(
            nn.Linear(embedding_size, n_concepts),
            nn.ReLU(),
            nn.Linear(n_concepts, n_concepts),
            nn.Sigmoid()
        )

        # Store ML model type and parameters
        self.ml_model_type = ml_model
        self.n_concepts = n_concepts
        self.n_classes = n_classes

        # Initialize ML model (decision tree or XGBoost)
        if ml_model == "tree":
            self.ml_model = DecisionTreeClassifier(**ml_params)
        elif ml_model == "xgboost":
            self.ml_model = XGBClassifier(**ml_params)
        else:
            raise ValueError(f"Unsupported ML model: {ml_model}")

    def forward(self, x):
        """
        Forward pass through the DNN.

        Args:
        - x (torch.Tensor): Input tensor of shape (n_samples, embedding_size).
        Returns:
        - torch.Tensor: Predicted concepts of shape (n_samples, n_concepts).
        """
        return self.concept_ff(x)

    def fit_ml_model(self, x, y):
        """
        Train the ML model using the concepts predicted by the DNN.

        Args:
        - x (torch.Tensor): Input tensor of shape (n_samples, embedding_size).
        - y (torch.Tensor): Tensor of shape (n_samples,), integer labels for predictions.
        """
        with torch.no_grad():
            concepts = self.forward(x).cpu().numpy()  # Predict concepts
        y_np = y.cpu().numpy()
        self.ml_model.fit(concepts, y_np)

    def predict(self, x):
        """
        Predict classes using the trained ML model.

        Args:
        - x (torch.Tensor): Input tensor of shape (n_samples, embedding_size).
        Returns:
        - np.ndarray: Predicted labels as a NumPy array.
        """
        with torch.no_grad():
            concepts = self.forward(x).cpu().numpy()  # Predict concepts
        return self.ml_model.predict(concepts)
