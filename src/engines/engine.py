from typing import Optional
from torch import nn
import torch
import pytorch_lightning as pl
from src.metrics import Task_Accuracy, Concept_Accuracy
from torchmetrics.classification import F1Score
from collections import OrderedDict
import pandas as pd
import torch.nn.functional as F
from src.models.base import BaseModel

class Engine(pl.LightningModule):
    """"""
    def __init__(self,
                model: Optional[BaseModel] = None,
                c_names: Optional[list] = None,
                y_name: Optional[str] = None,
                csv_log_dir: Optional[str] = None,
                concept_annotations: Optional[bool] = True,
                supervision: Optional[str] = 'supervised'
                ):
        super(Engine, self).__init__()         
        self.model = model
        self.save_hyperparameters(ignore=["model"], logger=False)
        self.concept_annotations = concept_annotations
        self.supervision = supervision

        self.c_names = c_names
        self.y_name = y_name
        self.num_classes = len(y_name) if len(y_name)>1 else 2
        self.class_names = y_name if len(y_name)>1 else ['0','1']

        self.task_metric = Task_Accuracy()
        #self.concept_metric = F1Score(task="multiclass", 
        #                              num_classes=self.num_classes, 
        #                              average="macro")
        self.concept_metric = Concept_Accuracy()

        self.csv_log_dir = csv_log_dir

        # If we are using the LinearConceptEmbeddingModel model,
        # we need to save the tensors required for the explanations.
        if self.model.__class__.__name__ == 'LinearConceptEmbeddingModel':
            self.pred_weights = []
            self.c_trues = []
            self.c_preds = []
            self.y_trues = []
            self.y_preds = []
            self.ids = []

    def forward(self, input):
        return self.model(input)

    def predict(self, input):
        return self.model(input)

    def unpack_batch(self, batch):
        ids = batch[0]
        type = batch[1]
        attention = batch[2]
        embedding = batch[3]
        c = batch[4]
        y = batch[5]
        gen_c = batch[6] 
        return ids, type, attention, embedding, c, y, gen_c

    def shared_step(self, batch):
        ids, type, attention, embedding, c, y, gen_c = self.unpack_batch(batch)

        inputs = {
            'ids': ids,
            'type': type,
            'attention': attention,
            'embedding': embedding,
            'c': c,
            'y': y,
            'gen_c': gen_c
        }

        # model forward
        model_output = self.forward(inputs)

        # Compute loss
        y_output, c_output = self.model.filter_output_for_loss(*model_output)

        # If the supervision is 'self-generative' or 'generative', the loss is copmuted with respcet to the 
        # generated concept labels.
        if self.supervision == 'self-generative' or self.supervision == 'generative':
            c_loss = gen_c
        else:
            c_loss = c

        loss = self.model.loss(y_output, y, c_output, c_loss)
    
        return loss, model_output, y, c, ids

    def training_step(self, batch, batch_idx):
        self.model.global_step = self.global_step
        loss, model_output, y, c, ids = self.shared_step(batch)
        self.log("train_loss", loss)
        y_output, c_output = self.model.filter_output_for_metrics(*model_output)
        task_acc = self.task_metric(y_output, y)
        self.log('train_task_acc', task_acc)
        # If the model has concepts and concept annotations are available, 
        # compute the concept accuracy.
        if self.model.has_concepts and not torch.any(c == -1):
            concept_acc = self.concept_metric(c_output, c)
            self.log('train_concept_acc', concept_acc)

        #if ('dt' in self.model.__class__.__name__ or 'xg' in self.model.__class__.__name__) and self.supervision == 'self-generative':
        #    return y_output.mean() * -1
        #else:
        return loss

    def validation_step(self, batch, batch_idx):
        loss, model_output, y, c, ids = self.shared_step(batch)
        self.log("val_loss", loss)
        y_output, c_output = self.model.filter_output_for_metrics(*model_output)
        task_acc = self.task_metric(y_output, y)
        self.log('val_task_acc', task_acc)
        # If the model has concepts and concept annotations are available, 
        # compute the concept accuracy.
        if self.model.has_concepts and not torch.any(c == -1):
            concept_acc = self.concept_metric(c_output, c)
            self.log('val_concept_acc', concept_acc)
        return loss 
    
    def test_step(self, batch, batch_idx):
        loss, model_output, y, c, ids = self.shared_step(batch)
        y_output, c_output = self.model.filter_output_for_metrics(*model_output)
        self.log("test_loss", loss)
        task_acc = self.task_metric(y_output, y)
        self.log('test_task_acc', task_acc)
        # If the model has concepts and concept annotations are available, 
        # compute the concept accuracy.
        if self.model.has_concepts and not torch.any(c == -1):
            concept_acc = self.concept_metric(c_output, c)
            self.log('test_concept_acc', concept_acc)

        # If the name of the class is LinearConceptEmbeddingModel,
        # update the tensors required for the explanations.
        if self.model.__class__.__name__ == 'LinearConceptEmbeddingModel':
            self.pred_weights.append(model_output[2])
            self.c_trues.append(c)
            self.c_preds.append(c_output)
            self.y_trues.append(y)
            self.y_preds.append(y_output)
            self.ids.append(ids)
        return loss 
    
    def on_test_epoch_end(self):
        '''
        If the name of the class is LinearConceptEmbeddingModel, store the tensors required for the explanations.
        '''
        if self.model.__class__.__name__ == 'LinearConceptEmbeddingModel':
            # Concatenate the tensors
            self.pred_weights = torch.cat(self.pred_weights, dim=0)
            self.c_trues = torch.cat(self.c_trues, dim=0)
            self.c_preds = torch.cat(self.c_preds, dim=0)
            self.y_trues = torch.cat(self.y_trues, dim=0)
            self.y_preds = torch.cat(self.y_preds, dim=0).argmax(-1)
            self.ids = torch.cat(self.ids, dim=0)

            # Convert the tensors to pandas dfs
            c_preds = pd.DataFrame(self.c_preds.cpu().numpy(), columns=self.c_names)
            c_trues = pd.DataFrame(self.c_trues.cpu().numpy(), columns=self.c_names)

            # Create a list of names for the y_preds and y_trues to create 
            # a pandas containing the list of predicted and true labels
            y_preds = pd.DataFrame(F.one_hot(self.y_preds.cpu(), self.num_classes).squeeze().numpy(),
                                   columns=self.class_names)
            y_trues = pd.DataFrame(F.one_hot(self.y_trues.long().cpu(), self.num_classes).squeeze().numpy(),
                                   columns=self.class_names)

            # Store the pandas dfs
            c_preds.to_csv(f"{self.csv_log_dir}/c_preds.csv", index=False)
            c_trues.to_csv(f"{self.csv_log_dir}/c_trues.csv", index=False)
            y_preds.to_csv(f"{self.csv_log_dir}/y_preds.csv", index=False)
            y_trues.to_csv(f"{self.csv_log_dir}/y_trues.csv", index=False)

            # Save the predicted_weights to a .pt file
            torch.save(self.pred_weights, f"{self.csv_log_dir}/pred_weights.pt")
            torch.save(self.ids, f"{self.csv_log_dir}/ids.pt")

    def configure_optimizers(self):
        if ('dt' in self.model.__class__.__name__ or 'xg' in self.model.__class__.__name__) and self.supervision == 'self-generative':
            # If the model is a decision tree or XGBoost, 
            # the supervision is 'self-generative',
            # then we do not need an optimizer.
            return None
        return [self.optimizer], [self.scheduler]
 