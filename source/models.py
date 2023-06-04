import numpy as np
import pandas as pd
import torch
from torch import nn
# Install required packages.
import os
import torch
#os.environ['TORCH'] = torch.__version__
#print(torch.__version__)

#!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
#!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
#!pip install -q git+https://github.com/pyg-team/pytorch_geometric.git

# Helper function for visualization.
from torch_geometric.nn import GCNConv, SAGEConv
from torch.nn import Linear
import torch.nn.functional as F

def train_step(model, optimizer, criterion, data):
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index)  # Perform a single forward pass.
        train_loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        train_loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        train_correct = pred[data.train_mask] == data.y[data.train_mask]  # Check against ground-truth labels.
        train_acc = int(train_correct.sum()) / int(data.train_mask.sum())
        return train_loss, train_acc

def valid_step(model, criterion, data):
        model.eval()
        out = model(data.x, data.edge_index)
        valid_loss = criterion(out[data.valid_mask], data.y[data.valid_mask])  # Compute the loss solely based on the validation nodes.
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        valid_correct = pred[data.valid_mask] == data.y[data.valid_mask]  # Check against ground-truth labels.
        valid_acc = int(valid_correct.sum()) / int(data.valid_mask.sum())  # Derive ratio of correct predictions.
        return valid_loss, valid_acc

def test(model, data):
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
        return test_acc

def train_model(model, data, epochs, es_patience=10, es_threshold=0.05):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(es_patience, es_threshold)

    epochs_list=[]
    train_losses=[]
    train_accuracies=[]
    valid_losses=[]
    valid_accuracies=[]


    for epoch in range(epochs):

        train_loss, train_accuracy = train_step(model, optimizer, criterion, data)
        valid_loss, valid_accuracy = valid_step(model, criterion, data)

        epochs_list.append(epoch)
        train_losses.append(train_loss.item())
        train_accuracies.append(train_accuracy)
        valid_losses.append(valid_loss.item())
        valid_accuracies.append(valid_accuracy)
        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Valid Acc: {valid_accuracy:.4f}')

        if (early_stopper.step(valid_loss)==0):
             continue
        elif(early_stopper.step(valid_loss)==1):
            pass
        else:
            print(f"Early stopping at epoch {epoch}")
            break
    
    zipped = list(zip(epochs_list, train_losses, train_accuracies, valid_losses, valid_accuracies))
    metrics = pd.DataFrame(zipped, columns=['epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc'])
   
    return metrics



class EarlyStopper:
    def __init__(self, patience: int, threshold: float) -> None:
        self.patience = patience
        self.threshold = threshold

        self.min_loss = np.inf
        self.counter = 0

    def step(self, loss: float) -> bool:
        if loss < self.min_loss * (1 - self.threshold):
            self.min_loss = loss
            self.counter = 0
            flag = 0
        else:
            self.counter += 1
            if self.counter <= self.patience:
                flag = 1
            else:
                flag = 2
        return flag

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(patience={self.patience}, threshold={self.threshold})"
        
class GCN(torch.nn.Module):
    def __init__(self, data, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(data.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, data.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, data, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = SAGEConv(data.num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, data.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GraphSAGE2(torch.nn.Module):
    def __init__(self, data, hidden_channels1, hidden_channels2):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = SAGEConv(data.num_features, hidden_channels1)
        self.conv2 = SAGEConv(hidden_channels1, hidden_channels2)
        self.conv3 = SAGEConv(hidden_channels2, data.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return x
