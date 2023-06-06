from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import tqdm.notebook as tq
import wandb
import torchmetrics
from torch.utils.data import DataLoader

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# Determine which device on import, and then use that elsewhere.
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

def plot_confusion_matrix(cm, class_names):
    cm = cm.astype(np.float32) / cm.sum(axis=1)[:, None]
    df_cm = pd.DataFrame(cm, class_names, class_names)
    ax = sn.heatmap(df_cm, annot=True, cmap='flare')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.show()
    
def count_classes(preds):
    pred_classes = preds.argmax(dim=1)
    n_classes = preds.shape[1]
    return [(pred_classes == c).sum().item() for c in range(n_classes)]

# Task 1

# def train_epoch(epoch, model, optimizer, criterion, loader, num_classes, device):
#     model.train()
#     loss_sum = 0
#     correct = 0
#     total = 0

#     for i, (inputs, lbls) in enumerate(loader):
#         inputs, lbls = inputs.to(device), lbls.to(device)

#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, lbls)
#         loss.backward()
#         optimizer.step()

#         loss_sum += loss.item()
#         _, preds = outputs.max(1)
#         correct += preds.eq(lbls).sum().item()
#         total += lbls.size(0)

#     metrics_dict = {
#         'Loss_train': loss_sum / len(loader),
#         'Accuracy_train': correct / total,
#     }

#     return metrics_dict

# def val_epoch(epoch, model, criterion, loader, num_classes, device):
#     model.eval()
#     loss_sum = 0
#     correct = 0
#     total = 0
#     predictions = []
#     targets = []
#     confusion_matrix = torch.zeros(num_classes, num_classes)

#     for inputs, lbls in loader:
#         inputs, lbls = inputs.to(device), lbls.to(device)

#         with torch.no_grad():
#             outputs = model(inputs)
#             loss = criterion(outputs, lbls)
        
#         loss_sum += loss.item()
#         _, preds = outputs.max(1)
#         correct += preds.eq(lbls).sum().item()
#         total += lbls.size(0)
#         predictions.append(preds.cpu().numpy())
#         targets.append(lbls.cpu().numpy())
        
#         # Update confusion matrix
#         flattened_preds = torch.from_numpy(preds.cpu().numpy().reshape(-1))
#         flattened_lbls = torch.from_numpy(lbls.cpu().numpy().reshape(-1))
#         indices = flattened_lbls * num_classes + flattened_preds
#         confusion_matrix += torch.bincount(indices, minlength=num_classes**2).view(num_classes, num_classes)

#     predictions = np.concatenate(predictions)
#     targets = np.concatenate(targets)
#     uar = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)

#     metrics_dict = {
#         'Loss_val': loss_sum / len(loader),
#         'Accuracy_val': correct / total,
#         'UAR_val': uar
#     }

#     return metrics_dict, confusion_matrix.numpy()



# Task 2

def train_epoch(epoch, model, optimizer, criterion, loader, num_classes, device):
    model.train()
    loss_sum = 0
    correct = 0
    total = 0

    for i, batch in enumerate(loader):
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    metrics_dict = {
        'Loss_train': loss_sum / len(loader),
        'Accuracy_train': correct / total,
    }

    return metrics_dict


def val_epoch(epoch, model, criterion, loader, num_classes, device):
    model.eval()
    loss_sum = 0
    correct = 0
    total = 0
    predictions = []
    targets = []
    confusion_matrix = torch.zeros(num_classes, num_classes)

    for batch in loader:
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        lbls = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(inputs, attention_mask)
            loss = criterion(outputs, lbls)
        
        loss_sum += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(lbls).sum().item()
        total += lbls.size(0)
        predictions.append(preds.cpu().numpy())
        targets.append(lbls.cpu().numpy())
        
        # Update confusion matrix
        flattened_preds = torch.from_numpy(preds.cpu().numpy().reshape(-1))
        flattened_lbls = torch.from_numpy(lbls.cpu().numpy().reshape(-1))
        indices = flattened_lbls * num_classes + flattened_preds
        confusion_matrix += torch.bincount(indices, minlength=num_classes**2).view(num_classes, num_classes)

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    uar = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)

    metrics_dict = {
        'Loss_val': loss_sum / len(loader),
        'Accuracy_val': correct / total,
        'UAR_val': uar
    }

    return metrics_dict, confusion_matrix.numpy()


def train_model(model, train_loader, val_loader, optimizer, criterion,
                class_names, n_epochs, project_name, ident_str=None):
                
    num_classes = len(class_names)
    model.to(device)
    
    if ident_str is None:
        ident_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{model.__class__.__name__}_{ident_str}"
    run = wandb.init(project=project_name, name=exp_name)

    try:
        for epoch in tq.tqdm(range(n_epochs), total=n_epochs, desc='Epochs'):
            train_metrics_dict = train_epoch(epoch, model, optimizer, criterion,
                                             train_loader, num_classes, device)
                    
            val_metrics_dict, cm = val_epoch(epoch, model, criterion, val_loader,
                                             num_classes, device)
            
            wandb.log({**train_metrics_dict, **val_metrics_dict})
            
    finally:
        run.finish()

    # Plot confusion matrix from results of last val epoch
    plot_confusion_matrix(cm, class_names)

    # Save the model weights to "saved_models/"
    torch.save(model.state_dict(), "saved_models/model.pth")
