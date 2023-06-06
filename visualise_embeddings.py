import numpy as np
import random
import sklearn.decomposition
import sklearn.manifold
import sklearn.cluster
import torch
import torch.nn as nn
import tqdm.notebook as tq
import sys
from transformers import AutoModelForSequenceClassification

import matplotlib.pyplot as plt

import datasets
import models

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

def as_masks(arr):
    '''
    Used for Task 2b Challenge.
    Takes an array of integer class predictions,
    and returns a list of boolean masks over `arr`
    such that:

        masks = as_masks(arr)
        arr[masks[n]]

    will select all instances of predictions for the `n`th class.
    This can then be used to select from a parallel array with
    different information.
    e.g.

        arr = np.array([0, 0, 1, 1, 3, 1, 0])
        masks = as_masks(arr)
        # (0 for False, 1 for True below)
        # masks[0] =   [1, 1, 0, 0, 0, 0, 1]
        # masks[1] =   [0, 0, 1, 1, 0, 1, 0]
        # masks[2] =   [0, 0, 0, 0, 0, 0, 0]
        # masks[3] =   [0, 0, 0, 0, 1, 0, 0]
        embeds = ... # some array shaped [7, 128]
        # embeds[masks[0]] will select all embeds that were for class 0
    '''
    n_classes = arr.max()+1
    one_hot = np.eye(n_classes)[arr]
    return [m == 1 for m in one_hot.T]

    
# def collect_outputs(dl, model, sequenceClassificationModel=True):
#     collected_outputs = []
#     collected_labels = []
#     desc = 'Passing data through model'
#     with torch.no_grad():
#         for batch in tq.tqdm(dl, total=len(dl), desc=desc):
#             texts = batch['input_ids'].to(device)
#             labels = batch['labels'].to(device)
#             out = model(texts)
#             if sequenceClassificationModel == False:
#                 collected_outputs.append(torch.mean(out['logits'][:, 1:], dim=1).cpu().numpy())
#             else:
#                 collected_outputs.append(out['logits'].cpu().numpy())
#             collected_labels.append(labels.cpu().numpy())

#     collected_outputs_np = np.concatenate(collected_outputs)
#     collected_labels_np = np.concatenate(collected_labels)

#     masks = as_masks(collected_labels_np)

#     return collected_outputs_np, masks

def collect_outputs(dl, model, sequenceClassificationModel=True):
    collected_outputs = []
    collected_labels = []
    desc = 'Passing data through model'
    with torch.no_grad():
        for batch in tq.tqdm(dl, total=len(dl), desc=desc):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)  # Get attention_mask from batch
            labels = batch['labels'].to(device)
            out = model(input_ids, attention_mask)  # Pass both input_ids and attention_mask to the model
            if sequenceClassificationModel == False:
                collected_outputs.append(torch.mean(out[:, 1:], dim=1).cpu().numpy())
            else:
                collected_outputs.append(out.cpu().numpy())
            collected_labels.append(labels.cpu().numpy())

    collected_outputs_np = np.concatenate(collected_outputs)
    collected_labels_np = np.concatenate(collected_labels)

    masks = as_masks(collected_labels_np)

    return collected_outputs_np, masks



def fit_kmeans(embeddings, n_classes=4):
    '''
    Fits kmeans to `embeddings`, producing `n_classes` clusters
    then matches each embedding to it's nearest cluster, and returns
    masks for each cluster of the embeddings.
    '''
    kmeans = sklearn.cluster.KMeans(n_classes)
    class_names = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
    pred = kmeans.fit_predict(embeddings)
    masks = as_masks(pred)
    return class_names, masks

def make_plottable(embeddings):
    '''
    Reduces the dimensionality of embeddings to 2 dimensions so that it can be
    plotted.
    Please be aware that this is a very lossy operation;
    the purpose of TSNE is to reduce the dimensionality of the embedding
    to 2D for visualization. TSNE is the most popular technique for dimensionality
    reduction for visualizing high dimensional data in 2D.
    '''
    # TODO Task 2b - Create an instance TSNE using the following
    #                sklearn library class:
    #   - sklearn.manifold.TSNE
    # Fill in the following to create the tsne objects: 
    #     tsne = sklearn....
    # Then use the above created object to transform the given embeddings 
    # to the reduced dimensions by first fitting the embeddings using by tsne.
    #     plottable = tsne.fit_transform(...)
    
    # Instantiate TSNE object
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=42)
    # Perform dimensionality reduction
    plottable = tsne.fit_transform(embeddings)
    return plottable

def plot_classifications(class_names, masks, arr, title):
    '''
    For each point in arr, plot it in a scatter plot,
    colouring it by the class indicated in `masks`.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i, mask in enumerate(masks):
        nm = class_names[i]
        ax.scatter(arr[mask][:, 0], arr[mask][:, 1], label=nm, alpha=0.2)
    leg = ax.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    ax.set_title(title)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.show()

# def mk_plots(sentence_len, model_fname=None, sequenceClassificationModel = True):
#     # Set random seed to ensure consistent results
#     torch.manual_seed(42)
#     random.seed(42)
#     np.random.seed(42)

    
#     # Prepare data
#     ds = datasets.TextDataset(fname='/content/drive/MyDrive/TranTasks/data/txt/val.csv', sentence_len=sentence_len)
#     dl = torch.utils.data.DataLoader(ds, batch_size=32)

#     # Instantiate models
#     if sequenceClassificationModel:
#         model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
#     else:
#         model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')

#     model.eval()

#     if model_fname is not None:
#         model.load_state_dict(torch.load(model_fname))
#         model.to(device)
#         preds, _ = collect_outputs(dl, model, sequenceClassificationModel)
#         pred_masks = as_masks(preds)
#         model.classifier = nn.Identity()
#     else:
#         model.to(device)
#         model.classifier = nn.Identity()

#     print("Collecting embeddings...")
#     embeds, label_masks = collect_outputs(dl, model)
#     print("Reducing dimensionality of embeddings...")
#     plottable = make_plottable(embeds)

#     with open('/content/drive/MyDrive/TranTasks/data/txt/classes.txt') as f:
#         class_names = [line.rstrip('\n') for line in f]

#     plot_classifications(class_names, label_masks, plottable, 'True Labels')
#     if model_fname is None:
#         print("Fitting kmeans...")
#         kmeans_names, kmeans_masks = fit_kmeans(embeds)
#         plot_classifications(
#             kmeans_names, kmeans_masks, plottable, 'Clustered Labels')
#     else:
#         plot_classifications(class_names, pred_masks, plottable, 'Predicted Labels')

def mk_plots(sentence_len, model_fname=None, sequenceClassificationModel = True):
    # Set random seed to ensure consistent results
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Prepare data
    ds = datasets.TextDataset(fname='/content/drive/MyDrive/TranTasks/data/txt/val.csv', sentence_len=sentence_len)
    dl = torch.utils.data.DataLoader(ds, batch_size=32)

    # Instantiate your custom model
    model = models.DistilBertForClassification(n_classes=4)

    model.eval()

    if model_fname is not None:
        # Load the state dict from the saved model file.
        model.load_state_dict(torch.load('/content/drive/MyDrive/TranTasks/saved_models/model.pth')) 
        model.to(device)
        preds, _ = collect_outputs(dl, model, sequenceClassificationModel)
        pred_masks = as_masks(preds)
    else:
        model.to(device)

    print("Collecting embeddings...")
    embeds, label_masks = collect_outputs(dl, model)
    print("Reducing dimensionality of embeddings...")
    plottable = make_plottable(embeds)

    with open('/content/drive/MyDrive/TranTasks/data/txt/classes.txt') as f:
        class_names = [line.rstrip('\n') for line in f]

    plot_classifications(class_names, label_masks, plottable, 'True Labels')
    if model_fname is None:
        print("Fitting kmeans...")
        kmeans_names, kmeans_masks = fit_kmeans(embeds)
        plot_classifications(
            kmeans_names, kmeans_masks, plottable, 'Clustered Labels')
    else:
        plot_classifications(class_names, pred_masks, plottable, 'Predicted Labels')
