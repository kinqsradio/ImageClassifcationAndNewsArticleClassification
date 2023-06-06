import torch
import torch.nn as nn
import torchvision
from transformers import AutoModelForSequenceClassification, DistilBertTokenizerFast
import torch.nn.functional as F

# TODO Task 1c - Implement a SimpleBNConv
class SimpleBNConv(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = self._make_layers()
        self.fc = nn.Linear(in_features=128 * 5 * 5, out_features=num_classes)
        self.resize = nn.AdaptiveAvgPool2d((5, 5))
    
    def _make_layers(self):
        layers = []
        in_channels = 3
        for out_channels in [8, 16, 32, 64, 128]:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
                nn.MaxPool2d(2),
            ])
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            print(f"After layer {i}, shape: {x.shape}")
        # x = self.feature_extractor(x)
        x = self.resize(x)
        x = x.view(x.size(0), -1)  # flatten the feature maps
        x = self.fc(x)
        return x

# TODO Task 2c - Complete the TextMLP class

# class TextMLP(nn.Module):
#     def __init__(self, vocab_size, sentence_len, hidden_size, n_classes=4):
#         super().__init__()
#         self.seq = nn.Sequential(
#             nn.Embedding(vocab_size, hidden_size//2),
#             nn.Flatten(),
#             # To determine the input size of the following linear layer think 
#             # about the number of words for each sentence and the size of each embedding. 
#             ## nn.Linear(.... ,  hidden_size),
#             #.....

#         )

class TextMLP(nn.Module):
    def __init__(self, vocab_size, sentence_len, hidden_size, n_classes=4):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Embedding(vocab_size, hidden_size // 2),
            nn.Flatten(),
            nn.Linear(sentence_len * (hidden_size // 2), hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            *[nn.Linear(hidden_size, hidden_size) for _ in range(5)],
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x):
        return self.seq(x)



# TODO Task 2c - Create a model which uses a distilbert-base-uncased
#                by completing the following.
# class DistilBertForClassification(nn.Module):
#     def __init__(self, n_classes=4):
#         super().__init__()
#     #   ....


# class DistilBertForClassification(nn.Module):
#     def __init__(self, n_classes=4):
#         super().__init__()
#         self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
#         self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids, attention_mask)
#         logits = self.classifier(outputs[0])
#         return logits


class DistilBertForClassification(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=n_classes)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
