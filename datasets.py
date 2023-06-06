import collections
import csv
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer

class LesionDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, labels_fname, augment=False):
        self.img_dir = Path(img_dir)
        self.labels = pd.read_csv(labels_fname)
        self.augment = augment

        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_fname = self.img_dir / f"{self.labels.iloc[idx]['image']}.jpg"
        img = Image.open(img_fname).convert("RGB")
        img_tensor = self.transform(img)
        
        label_values = self.labels.iloc[idx][1:].to_numpy()
        label = torch.tensor(np.argmax(label_values), dtype=torch.long)
        
        return img_tensor, label
        
    def get_image_size(self, idx):
        img_fname = self.img_dir / f"{self.labels.iloc[idx]['image']}.jpg"
        img = Image.open(img_fname).convert("RGB")
        return img.size





class TextDataset(torch.utils.data.Dataset):
    def __init__(self, fname, sentence_len):
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.sentence_len = sentence_len
        self.data = pd.read_csv(fname, header=None)
        self.texts = self.data.iloc[:, 2].values
        self.labels = self.data.iloc[:, 0].values - 1 

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.sentence_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            # 'labels': torch.eye(4)[label].squeeze().long()
            'labels': torch.tensor(label).long()

        }

