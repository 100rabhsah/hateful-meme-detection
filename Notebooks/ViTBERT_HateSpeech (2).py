from google.colab import drive
drive.mount('/content/drive')

!pip install transformers
!pip install torch torchvision

import numpy as np
import pandas as pd
import json
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, Dict, List
import torch
from torchvision import transforms
import torchvision.models as models
from timeit import default_timer as timer
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer, ViTModel

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

train_data = read_jsonl('/content/drive/MyDrive/hateful_memes/train.jsonl')

def create_dataframe(data):
    df = pd.DataFrame(data)
    return df

df = create_dataframe(train_data)

df

all_ids = df['id'].tolist()

all_ids[:10]

images_dict = {}
for i in range(len(df)):
  img = df.loc[df['id'] == all_ids[i]].iloc[0]['img']
  images_dict[all_ids[i]] = img

text_dict = {}
for i in range(len(df)):
  text = df.loc[df['id'] == all_ids[i]].iloc[0]['text']
  text_dict[all_ids[i]] = text

directory2 = '/content/drive/MyDrive/hateful_memes/'

labels = df['label'].tolist()

checkpoint_path = '/content/drive/MyDrive/hateful_memes/checkpoint.pth'

len(labels)

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from PIL import Image
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_dict, text_dict, tokenizer, max_length, labels, transform=None):
        self.image_dict = image_dict
        self.text_dict = text_dict
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        self.labels = labels
        self.image_ids = list(image_dict.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = self.image_dict[image_id]
        image = Image.open(directory2 + img_path).convert('RGB')
        text = self.text_dict[image_id]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        if self.transform:
            image = self.transform(image)
        return image, input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

from transformers import BertTokenizer

# Assume text_dict is already defined
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Step 1: Calculate the maximum length
def get_max_length(text_dict, tokenizer):
    max_length = 0
    for text in text_dict.values():
        tokens = tokenizer.encode(text, add_special_tokens=True)
        max_length = max(max_length, len(tokens))
    return max_length

# Calculate max_length from data
max_length = get_max_length(text_dict, tokenizer)
print(f"Determined max_length: {max_length}")

# Step 2: Use the calculated max_length in the dataset

dataset = CustomDataset(images_dict, text_dict, tokenizer, max_length=max_length, labels=labels, transform=transform)

train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, pin_memory=True,shuffle=False)

for image, id, att, label in train_loader:
  print(label)
  print(label.shape)
  break

import torch
import torch.optim as optim
import torch.nn as nn
from transformers import BertModel, BertTokenizer, ViTModel

class HatefulMemesClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', vit_model_name='google/vit-base-patch16-224'):
        super(HatefulMemesClassifier, self).__init__()

        # BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_fc1 = nn.Linear(768, 128)
        self.bert_fc2 = nn.Linear(128, 64)
        self.bert_dropout = nn.Dropout(0.3)

        # Vision Transformer (ViT) model
        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.vit_lstm = nn.LSTM(input_size=768, hidden_size=128, batch_first=True, bidirectional=True)
        self.vit_fc = nn.Linear(256, 64)  # 128*2 because of bidirectional LSTM
        self.vit_dropout = nn.Dropout(0.3)

        # Fusion and classification head
        self.fusion_fc1 = nn.Linear(128, 64)
        self.fusion_fc2 = nn.Linear(64, 2)  # Assuming binary classification

    def forward(self, input_ids, attention_mask, pixel_values):
        # BERT forward pass
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls = bert_outputs.last_hidden_state[:, 0, :]  # CLS token
        bert_x = self.bert_dropout(bert_cls)
        bert_x = torch.relu(self.bert_fc1(bert_x))
        bert_x = torch.relu(self.bert_fc2(bert_x))

        # ViT forward pass
        vit_outputs = self.vit(pixel_values=pixel_values)
        vit_cls = vit_outputs.last_hidden_state  # All tokens
        vit_x, _ = self.vit_lstm(vit_cls)
        vit_x = vit_x[:, -1, :]  # Take the last hidden state
        vit_x = self.vit_dropout(vit_x)
        vit_x = torch.relu(self.vit_fc(vit_x))

        # Fusion
        combined = torch.cat((bert_x, vit_x), dim=1)
        combined = torch.relu(self.fusion_fc1(combined))
        logits = self.fusion_fc2(combined)

        return logits

# Example usage
model = HatefulMemesClassifier()
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

from sklearn.metrics import precision_recall_fscore_support
def save_model(model, save_dir=directory2, filename='final_model.pth'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def validate_model(model, dataloader, criterion, device='cuda'):
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, input_ids, attention_mask, labels in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            total_samples += labels.size(0)

            _, preds = torch.max(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    accuracy = sum(np.array(all_preds) == np.array(all_labels)) / total_samples

    epoch_loss = running_loss / len(dataloader)
    print(f'Validation Loss: {epoch_loss:.4f} Accuracy: {accuracy:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1 Score: {f1:.4f}')
    return epoch_loss, accuracy, precision, recall, f1

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def calculate_metrics(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
    accuracy = torch.sum(preds == labels).item() / len(labels)
    return accuracy, precision, recall, f1

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, device='cuda'):
    model.to(device)  # Move model to the specified device

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        all_labels = []
        all_preds = []

        # Create a progress bar with tqdm
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for i, (images, input_ids, attention_mask, labels) in loop:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_samples += labels.size(0)

            _, preds = torch.max(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # Update the progress bar with current loss and accuracy
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=running_loss / (i+1))

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        accuracy = sum(np.array(all_preds) == np.array(all_labels)) / total_samples

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_loss:.4f} Accuracy: {accuracy:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1 Score: {f1:.4f}")

        # Validate the model after each epoch
        validate_model(model, val_loader, criterion, device)

    # Save the model at the end of training
    save_model(model)
    print("Training complete!")

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, device='cuda')

def test_model(model, test_loader, criterion, device='cuda'):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    total_samples = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, input_ids, attention_mask, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            total_samples += labels.size(0)

            _, preds = torch.max(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    accuracy = sum(np.array(all_preds) == np.array(all_labels)) / total_samples

    epoch_loss = running_loss / len(test_loader)
    print(f'Test Loss: {epoch_loss:.4f} Accuracy: {accuracy:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1 Score: {f1:.4f}')
    return epoch_loss, accuracy, precision, recall, f1

test_loss, test_accuracy, test_precision, test_recall, test_f1 = test_model(model, test_loader, criterion, device='cuda')



