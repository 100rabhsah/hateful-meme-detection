from google.colab import drive
drive.mount('/content/drive')

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

input_dir = '/content/drive/MyDrive/hateful_memes/'
output_dir = '/content/drive/MyDrive/hateful_memes/'

import albumentations as A
from PIL import Image
import numpy as np
import os
import pandas as pd

# Define the augmentation pipeline
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.3),
])

df['label'].value_counts()

augmented_data = []
# Set the desired number of augmented images
desired_augmented_count = 2500
augmented_count = 0  # Counter for augmented images

# Augment each image in class 1
for idx, row in df.iterrows():
    # Check if the label is 1
    if row['label'] == 1:
        img_path = os.path.join(input_dir, row['img'])
        img = np.array(Image.open(img_path))

        # Perform augmentation
        augmented = augmentations(image=img)['image']
        augmented_image = Image.fromarray(augmented)

        # Create new image ID and name
        new_image_id = f"aug_{row['id']}_{augmented_count}"
        new_image_name = f"augmented_img/aug_{row['id']}_{augmented_count}.png"

        # Save the augmented image
        augmented_image.save(os.path.join(output_dir, new_image_name))

        # Add new entry to the augmented data (new id and name, but same label and text)
        augmented_data.append({
            'id': new_image_id,
            'img': new_image_name,
            'label': row['label'],  # Same label
            'text': row['text']     # Same text
        })

        augmented_count += 1  # Increment the counter

        # Break the loop if the desired count is reached
        if augmented_count >= desired_augmented_count:
            break

# Convert augmented data to DataFrame
augmented_df = pd.DataFrame(augmented_data)

# Concatenate original dataframe and augmented dataframe
final_df = pd.concat([df, augmented_df], ignore_index=True)

final_df

# Shuffle the DataFrame
shuffled_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

shuffled_df

# Save the updated dataframe
final_df.to_csv('/content/drive/MyDrive/hateful_memes/updated_training_data.csv', index=False)
# Save the shuffled DataFrame (optional)
shuffled_df.to_csv('/content/drive/MyDrive/hateful_memes/shuffled_training_data.csv', index=False)

shuffled_df['label'].value_counts()



