import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from PIL import Image

import config
import pathlib


QUESTIONS_TO_BE_ANSWERED = [
    "What type of procedure is the image taken from?",
    "How many instrumnets are in the image?", #spelled wrong, but is like this in their test script too
    "Have all polyps been removed?",
    "Where in the image is the abnormality?",
    "Is this finding easy to detect?",
    "Where in the image is the instrument?",
    "Is there a green/black box artifact?",
    "Are there any abnormalities in the image?",
    "Is there text?",
    "Are there any anatomical landmarks in the image?",
    "What color is the abnormality?",
    "Are there any instruments in the image?",
    "What color is the anatomical landmark?",
    "Where in the image is the anatomical landmark?",
    "How many findings are present?",
    "What is the size of the polyp?",
    "How many polyps are in the image?",
    "What type of polyp is present?"
]

class CustomVQADataset(Dataset):
    def __init__(self, X, y, image_dir, transform=None):
        self.X = X
        self.y = y
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image_id, question = self.X[idx]
        label = self.y[idx]

        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return (image, question), torch.tensor(label, dtype=torch.float32)

def load_image_data():
    # Assuming `config.data_processed_dev` is a directory storing numpy arrays for X_train, y_train, etc.
    X_train = np.load(os.path.join(config.data_processed_dev, "X_train.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(config.data_processed_dev, "y_train.npy"), allow_pickle=True)
    X_test = np.load(os.path.join(config.data_processed_dev, "X_test.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(config.data_processed_dev, "y_test.npy"), allow_pickle=True)
    X_val = np.load(os.path.join(config.data_processed_dev, "X_val.npy"), allow_pickle=True)
    y_val = np.load(os.path.join(config.data_processed_dev, "y_val.npy"), allow_pickle=True)
    return X_train, X_test, X_val, y_train, y_test, y_val

def get_label_encoder():
    labels_json_path = list(pathlib.Path(config.data_raw_dev).rglob("*.json"))[0]
    with open(labels_json_path, "r") as f:
        data = json.load(f)
    
    question_answers = {}
    for image in data:
        for label in image["Labels"]:
            if label["Question"] in QUESTIONS_TO_BE_ANSWERED:
                if label["Question"] not in question_answers:
                    question_answers[label["Question"]] = []
                
                # Add answers to the question
                question_answers[label["Question"]] +=  [label["Question"] + "_" + answer for answer in label["Answer"]]  
                question_answers[label["Question"]] = list(set(question_answers[label["Question"]]))
    
    labels = []
    answers_all_questions = list(question_answers.values())
    for q in answers_all_questions:
        labels.extend(q)
    
    mlb = MultiLabelBinarizer()
    mlb.fit([labels])
    return mlb

def encode_answer(answers, encoder):
    answer_binarized = encoder.transform([answers])
    return answer_binarized[0]

def prepare_datasets(batch_size=64, transform=None):
    # Load preprocessed data
    X_train, X_test, X_val, y_train, y_test, y_val = load_image_data()
    
    image_dir = os.path.join(config.data_raw_dev, "images")
    # Create datasets
    train_dataset = CustomVQADataset(X_train, y_train, image_dir, transform=transform)
    test_dataset = CustomVQADataset(X_test, y_test, image_dir, transform=transform)
    val_dataset = CustomVQADataset(X_val, y_val, image_dir, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, val_loader

if __name__ == "__main__":
    data_dir = config.data_raw_dev  # Define your image directory here
    train_loader, test_loader, val_loader = prepare_datasets(data_dir)
    
    # Print batch example
    for batch in train_loader:
        print(batch)
        break
