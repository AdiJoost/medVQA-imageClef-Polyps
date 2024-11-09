import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoImageProcessor
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import os
from os.path import join
import pathlib
import json
from PIL import Image
import config
from torch.nn.utils.rnn import pad_sequence



QUESTIONS_TO_BE_ANSWERED = [
    "What type of procedure is the image taken from?",
    "How many instrumnets are in the image?",  # spelled wrong, but is like this in their test script too
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

class VQADataset(Dataset):
    def __init__(self, X, y, image_processor, tokenizer):
        self.X = X
        self.y = y
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        question = self.X[idx][1]
        image_id = self.X[idx][0]
        image_path = join(config.data_raw_dev_images, f"{image_id}.jpg")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze()
        
        # Tokenize question with attention mask
        question_tokens = self.tokenizer(
            question,
            padding='longest', #padded to longest question
            return_tensors='pt'
        )
        
        question_ids = question_tokens["input_ids"].squeeze()
        attention_mask = question_tokens["attention_mask"].squeeze()  # Get attention mask
        
        # Get answer vector
        answer_vector = torch.tensor(self.y[idx], dtype=torch.float32)
        
        return (image, question_ids, attention_mask), answer_vector

def _load_dev_data():
    if not os.path.exists(join(config.data_processed_dev, "X_train.npy")):
        _encode_dev_data()
    
    X_train = np.load(join(config.data_processed_dev, "X_train.npy"), allow_pickle=True)
    y_train = np.load(join(config.data_processed_dev, "y_train.npy"), allow_pickle=True)
    X_test = np.load(join(config.data_processed_dev, "X_test.npy"), allow_pickle=True)
    y_test = np.load(join(config.data_processed_dev, "y_test.npy"), allow_pickle=True)
    X_val = np.load(join(config.data_processed_dev, "X_val.npy"), allow_pickle=True)
    y_val = np.load(join(config.data_processed_dev, "y_val.npy"), allow_pickle=True)

    return X_train, X_test, X_val, y_train, y_test, y_val

def _get_label_encoder():
    labels_json_path = list(pathlib.Path(config.data_raw_dev).rglob("*.json"))[0]
    with open(labels_json_path, "r") as f:
        data = json.load(f)
    
    question_answers = {}
    for image in data:
        for label in image["Labels"]:
            if label["Question"] in QUESTIONS_TO_BE_ANSWERED:
                if label["Question"] not in question_answers:
                    question_answers[label["Question"]] = []
                
                question_answers[label["Question"]] += [label["Question"] + "_" + answer for answer in label["Answer"]]  
                question_answers[label["Question"]] = list(set(question_answers[label["Question"]]))

    labels = [item for sublist in question_answers.values() for item in sublist]
    mlb = MultiLabelBinarizer()
    mlb.fit([labels])
    return mlb

def _encode_answer(answers, encoder):
    return encoder.transform([answers])[0]

def _get_X_and_encoded_y(encoder):
    labels_json_path = list(pathlib.Path(config.data_raw_dev).rglob("*.json"))[0]
    with open(labels_json_path, "r") as f:
        data = json.load(f)

    no_answer_vector = np.zeros(shape=(117,), dtype=np.int32)
    stratify_question = "Are there any abnormalities in the image?"

    X, y, stratify = [], [], []

    for image in data:
        image_id = image["ImageID"]
        questions = [label["Question"] for label in image["Labels"]]
        answers_all_questions = [label["Answer"] for label in image["Labels"]]
        stratify_label = "_".join(answers_all_questions[questions.index(stratify_question)])

        for q in QUESTIONS_TO_BE_ANSWERED:
            if q in questions:
                q_index = questions.index(q)
                answers = answers_all_questions[q_index]
                answer_vector = _encode_answer([q + "_" + answer for answer in answers], encoder=encoder)
            else:
                answer_vector = no_answer_vector

            X.append([image_id, q])
            y.append(answer_vector)
            stratify.append(stratify_label)
            
    return X, y, stratify

def _train_test_val_split_stratified(X, y, stratify):
    X_train, X_test, y_train, y_test, stratify_train, stratify_test = train_test_split(
        X, y, stratify, test_size=0.2, random_state=42, stratify=stratify
    )
    
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=stratify_test
    )
    return X_train, X_test, X_val, y_train, y_test, y_val

def _encode_dev_data(save=True):
    mlb = _get_label_encoder()
    X, y, stratify = _get_X_and_encoded_y(encoder=mlb)
    X_train, X_test, X_val, y_train, y_test, y_val = _train_test_val_split_stratified(X, y, stratify)

    if save:
        os.makedirs(config.data_processed_dev, exist_ok=True)
        np.save(join(config.data_processed_dev, "X_train.npy"), X_train)
        np.save(join(config.data_processed_dev, "y_train.npy"), y_train)
        np.save(join(config.data_processed_dev, "X_test.npy"), X_test)
        np.save(join(config.data_processed_dev, "y_test.npy"), y_test)
        np.save(join(config.data_processed_dev, "X_val.npy"), X_val)
        np.save(join(config.data_processed_dev, "y_val.npy"), y_val)

def collate_fn(batch):
    images, questions, attention_masks, labels = zip(
        *[(item[0][0], item[0][1], item[0][2], item[1]) for item in batch]
    )
    
    images = torch.stack(images)
    questions_padded = pad_sequence(questions, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    
    return (images, questions_padded, attention_masks_padded), labels


    
def get_dev_data():
    image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    mlb = _get_label_encoder()

    # Load data
    X_train, X_test, X_val, y_train, y_test, y_val = _load_dev_data()

    # Create datasets and dataloaders
    train_dataset = VQADataset(X_train, y_train, image_processor, tokenizer)
    test_dataset = VQADataset(X_test, y_test, image_processor, tokenizer)
    val_dataset = VQADataset(X_val, y_val, image_processor, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    print("Data loaded successfully in PyTorch format.")

    return train_loader, test_loader, val_loader