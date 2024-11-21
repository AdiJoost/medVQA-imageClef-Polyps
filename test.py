import torch
from torch import nn
from datahandling import get_dev_data
from tqdm import tqdm
import config 
from models.vqa_model import VQA
from os.path import join

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datahandling import load_multilabel_binarizer




# Change this to the filename of the saved model in the models/trained folder
MODEL_NAME = "vqa_model.pth"

# Dont change this unless there are mor answers possible than before
mlb = load_multilabel_binarizer()
NUM_LABELS = len(mlb.classes_)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    model = load_model(device)
    _, _, test_loader = get_dev_data(debug_mode=False)  
    criterion = nn.BCEWithLogitsLoss()

    loss = 0
    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            (images, questions, question_attention_mask), labels = batch
            images, questions, question_attention_mask, labels = images.to(device), questions.to(device), question_attention_mask.to(device), labels.to(device)
            
            outputs = model(images, questions, question_attention_mask)
            loss = criterion(outputs, labels)
            loss += loss.item()
            
            y_pred.append(outputs)
            y_true.append(labels)

    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    
    
    results = f""" Performance on Dev test set
    accuracy: {accuracy_score(y_true, torch.sigmoid(y_pred)>=0.5):.4f},  
    precision: {precision_score(y_true, torch.sigmoid(y_pred)>=0.5, average="samples", zero_division=0):.4f},  
    recall: {recall_score(y_true, torch.sigmoid(y_pred)>=0.5, average="samples", zero_division=0):.4f}
    F1 Score: {f1_score(y_true, torch.sigmoid(y_pred)>=0.5, average="samples", zero_division=0):.4f},  
    """
    print(results)
    
    with open("test_results.txt", "w") as f:
        f.write(results)
    


def load_model(device) -> torch.nn.Module:
    model = VQA()
    model.load_state_dict(torch.load(join(config.trained_model_path, MODEL_NAME), weights_only=True))
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    main()