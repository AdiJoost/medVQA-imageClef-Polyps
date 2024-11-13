import torch
from torch import nn
from datahandling import get_dev_data
from tqdm import tqdm
import config 
from models.vqa_model import VQA
from os.path import join
import torchmetrics


# Change this to the filename of the saved model in the models/trained folder
MODEL_NAME = "vqa_model.pth"

# Dont change this unless there are mor answers possible than before
NUM_LABELS = 117

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")


model = VQA()
model.load_state_dict(torch.load(join(config.trained_model_path, MODEL_NAME), weights_only=True))
model.to(device)
model.eval()

_, _, test_loader = get_dev_data(debug_mode=True)  
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

# todo check if these are correct
accuracy = torchmetrics.functional.accuracy(y_pred, y_true.int(), task="multilabel", average='macro', multidim_average='global', num_labels=NUM_LABELS)
precision = torchmetrics.functional.precision(y_pred, y_true.int(), task="multilabel", average='macro', multidim_average='global', num_labels=NUM_LABELS)
recall = torchmetrics.functional.recall(y_pred, y_true.int(), task="multilabel", average='macro', multidim_average='global', num_labels=NUM_LABELS)
f1_score = torchmetrics.functional.f1_score(y_pred, y_true.int(), task="multilabel", average='macro', multidim_average='global', num_labels=NUM_LABELS)

test_loss = loss / len(test_loader)

results = f""" Performance on test set
      Test Loss: {test_loss}, 
      F1 Score: {f1_score},  
      accuracy: {accuracy},  
      precision: {precision},  
      recall: {recall}"""

print(results)
