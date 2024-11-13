import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torchmetrics
from transformers import AutoTokenizer, AutoModel, AutoModelForImageClassification
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
from dataset_torch import get_dev_data
from tqdm import tqdm
import config 
from torchsummary import summary

# Setup paths and constants
MODEL_NAME = "vqa_model.pth"
MODEL_PATH = os.path.join(config.trained_model_path, MODEL_NAME)
LOG_DIR = os.path.join(config.train_logs_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
writer = SummaryWriter(LOG_DIR)

# Constants
NUM_LABELS = 117
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01


# Define Model Classes
class VQA(nn.Module):
    def __init__(self):
        super(VQA, self).__init__()
        self.image_encoder = ImageEncoder()
        self.question_encoder = QuestionEncoder()
        self.classifier = Classifier(self.image_encoder.output_dim, self.question_encoder.output_dim)
        
    def forward(self, image, question, question_attention_mask):
        image_embedding = self.image_encoder(image)
        question_embedding = self.question_encoder(question, question_attention_mask)
        return self.classifier(question_embedding, image_embedding)

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.pre_trained = AutoModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")        
        self.output_dim = self.pre_trained.config.hidden_size

    def forward(self, image):
        outputs = self.pre_trained(pixel_values=image)
        image_embedding = outputs.pooler_output # single vector (instead of embeddings for different patches for example)
        return image_embedding

class QuestionEncoder(nn.Module):
    def __init__(self):
        super(QuestionEncoder, self).__init__()
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        self.output_dim = self.bert_model.config.hidden_size  

    def forward(self, question, question_attention_mask):
        outputs = self.bert_model(input_ids=question, attention_mask=question_attention_mask)
        question_embedding = outputs.pooler_output # single vector output, sequence_output would be each word embedded separately
        return question_embedding

class Classifier(nn.Module):
    def __init__(self, image_output_dim, question_output_dim):
        super(Classifier, self).__init__()
        self.fc_1 = nn.Linear(image_output_dim + question_output_dim, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc_2 = nn.Linear(512, NUM_LABELS)
        
    def forward(self, question_embedding, image_embedding):
        full_embedding = torch.cat((question_embedding, image_embedding), dim=1)
        x = self.fc_1(full_embedding)
        x = torch.relu(x)
        x = self.dropout(x)
        logits = self.fc_2(x)
        return logits # BCEwithlogits combines sigmoid and bceloss (takes logits as input)


# Define training components
def get_optimizer(model):
    return AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def get_scheduler(optimizer):
    decay_rate = 0.9333
    scheduler = LambdaLR(optimizer,  lr_lambda=lambda epoch: decay_rate**epoch)
    return scheduler
    
    
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for batch in tqdm(dataloader, desc="Train Epoch"):
        (images, questions, question_attention_mask), labels = batch
        images, questions, question_attention_mask, labels = images.to(device), questions.to(device), question_attention_mask.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, questions, question_attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val Epoch"):
            (images, questions, question_attention_mask), labels = batch
            images, questions, question_attention_mask, labels = images.to(device), questions.to(device), question_attention_mask.to(device), labels.to(device)
            
            outputs = model(images, questions, question_attention_mask)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            
            all_outputs.append(outputs)
            all_labels.append(labels)
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    
    return epoch_loss / len(dataloader), all_outputs, all_labels # returns average loss over all batches in dataset, the predicted labels and true labels

def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    
    best_f1 = 0
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_outputs, val_labels = validate_epoch(model, val_loader, criterion, device)
        
        # Calculate metrics
        # multilabel: bc. multiple answers can be predicted as 1 in the answer vector
        # samples: calculates the acc/rec/pre/f1 for each sample separately and then averages over all samples. 
        accuracy = torchmetrics.functional.accuracy(val_outputs, val_labels.int(), task="multilabel", average='macro', multidim_average='global', num_labels=NUM_LABELS)
        precision = torchmetrics.functional.precision(val_outputs, val_labels.int(), task="multilabel", average='macro', multidim_average='global', num_labels=NUM_LABELS)
        recall = torchmetrics.functional.recall(val_outputs, val_labels.int(), task="multilabel", average='macro', multidim_average='global', num_labels=NUM_LABELS)
        f1_score = torchmetrics.functional.f1_score(val_outputs, val_labels.int(), task="multilabel", average='macro', multidim_average='global', num_labels=NUM_LABELS)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('F1 Score/val', f1_score, epoch)
        writer.add_scalar('accuracy/val', accuracy, epoch)
        writer.add_scalar('precision/val', precision, epoch)
        writer.add_scalar('recall/val', recall, epoch)


        # Save the model if it improved
        if f1_score > best_f1:
            best_f1 = f1_score
            torch.save(model.state_dict(), MODEL_PATH)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, F1 Score: {f1_score}")
        
        scheduler.step()

