from typing import Iterable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.utils.data.dataloader
import torchmetrics
from transformers import AutoTokenizer, AutoModel, AutoModelForImageClassification
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
from datahandling import get_dev_data
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
    """
    VQA Model class that gets the embeddings of the different modalities and then gets a prediction from the classifier
    """
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
    """
    The Image Encoder Takes the image as input and gets an embedding from the pretrained model
    We take the pooled output
    """
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.pre_trained = AutoModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")        
        self.output_dim = self.pre_trained.config.hidden_size

    def forward(self, image):
        outputs = self.pre_trained(pixel_values=image)
        image_embedding = outputs.pooler_output # single vector (instead of embeddings for different patches for example)
        return image_embedding

class QuestionEncoder(nn.Module):
    """
    Takes the Tokenized Question and the attention mask as input to produce an embedding 
    Because all samples in the batch need to be the same length, some need to be padded,
    the attention mask basically tells the encoder where the text is and where is just padding
    ex. 1 for text 0 for padding
    """
    def __init__(self):
        super(QuestionEncoder, self).__init__()
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        self.output_dim = self.bert_model.config.hidden_size  

    def forward(self, question, question_attention_mask):
        outputs = self.bert_model(input_ids=question, attention_mask=question_attention_mask)
        question_embedding = outputs.pooler_output # single vector output, sequence_output would be each word embedded separately
        return question_embedding

class Classifier(nn.Module):
    """
    The Classifier concatenates the embeddings from image and question
    It is then fed into a simple net for classification.
    It predicts an answer vector 
    """
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


def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """
    Returns an Adam optimizer with weighted decay

    Args:
        model (nn.Module): VQA Model

    Returns:
        torch.optim.Optimizer: The AdamW Optimizer
    """
    return AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

def get_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler:
    """
    The linear scheduler
    it updates the learningrate of the optimizer everytime it is called with scheduler.step()

    Args:
        optimizer (torch.optim.Optimizer): The optimizer where the linear learning rate should be applied

    Returns:
        torch.optim.lr_scheduler: the scheduler
    """
    decay_rate = 0.9333
    scheduler = LambdaLR(optimizer,  lr_lambda=lambda epoch: decay_rate**epoch)
    return scheduler
    
def train_epoch(model: nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer,
                criterion,
                device: torch.device) -> float:
    """
    One train Epoch

    Args:
        model (nn.Module): the vqa model
        dataloader (torch.utils.data.DataLoader): dataloader with ((image, question, attentionmask),answer) format per sample
        optimizer (torch.optim.Optimizer): _description_
        criterion (Loss): _description_
        device (torch.device): _description_

    Returns:
        float: Average loss over epoch
    """
    
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

def validate_epoch(model: nn.Module, 
                   dataloader: torch.utils.data.DataLoader, 
                   criterion, 
                   device: torch.device)-> tuple[float, torch.Tensor, torch.Tensor]:
    """
    Runs a Validation and returns the average loss over the validataion dataset 
    Also returns y_true and y_pred for calculation evaluation metrics later

    Args:
        model (nn.Module): The VQA Model
        dataloader (torch.utils.data.DataLoader): Validation Datasetloader
        criterion (_type_): Lossfunction
        device (torch.device): Where to run the validation 

    Returns:
        tuple[float, list, list]: Avg Loss over batches in dataset, y_pred, y_true
    """
    model.eval()
    epoch_loss = 0
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val Epoch"):
            (images, questions, question_attention_mask), labels = batch
            images, questions, question_attention_mask, labels = images.to(device), questions.to(device), question_attention_mask.to(device), labels.to(device)
            
            outputs = model(images, questions, question_attention_mask)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            
            y_pred.append(outputs)
            y_true.append(labels)
    
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    
    return epoch_loss / len(dataloader), y_pred, y_true # returns average loss over all batches in dataset, the predicted labels and true labels

def train_model(model: nn.Module, 
                train_loader: torch.utils.data.DataLoader, 
                val_loader: torch.utils.data.DataLoader, 
                num_epochs: int, 
                device: torch.device):
    """
    Trains the VQA Model with procided data, for num_epochs 

    Args:
        model (nn.Module): The VQA Model
        train_loader (torch.utils.data.DataLoader): Train Data with ((image, question, attentionmask),answer) format per sample
        val_loader (torch.utils.data.DataLoader): Val Data with ((image, question, attentionmask),answer) format per sample
        num_epochs (int): number of epochs to train the model
        device (torch.device): the device to train on
    """
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    best_f1 = 0
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_outputs, val_labels = validate_epoch(model, val_loader, criterion, device)
        
        # Calculate metrics
        # multilabel: bc. multiple answers can be predicted as 1 in the answer vector
        # TODO Check if this is correct!!! 
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
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, F1 Score: {f1_score},  accuracy: {accuracy},  precision: {precision},  recall: {recall}")
        
        scheduler.step()

