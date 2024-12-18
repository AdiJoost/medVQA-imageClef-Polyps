from typing import Iterable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.utils.data.dataloader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModel, AutoModelForImageTextToText
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
from tqdm import tqdm
import config 
from datahandling import load_multilabel_binarizer

# Setup paths and constants
LOG_DIR = os.path.join(config.train_logs_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
writer = SummaryWriter(LOG_DIR)
print(f"Find logs at {LOG_DIR}")


mlb = load_multilabel_binarizer()

# Constants
NUM_LABELS = len(mlb.classes_)
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01


# Define Model Classes
class VQA(nn.Module):
    """
    VQA Model class that gets the embeddings of the different modalities and then gets a prediction from the classifier
    """
    def __init__(self, vision_model_id: str = "microsoft/beit-base-patch16-224-pt22k-ft22k"):
        super(VQA, self).__init__()
        self.vision_model_id = vision_model_id
        self.image_encoder = ImageEncoder(self.vision_model_id)
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
    def __init__(self, vision_model_id: str):
        """
        Args:
            vision_model_id (str): the model_id from huggingface 
        """
        super(ImageEncoder, self).__init__()
        self.vision_model_id = vision_model_id
        self.pre_trained = AutoModel.from_pretrained(self.vision_model_id, trust_remote_code=True)          
        self.output_dim = self.pre_trained.config.hidden_size

    def forward(self, image):
        outputs = self.pre_trained(pixel_values=image)
        
        if hasattr(outputs, "pooler_output"):
            pooled_output = outputs.pooler_output
        else:
            # Fallback to mean pooling over the last_hidden_state
            last_hidden_state = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]
            pooled_output = last_hidden_state.mean(dim=1)  # Mean pooling along sequence length
                    
        image_embedding = pooled_output
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
        return logits # BCEwithlogits combines sigmoid and bceloss (takes logits as input) it is more numerically stable to use the one with bcewithlogits


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
                model_name: str,
                train_loader: torch.utils.data.DataLoader, 
                val_loader: torch.utils.data.DataLoader, 
                num_epochs: int, 
                device: torch.device):
    """
    Trains the VQA Model with provided data, for num_epochs 

    Args:
        model (nn.Module): The VQA Model
        train_loader (torch.utils.data.DataLoader): Train Data with ((image, question, attentionmask), answer) format per sample
        val_loader (torch.utils.data.DataLoader): Val Data with ((image, question, attentionmask), answer) format per sample
        num_epochs (int): Number of epochs to train the model
        device (torch.device): The device to train on
    """
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    best_f1 = 0
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, y_pred, y_true = validate_epoch(model, val_loader, criterion, device)
        
        # last layer outputs logits sigmoid(logits) -> probabilities
        y_pred = torch.sigmoid(y_pred).detach().cpu() >= 0.5  # Threshold predictions
        y_true = y_true.detach().cpu()

        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="samples", zero_division=0)
        recall = recall_score(y_true, y_pred, average="samples", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="samples", zero_division=0)

        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('F1/val', f1, epoch)
        writer.add_scalar('accuracy/val', accuracy, epoch)
        writer.add_scalar('precision/val', precision, epoch)
        writer.add_scalar('recall/val', recall, epoch)

        # Save the model if it improved
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(config.trained_model_path, model_name))
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        scheduler.step()
    
    writer.flush()