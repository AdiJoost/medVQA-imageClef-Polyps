# wtf are tf strings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, ViTFeatureExtractor, ViTModel
import torchvision.models as models
from dataset_torch import prepare_datasets
import config
import os
import datetime
# Setup paths
MODEL_NAME = os.path.basename(__file__).split(".")[0] + ".pt"
MODEL_PATH = os.path.join(config.trained_model_path, MODEL_NAME)
MODEL_LOG_DIR = os.path.join(config.train_logs_path, os.path.basename(__file__).split(".")[0])
print(MODEL_LOG_DIR)
os.makedirs(MODEL_LOG_DIR, exist_ok=True)


class VQA(nn.Module):
    def __init__(self):
        super(VQA, self).__init__()
        self.image_encoder = ImageEncoder()
        self.question_encoder = QuestionEncoder()
        self.classifier = Classifier()

    def forward(self, image, question):
        image_embedding = self.image_encoder(image)
        question_embedding = self.question_encoder(question)
        answer_probabilities = self.classifier(question_embedding, image_embedding)
        return answer_probabilities


class ImageEncoder(nn.Module):
    """
    Encodes the image input to an embedding vector of size 512.
    """
    def __init__(self):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove last fully connected layer
        self.pooling = nn.AdaptiveAvgPool2d((1, 1)) #should be same as globalaveragepooling but check
        self.dense = nn.Linear(2048, 512)

    def forward(self, image):
        x = self.feature_extractor(image)
        x = self.pooling(x).view(x.size(0), -1)  # Flatten
        image_embedding = torch.relu(self.dense(x))
        return image_embedding


class QuestionEncoder(nn.Module):
    """
    Encodes the question input to an embedding vector.
    """
    def __init__(self):
        super(QuestionEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, question):
        inputs = self.tokenizer(question, return_tensors='pt', padding='max_length', max_length=64, truncation=True)
        outputs = self.bert_model(**inputs)
        question_embedding = outputs.pooler_output  # Shape: (batch_size, hidden_size)
        return question_embedding


class Classifier(nn.Module):
    """
    Classifies based on combined question and image embeddings.
    """
    def __init__(self, num_labels=117):
        super(Classifier, self).__init__()
        self.dense_fc = nn.Linear(512 + 768, 512)  # Combined embedding sizes
        self.dropout = nn.Dropout(0.5)
        self.dense_output = nn.Linear(512, num_labels)

    def forward(self, question_embedding, image_embedding):
        full_embedding = torch.cat((question_embedding, image_embedding), dim=1)
        x = torch.relu(self.dense_fc(full_embedding))
        x = self.dropout(x)
        probabilities = torch.sigmoid(self.dense_output(x))
        return probabilities


def train_model():
    train_loader, test_loader, val_loader = prepare_datasets()

    model = VQA()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.BCEWithLogitsLoss()
    best_f1 = 0.0

    for epoch in range(15):
        model.train()
        train_loss = 0.0
        for images, questions, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images, questions)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss, val_f1 = 0.0, 0.0
        with torch.no_grad():
            for images, questions, labels in val_loader:
                outputs = model(images, questions)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = (outputs > 0.5).float()
                val_f1 += f1_score(labels, preds, average='samples')

        val_f1 /= len(val_loader)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), MODEL_PATH)

        print(f"Epoch {epoch + 1} - Train Loss: {train_loss / len(train_loader):.4f} "
              f"Val Loss: {val_loss / len(val_loader):.4f} Val F1: {val_f1:.4f}")


if __name__ == "__main__":
    train_model()
