import torch

from datahandling import get_dev_data
from tqdm import tqdm
import config 

model_id = "apple/aimv2-large-patch14-224"
# model_id = "apple/aimv2-1B-patch14-224"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")
train_loader, val_loader, test_loader = get_dev_data(image_processor_model_id=model_id, debug_mode=False)  

# Initialize model and move to device
from models.vqa_model import VQA, train_model
model = VQA(vision_model_id=model_id).to(device)

print(model)
print(f"Number of parameters in model: {sum([param.nelement() for param in model.parameters()])}")

train_model(model, 
            "vqa_aimv2.pth",
            train_loader, 
            val_loader,
            num_epochs=15,
            device=device)





