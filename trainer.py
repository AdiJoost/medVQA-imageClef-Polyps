import torch

from dataset_torch import get_dev_data
from tqdm import tqdm
import config 

from models.torch_model import VQA, train_model



# Prepare Data and Start Training
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Running on device: {device}")
    
    train_loader, val_loader, test_loader = get_dev_data()  
    
    # Initialize model and move to device
    model = VQA().to(device)
    
    print(model)
    print(f"Number of parameters in model: {sum([param.nelement() for param in model.parameters()])}")

    train_model(model, train_loader, val_loader, num_epochs=15, device=device)





# model.eval() set this after loading the model before inference