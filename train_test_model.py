import os
import json
import yaml
from pathlib import Path
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ESC50Dataset import prepare_esc50_loaders
from models import SoundCNN, ResNet18, SoundCNN_Variable, SoundCNN_MFCCConcat
from train import train_epoch, validate

 # MAKE EXPERIMENTS REPEATABLE
SEED = 1

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config():
    config_path = Path("Config/config.yaml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(f"Configurazione caricata da {config_path}: {config}")
    return config


def train_test_model():
    config = load_config()
    
    lr = config['learning_rate']
    batch_size = config['batch_size']
    epochs =  config['epochs']
    optimizer_name = config['optimizer']
    dropout = config['dropout']
    input_size = config['input_size']
    kernel_size = config['kernel_size']
    stride = config['stride']
    n_blocks = config['n_blocks']
    data_augmentation = config['data_augmentation']

    # save_dir = Path("Datasets/ESC50/Dataloaders")
    # trainloader_path = Path(f"Datasets/ESC50/Dataloaders/trainloader.pt")
    # valloader_path = Path(f"Datasets/ESC50/Dataloaders/valloader.pt")
    # testloader_path = Path(f"Datasets/ESC50/Dataloaders/testloader.pt")

    # if trainloader_path.exists() and valloader_path.exists() and testloader_path.exists():
    #     print(f"Dataloaders trovati, uploading da disco...")

    #     train_loader, val_loader, test_loader = load_dataloaders_from_disk(save_dir, batch_size)

    # Imposta i path automatici relativi al file corrente
    base_dir = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(base_dir, "Datasets", "ESC50", "audio/audio")
    meta_path = os.path.join(base_dir, "Datasets", "ESC50", "meta", "esc50.csv")

    # Verify Dataset Directory exists
    if os.path.isdir(audio_path) and os.path.isfile(meta_path):
        dataset_dir = audio_path
        meta_file = meta_path
        print(f"Loaded ESC-50 dataset:\n  Audio: {audio_path}\n  Meta: {meta_path}")

        # Preprocessing: chiama la funzione per preparare i DataLoader
        try:
            train_loader, val_loader, test_loader = prepare_esc50_loaders(
                dataset_dir, meta_file, batch_size, data_augmentation=data_augmentation
            )
            print(f"Dataset preprocessing completed.")
        except Exception as e:
            print(f"Error during dataset preprocessing: {e}")

    else:
        print("ESC-50 dataset not found. Check the directory structure.")
        
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SoundCNN_MFCCConcat(
        input_size=input_size,
        kernel_size=kernel_size, 
        stride=stride, 
        dropout=dropout, 
        n_blocks=n_blocks, 
        mfcc_feature_size=2000).to(device)

    criterion = nn.CrossEntropyLoss()

    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        print("Error, strategy not available")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    print("Start training...")
    print("Training on device:", device)

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, epochs)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

    print("Training completed.")

    # Valutazione finale sul test set
    print("Final test of the model.")
    print("Testing on device:", device)
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    set_seed(SEED)
    train_test_model()