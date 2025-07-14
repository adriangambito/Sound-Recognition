import os
import json

import torch
from tqdm import tqdm


# Funzione di training
def train_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc=f'Epoch {epoch+1}/{total_epochs}'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()

    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy



# Funzione di validazione
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# TODO: test function




def load_hyperparams_from_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Hyperparameters model.pt not found: {checkpoint_path}")

    json_path = checkpoint_path.replace(".pt", ".json")
    with open(json_path, "r") as f:
        hyperparams = json.load(f)

    return hyperparams

