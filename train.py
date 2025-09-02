import os
import json
import torch
from tqdm import tqdm


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# Training function stays the same
def train_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, total_epochs):
    model.to(device)
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for mel_inputs, mfcc_inputs, labels in tqdm(loader, desc=f'Epoch {epoch+1}/{total_epochs}'):
        mel_inputs = mel_inputs.to(device)
        mfcc_inputs = mfcc_inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(mel_inputs, mfcc_inputs)
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


# Validation function stays the same
def validate(model, loader, criterion, device):
    model.to(device)
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for mel_inputs, mfcc_inputs, labels in loader:
            mel_inputs = mel_inputs.to(device)
            mfcc_inputs = mfcc_inputs.to(device)
            labels = labels.to(device)

            outputs = model(mel_inputs, mfcc_inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def load_hyperparams_from_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Hyperparameters model.pt not found: {checkpoint_path}")

    json_path = checkpoint_path.replace(".pt", ".json")
    with open(json_path, "r") as f:
        hyperparams = json.load(f)

    return hyperparams


# Example usage of early stopping in a training loop
def train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, scheduler, device, total_epochs=50, patience=5):
    early_stopping = EarlyStopping(patience=5)
    best_model_state = None

    for epoch in range(total_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, total_epochs)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

        # Check early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        # Save best model
        if best_model_state is None or val_loss < early_stopping.best_loss:
            best_model_state = model.state_dict()

    # Load best model before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model
