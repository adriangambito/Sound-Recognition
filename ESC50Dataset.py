import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


import pandas as pd
import numpy as np
import librosa
from skimage.transform import resize

# Parametri spettrogramma
N_MELS = 128
TARGET_SIZE = (128, 128)

def save_dataloaders(train_loader, val_loader, test_loader, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    def save_loader(loader, name):
        mel_list, mfcc_list, label_list = [], [], []
        for mel, mfcc, label in loader:
            mel_list.append(mel)
            mfcc_list.append(mfcc)
            label_list.append(label)

        mel_tensor = torch.cat(mel_list, dim=0)
        mfcc_tensor = torch.cat(mfcc_list, dim=0)
        label_tensor = torch.cat(label_list, dim=0)

        torch.save((mel_tensor, mfcc_tensor, label_tensor), os.path.join(save_dir, f"{name}.pt"))
        print(f"Salvato {name} in {os.path.join(save_dir, f'{name}.pt')} con {mel_tensor.shape[0]} esempi.")

    save_loader(train_loader, "trainloader")
    save_loader(val_loader, "valloader")
    save_loader(test_loader, "testloader")


def load_dataloaders_from_disk(save_dir, batch_size):
    def load_loader(name):
        mel_tensor, mfcc_tensor, label_tensor = torch.load(os.path.join(save_dir, f"{name}.pt"))

        dataset = torch.utils.data.TensorDataset(mel_tensor, mfcc_tensor, label_tensor)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=(name=="train")
        )
        print(f"Caricato {name} loader da {os.path.join(save_dir, f'{name}.pt')} con {len(dataset)} esempi.")
        return loader

    train_loader = load_loader("trainloader")
    val_loader = load_loader("valloader")
    test_loader = load_loader("testloader")

    return train_loader, val_loader, test_loader



def extract_mel_spectrogram(file_path, n_mels=N_MELS):
    """Extract Mel Spectrogram from audio file."""

    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB


def prepare_spectrogram(spec, target_shape=TARGET_SIZE):
    """Resize the Mel spectogram to fixed size."""

    spec_resized = resize(spec, target_shape, mode='constant', anti_aliasing=True)
    
    return spec_resized


def extract_mfcc(file_path, n_mfcc=40, target_shape=(40, 50)):
    """Extract and resize MFCC features from a WAV file."""
    y, sr = librosa.load(file_path, sr=44100)  # fixed 44.1kHz
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Resize to fixed size (e.g., 40 x 50)
    mfcc_resized = resize(mfcc, target_shape, mode='constant', anti_aliasing=True)

    # Normalize (per-sample z-score)
    mfcc_normalized = (mfcc_resized - mfcc_resized.mean()) / (mfcc_resized.std() + 1e-9)

    return mfcc_normalized  # shape: (40, 50)


class ESC50Dataset(Dataset):
    def __init__(self, audio_dir, meta_file=None, df=None, transform=None):
        self.audio_dir = audio_dir
        if df is not None:
            self.df = df.reset_index(drop=True)
        elif meta_file is not None:
            self.df = pd.read_csv(meta_file)
        else:
            raise ValueError("Devi fornire almeno meta_file o df.")
        self.transform = transform

    def __len__(self):
        return len(self.df)

 
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row['filename']
        label_idx = row['target']
        audio_path = os.path.join(self.audio_dir, file_name)

        # Mel spectrogram
        mel_spec = extract_mel_spectrogram(audio_path)
        mel_spec = prepare_spectrogram(mel_spec)
        mel_spec = (mel_spec - np.mean(mel_spec)) / np.std(mel_spec)
        mel_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)  # [1, 128, 128]

        # MFCC
        mfcc = extract_mfcc(audio_path, n_mfcc=40, target_shape=(40, 50))
        mfcc_tensor = torch.tensor(mfcc.flatten(), dtype=torch.float32)  # [2000]

        return mel_tensor, mfcc_tensor, label_idx

def custom_collate_fn(batch):
    mel, mfcc, label = zip(*batch)
    return (
        torch.stack(mel),
        torch.stack(mfcc),
        torch.tensor(label)
    )



def prepare_esc50_loaders(audio_dir, meta_file, batch_size):

    save_dir = Path("Datasets/ESC50/Dataloaders")
    trainloader_path = Path(f"Datasets/ESC50/Dataloaders/trainloader.pt")
    valloader_path = Path(f"Datasets/ESC50/Dataloaders/valloader.pt")
    testloader_path = Path(f"Datasets/ESC50/Dataloaders/testloader.pt")

    if trainloader_path.exists() and valloader_path.exists() and testloader_path.exists():
        print(f"Dataloaders trovati, uploading da disco...")

        train_loader, val_loader, test_loader = load_dataloaders_from_disk(save_dir, batch_size)
        return train_loader, val_loader, test_loader
        
        # batch_size = 32  # o il valore che preferisci
        # train_loader, val_loader, test_loader = load_dataloaders_from_disk("./saved_loaders_esc50", batch_size)

        # return hf_ds, hf_test_ds
    

    dataset = ESC50Dataset(audio_dir, meta_file)

    print(f"Dataset caricato: {len(dataset)} esempi totali.")
    print(f"Batch size: {batch_size}")
    
    # Usa i fold per la suddivisione: fold 1-4 -> train/val, fold 5 -> test
    train_val_df = dataset.df[dataset.df['fold'] != 5].reset_index(drop=True)
    test_df = dataset.df[dataset.df['fold'] == 5].reset_index(drop=True)
    
    print(f"Esempi Train+Val (fold 1-4): {len(train_val_df)}\n {train_val_df}")
    print(f"Esempi Test (fold 5): {len(test_df)}")
    

    # Crea direttamente i dataset senza duplicare
    train_val_dataset = ESC50Dataset(audio_dir, df=train_val_df)
    test_dataset = ESC50Dataset(audio_dir, df=test_df)


    # Split train/val
    indices = np.arange(len(train_val_dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=train_val_dataset.df["target"]
    )

    train_subset = torch.utils.data.Subset(train_val_dataset, train_idx)
    val_subset = torch.utils.data.Subset(train_val_dataset, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=4
    )

    # Log dimensioni
    print(f"Train loader: {len(train_loader.dataset)} examples")
    print(f"Validation loader: {len(val_loader.dataset)} examples")
    print(f"Test loader: {len(test_loader.dataset)} examples")

    # Log un batch di esempio
    try:
        train_batch = next(iter(train_loader))
        mel, mfcc, label = train_batch
        print(f"Batch di esempio (Train): mel shape={mel.shape}, mfcc shape={mfcc.shape}, label shape={label.shape}")
    except Exception as e:
        print(f"Errore nel caricare un batch di esempio: {e}")

    save_dataloaders(train_loader, val_loader, test_loader, save_dir=save_dir)

    return train_loader, val_loader, test_loader