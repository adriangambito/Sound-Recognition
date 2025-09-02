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
from librosa.effects import time_stretch
from skimage.transform import resize

# Parametri spettrogramma
N_MELS = 128
TARGET_SIZE = (128, 128)


class ESC50Dataset(Dataset):
    def __init__(self, audio_dir, meta_file=None, df=None, augment=False, samples_per_class=40):
        self.audio_dir = audio_dir
        self.augment = augment
        self.samples_per_class = samples_per_class
        
        if df is not None:
            self.df = df.reset_index(drop=True)
        elif meta_file is not None:
            self.df = pd.read_csv(meta_file)
        else:
            raise ValueError("Devi fornire almeno meta_file o df.")
        
        # Se samples_per_class è specificato, aumenta i samples
        if self.samples_per_class != 40:
            self.df = self._increase_samples_per_class(self.df, self.samples_per_class)
            print(f"Dataset espanso: {len(self.df)} samples totali ({self.samples_per_class} per classe)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row['filename']
        label_idx = row['target']
        audio_path = os.path.join(self.audio_dir, file_name)

        # Se augmentation è richiesta, applica le trasformazioni audio
        if self.augment:
            y, sr = librosa.load(audio_path, sr=44100)
            y = self._random_augment(y, sr)
            # Salva temporaneamente per elaborazione
            mel_spec = self._compute_mel_from_audio(y, sr)
            mfcc = self._compute_mfcc_from_audio(y, sr)
        else:
            # Usa le funzioni standard per estrarre features dal file
            mel_spec = extract_mel_spectrogram(audio_path)
            mfcc = extract_mfcc(audio_path, n_mfcc=40, target_shape=(40, 50))

        # Preprocessing comune
        mel_spec = prepare_spectrogram(mel_spec)
        mel_spec = (mel_spec - np.mean(mel_spec)) / np.std(mel_spec)
        mel_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)  # [1, 128, 128]

        mfcc_tensor = torch.tensor(mfcc.flatten(), dtype=torch.float32)  # [2000]

        return mel_tensor, mfcc_tensor, label_idx

    def _increase_samples_per_class(self, df, target_samples_per_class):
        """
        Aumenta il numero di samples per classe tramite oversampling.
        Replica i samples esistenti fino a raggiungere target_samples_per_class per ogni classe.
        """
        expanded_dfs = []
        
        for class_id in df['target'].unique():
            class_df = df[df['target'] == class_id].copy()
            current_samples = len(class_df)
            
            if current_samples >= target_samples_per_class:
                # Se abbiamo già abbastanza samples, prendi solo i primi N
                expanded_dfs.append(class_df.head(target_samples_per_class))
            else:
                # Calcola quante volte replicare + samples extra
                full_replicas = target_samples_per_class // current_samples
                extra_samples = target_samples_per_class % current_samples
                
                # Replica il dataframe
                replicated_dfs = [class_df] * full_replicas
                if extra_samples > 0:
                    replicated_dfs.append(class_df.head(extra_samples))
                
                # Aggiungi un suffisso per distinguere le repliche
                combined_df = pd.concat(replicated_dfs, ignore_index=True)
                for i in range(len(combined_df)):
                    if i >= current_samples:  # Solo per le repliche
                        replica_num = i // current_samples
                        original_filename = combined_df.loc[i, 'filename']
                        # Mantieni l'estensione ma aggiungi il suffisso
                        name, ext = os.path.splitext(original_filename)
                        combined_df.loc[i, 'filename'] = f"{name}_rep{replica_num}{ext}"
                        combined_df.loc[i, 'replica_id'] = replica_num
                        combined_df.loc[i, 'original_filename'] = original_filename
                
                expanded_dfs.append(combined_df)
            
            print(f"Classe {class_id}: {current_samples} -> {len(expanded_dfs[-1])} samples")
        
        return pd.concat(expanded_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    def _get_audio_path_and_augment_id(self, idx):
        """
        Restituisce il path dell'audio e l'ID di augmentation da applicare.
        Se è una replica, usa il file originale ma con augmentation diversa.
        """
        row = self.df.iloc[idx]
        
        # Se è una replica, usa il file originale
        if 'original_filename' in row and pd.notna(row['original_filename']):
            filename = row['original_filename']
            replica_id = int(row.get('replica_id', 0))
        else:
            filename = row['filename']
            replica_id = 0
            
        audio_path = os.path.join(self.audio_dir, filename)
        return audio_path, replica_id
    
    def _random_augment(self, y, sr):
        """Applica una trasformazione casuale al segnale audio"""
        aug_choice = np.random.choice(['noise', 'stretch', 'pitch', 'none'])
        if aug_choice == 'noise':
            return self._add_noise(y)
        elif aug_choice == 'stretch':
            rate = np.random.uniform(0.9, 1.1)
            return self._stretch_time(y, rate)
        elif aug_choice == 'pitch':
            steps = np.random.uniform(-2, 2)
            return self._pitch_shift(y, sr, steps)
        else:
            return y

    def _add_noise(self, y, noise_factor=0.005):
        noise = np.random.randn(len(y))
        return y + noise_factor * noise

    def _stretch_time(self, y, rate=1.1):
        return librosa.effects.time_stretch(y=y, rate=rate)

    def _pitch_shift(self, y, sr, n_steps=2):
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


    def _compute_mel_from_audio(self, y, sr, n_mels=N_MELS):
        """Calcola mel spectrogram da array audio"""
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        return S_DB

    def _compute_mfcc_from_audio(self, y, sr, n_mfcc=40, target_shape=(40, 50)):
        """Calcola MFCC da array audio"""
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_resized = resize(mfcc, target_shape, mode='constant', anti_aliasing=True)
        mfcc_normalized = (mfcc_resized - mfcc_resized.mean()) / (mfcc_resized.std() + 1e-9)
        return mfcc_normalized


# Manteniamo ESC50DatasetAug per retrocompatibilità
class ESC50DatasetAug(ESC50Dataset):
    def __init__(self, audio_dir, df, augment=False, samples_per_class=None):
        super().__init__(audio_dir, df=df, augment=augment, samples_per_class=samples_per_class)


def save_dataloaders(train_loader, val_loader, test_loader, save_dir):
    """Salva i dataloader su disco"""
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
    """Carica i dataloader da disco"""
    def load_loader(name):
        mel_tensor, mfcc_tensor, label_tensor = torch.load(os.path.join(save_dir, f"{name}.pt"))
        dataset = torch.utils.data.TensorDataset(mel_tensor, mfcc_tensor, label_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=(name=="trainloader")
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
    y, sr = librosa.load(file_path, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_resized = resize(mfcc, target_shape, mode='constant', anti_aliasing=True)
    mfcc_normalized = (mfcc_resized - mfcc_resized.mean()) / (mfcc_resized.std() + 1e-9)
    return mfcc_normalized


def custom_collate_fn(batch):
    """Custom collate function per i dataloader"""
    mel, mfcc, label = zip(*batch)
    return (
        torch.stack(mel),
        torch.stack(mfcc),
        torch.tensor(label)
    )


def prepare_esc50_loaders(audio_dir, meta_file, batch_size, data_augmentation=False, samples_per_class=None):
    """
    Prepara i dataloader per ESC50, con o senza data augmentation.
    Gestisce automaticamente il caching su disco.
    
    Args:
        audio_dir: Directory contenente i file audio
        meta_file: File CSV con i metadata
        batch_size: Dimensione del batch
        data_augmentation: Se applicare data augmentation al training set
        samples_per_class: Numero target di samples per classe (None = usa originali)
    """
    # Determina la directory di salvataggio in base ai parametri
    dir_suffix = ""
    if data_augmentation:
        dir_suffix += "_augmented"
    if samples_per_class is not None:
        dir_suffix += f"_samples{samples_per_class}"
    
    save_dir = Path(f"Datasets/ESC50/Dataloaders{dir_suffix}")
    
    # Path dei file salvati
    trainloader_path = save_dir / "trainloader.pt"
    valloader_path = save_dir / "valloader.pt"
    testloader_path = save_dir / "testloader.pt"

    # Se esistono già, carica da disco
    if trainloader_path.exists() and valloader_path.exists() and testloader_path.exists():
        config_msg = f"batch_size={batch_size}"
        if data_augmentation:
            config_msg += ", augmented"
        if samples_per_class:
            config_msg += f", {samples_per_class} samples/class"
        print(f"Dataloaders trovati ({config_msg}), caricamento da disco...")
        return load_dataloaders_from_disk(save_dir, batch_size)

    # Altrimenti crea nuovi dataloader
    config_msg = ""
    if data_augmentation:
        config_msg += "con augmentations"
    if samples_per_class:
        if config_msg:
            config_msg += " e "
        config_msg += f"{samples_per_class} samples per classe"
    if not config_msg:
        config_msg = "configurazione standard"
        
    print(f"Creazione dataset {config_msg}...")

    # Carica metadata
    meta_df = pd.read_csv(meta_file)
    print(f"Dataset originale: {len(meta_df)} esempi totali.")
    print(f"Batch size: {batch_size}")

    # Split usando i fold
    train_val_df = meta_df[meta_df['fold'] != 5].reset_index(drop=True)
    test_df = meta_df[meta_df['fold'] == 5].reset_index(drop=True)

    print(f"Esempi Train+Val originali (fold 1-4): {len(train_val_df)}")
    print(f"Esempi Test originali (fold 5): {len(test_df)}")

    # Split stratificato train/val
    indices = np.arange(len(train_val_df))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=train_val_df["target"]
    )

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    # Crea dataset con parametri specificati
    train_dataset = ESC50Dataset(audio_dir, df=train_df, augment=data_augmentation, samples_per_class=samples_per_class)
    val_dataset = ESC50Dataset(audio_dir, df=val_df, augment=False)  # Val sempre senza augment/oversampling
    test_dataset = ESC50Dataset(audio_dir, df=test_df, augment=False)  # Test sempre senza augment/oversampling

    # Crea dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=custom_collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=custom_collate_fn, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=custom_collate_fn, num_workers=4
    )

    # Log dimensioni finali
    print(f"\nDimensioni finali:")
    print(f"Train loader: {len(train_loader.dataset)} examples")
    print(f"Validation loader: {len(val_loader.dataset)} examples")
    print(f"Test loader: {len(test_loader.dataset)} examples")

    # Log batch di esempio
    try:
        train_batch = next(iter(train_loader))
        mel, mfcc, label = train_batch
        print(f"Batch di esempio (Train): mel shape={mel.shape}, mfcc shape={mfcc.shape}, label shape={label.shape}")
    except Exception as e:
        print(f"Errore nel caricare un batch di esempio: {e}")

    # Salva i dataloader
    save_dataloaders(train_loader, val_loader, test_loader, save_dir=save_dir)

    return train_loader, val_loader, test_loader