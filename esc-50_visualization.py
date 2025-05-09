import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('./Datasets/ESC50/'))
from Datasets.ESC50.utils import ESC50

train_splits = [1,2,3,4]
test_split = 5

shared_params = {'csv_path': './Datasets/ESC50/esc50.csv',
                 'wav_dir': './Datasets/ESC50/audio/audio',
                 'dest_dir': './Datasets/ESC50/audio/audio/16000',
                 'audio_rate': 16000,
                 'only_ESC10': True,
                 'pad': 0,
                 'normalize': True}

train_gen = ESC50(folds=train_splits,
                  randomize=True,
                  strongAugment=True,
                  random_crop=True,
                  inputLength=2,
                  mix=True,
                  **shared_params).batch_gen(16)

test_gen = ESC50(folds=[test_split],
                 randomize=False,
                 strongAugment=False,
                 random_crop=False,
                 inputLength=4,
                 mix=False,
                 **shared_params).batch_gen(16)

X, Y = next(train_gen)
print(f"X.shape: {X.shape}, Y.shape: {Y.shape}")

df = pd.read_csv('./Datasets/ESC50/esc50.csv')
# Estrai target e categoria come array NumPy
classes = df[['target', 'category']].to_numpy().tolist()
# Costruisci un set di stringhe uniche (target + category)
classes = set(f"{c[0]} {c[1]}" for c in classes)
# Riconverti in array 2D separando di nuovo target e category
classes = np.array([c.split(' ', 1) for c in classes])
# Crea un dizionario: {target: category}
classes = {int(k): v for k, v in classes}
print(classes)