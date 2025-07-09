import os
import sys
import threading
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit,
    QFileDialog, QTextEdit, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QGridLayout,
    QComboBox
)
from PySide6.QtCore import Qt, Signal


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from train import train_epoch, validate
from ESC50Dataset import prepare_esc50_loaders
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim


from models import SoundCNN, ResNet18

# Dummy training logic to simulate AI training
import time
import random

def run_train(gui_ref):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gui_ref.log(f"Training on: {device}")

    try:
        lr = float(gui_ref.learning_rate_input.text())
        batch_size = int(gui_ref.batch_size_input.text())
        epochs = int(gui_ref.epochs_input.text())
        optimizer_name = gui_ref.optimizer_combo.currentText()
        dropout = float(gui_ref.dropout.text())
    except ValueError:
        gui_ref.log("Invalid hyperparameters.")
        return
    
    if not gui_ref.dataset_dir or not gui_ref.meta_file:
        gui_ref.log("Dataset or meta file not loaded.")
        return

    model = SoundCNN(dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        print("Error, strategy not available")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Ottieni i DataLoader dalla GUI
    train_loader = gui_ref.train_loader
    val_loader = gui_ref.val_loader
    test_loader = gui_ref.test_loader

    if train_loader is None or val_loader is None or test_loader is None:
        gui_ref.log("DataLoaders not initialized. Please load the dataset first.")
        return

    
    gui_ref.train_losses.clear()
    gui_ref.train_accuracies.clear()
    gui_ref.vall_losses.clear()
    gui_ref.vall_accuracies.clear()


    best_val_acc = 0.0

    for epoch in range(epochs):
        if gui_ref.stop_requested:
            gui_ref.log(f"Training stopped at epoch {epoch+1}.")
            break
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, epochs)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        gui_ref.log(f"Epoch {epoch+1}/{epochs}")
        gui_ref.log(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        gui_ref.log(f"  Val Loss:   {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        # Update plot
        gui_ref.update_signal.emit(epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc)

        # if save_best_path and val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     torch.save(model.state_dict(), save_best_path)
        #     gui_ref.log(f"Saved best model with val accuracy {val_acc:.2f}%")

    gui_ref.log("Training completed.")

    # Valutazione finale sul test set
    gui_ref.log("Final test of the model.")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    gui_ref.log(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")




class TrainingGUI(QWidget):

    update_signal = Signal(int, int, float, float, float, float)  # epoch, loss, acc

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Model Trainer - Deep Convolutional Network")
        self.setGeometry(100, 100, 800, 600)

        self.training_thread = None

        self.dataset_dir = ""
        self.meta_file = ""

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.stop_requested = False

        self.init_ui()
        self.update_signal.connect(self.update_plot)

    
    
    def init_ui(self):
        layout = QVBoxLayout()

        # Hyperparameters input section
        hyper_params_group = QGroupBox("Hyper Parameters")
        grid_layout = QGridLayout()

        self.learning_rate_input = QLineEdit("0.001")
        self.batch_size_input = QLineEdit("32")
        self.epochs_input = QLineEdit("10")
        self.dropout = QLineEdit("0.5")
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["AdamW", "SGD", "Adam"])

        # Aggiungi etichette e campi sulla griglia (2 colonne)
        grid_layout.addWidget(QLabel("Learning Rate:"), 0, 0)
        grid_layout.addWidget(self.learning_rate_input, 0, 1)

        grid_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        grid_layout.addWidget(self.batch_size_input, 1, 1)

        grid_layout.addWidget(QLabel("Optimizer:"), 2, 0)
        grid_layout.addWidget(self.optimizer_combo, 2, 1)

        grid_layout.addWidget(QLabel("Epochs:"), 0, 2)
        grid_layout.addWidget(self.epochs_input, 0, 3)

        grid_layout.addWidget(QLabel("Droput"), 1, 2)
        grid_layout.addWidget(self.dropout, 1, 3)

        hyper_params_group.setLayout(grid_layout)
        layout.addWidget(hyper_params_group)

        # Command buttons
        #self.save_dataset_btn = QPushButton("Save Dataset")
        #self.load_params_btn = QPushButton("Load Network Params")
        #self.save_params_btn = QPushButton("Save Network Params")

        self.load_dataset_btn = QPushButton("Load Dataset")
        self.start_btn = QPushButton("Start Training")
        self.stop_btn = QPushButton("Stop Training")

        
        #button_layout.addWidget(self.save_dataset_btn)
        #button_layout.addWidget(self.load_params_btn)
        #button_layout.addWidget(self.save_params_btn)


        # Training evolution display
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax_loss = self.figure.add_subplot(211)  # primo grafico sopra
        self.ax_accuracy = self.figure.add_subplot(212)  # secondo grafico sotto

        self.train_losses = []
        self.train_accuracies = []
        self.vall_losses = []
        self.vall_accuracies = []

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        # Connect buttons
        self.load_dataset_btn.clicked.connect(self.load_dataset)
        #self.save_dataset_btn.clicked.connect(self.save_dataset)
        #self.load_params_btn.clicked.connect(self.load_params)
        #self.save_params_btn.clicked.connect(self.save_params)
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)


        # Dataset buttons layout
        dataset_button_layout = QHBoxLayout()
        dataset_button_layout.addWidget(self.load_dataset_btn)
        dataset_button_layout.addStretch(1)

        # Control buttons layout
        # Layout for training control buttons aligned bottom-right
        control_button_layout = QHBoxLayout()
        control_button_layout.addStretch(1)
        control_button_layout.addWidget(self.start_btn)
        control_button_layout.addWidget(self.stop_btn)

        # Assemble the layout
        #layout.addLayout(form_layout)
        layout.addLayout(dataset_button_layout)
        layout.addWidget(self.canvas)
        layout.addWidget(QLabel("Console Log:"))
        layout.addWidget(self.log_output)
        layout.addLayout(control_button_layout)

        self.setLayout(layout)



    def load_dataset(self):
        """Function to extract the ESC50 Dataset."""

        # Imposta i path automatici relativi al file corrente
        base_dir = os.path.dirname(os.path.abspath(__file__))
        audio_path = os.path.join(base_dir, "Datasets", "ESC50", "audio/audio")
        meta_path = os.path.join(base_dir, "Datasets", "ESC50", "meta", "esc50.csv")

        # Verify Dataset Directory exists
        if os.path.isdir(audio_path) and os.path.isfile(meta_path):
            self.dataset_dir = audio_path
            self.meta_file = meta_path
            self.log(f"Loaded ESC-50 dataset:\n  Audio: {audio_path}\n  Meta: {meta_path}")

            # Leggi il batch size dalla GUI
            try:
                batch_size = int(self.batch_size_input.text())
            except ValueError:
                self.log("Invalid batch size; using default batch size of 32.")
                batch_size = 32

            # Preprocessing: chiama la funzione per preparare i DataLoader
            try:
                self.train_loader, self.val_loader, self.test_loader = prepare_esc50_loaders(
                    self.dataset_dir, self.meta_file, batch_size
                )
                self.log(f"Dataset preprocessing completed.")
            except Exception as e:
                self.log(f"Error during dataset preprocessing: {e}")

        else:
            self.log("ESC-50 dataset not found. Check the directory structure.")

            

    # def save_dataset(self):
    #     file_name, _ = QFileDialog.getSaveFileName(self, "Save Dataset")
    #     if file_name:
    #         self.log(f"Dataset saved to: {file_name}")

    # def load_params(self):
    #     file_name, _ = QFileDialog.getOpenFileName(self, "Open Network Parameters")
    #     if file_name:
    #         self.log(f"Network parameters loaded from: {file_name}")

    # def save_params(self):
    #     file_name, _ = QFileDialog.getSaveFileName(self, "Save Network Parameters")
    #     if file_name:
    #         self.log(f"Network parameters saved to: {file_name}")

    def start_training(self):
        if self.training_thread and self.training_thread.is_alive():
            self.log("Training already running!")
            return

        self.stop_requested = False  # Reset stop flag

        self.ax_loss.cla()
        self.ax_accuracy.cla()

        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        #self.ax_loss.set_xlim(1, 30)
        self.ax_accuracy.set_xlabel("Epoch")
        self.ax_accuracy.set_ylabel("Accuracy (%)")
        #self.ax_accuracy.set_xlim(1, 30)
        self.ax_accuracy.set_ylim(0, 100)  # Fissa la scala da 0 a 100


        self.training_thread = threading.Thread(target=run_train, args=(self,))
        self.training_thread.start()


    def stop_training(self):
        self.log("Stopping training and closing the application.")
        self.stop_requested = True

        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join()  # Aspetta la chiusura del thread
            self.log("Training stopped.")

        QApplication.quit()
        #sys.exit(0)

    def update_plot(self, epoch, max_epochs, train_loss, train_acc, vall_loss, vall_acc):
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.vall_losses.append(vall_loss)
        self.vall_accuracies.append(vall_acc)

        # Aggiorna grafico loss
        self.ax_loss.cla()
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.set_xlim(1, max_epochs)
        self.ax_loss.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Train Loss', color='red')
        self.ax_loss.plot(range(1, len(self.vall_losses) + 1), self.vall_losses, label='Validation Loss', color='blue')
        self.ax_loss.legend()

        # Aggiorna grafico accuracy
        self.ax_accuracy.cla()
        self.ax_accuracy.set_xlabel("Epoch")
        self.ax_accuracy.set_ylabel("Accuracy (%)")
        self.ax_accuracy.set_xlim(1, max_epochs)
        self.ax_accuracy.set_ylim(0, 100)  # Sempre da 0 a 100
        self.ax_accuracy.plot(range(1, len(self.train_accuracies) + 1), self.train_accuracies, label='Train Accuracy', color='red')
        self.ax_accuracy.plot(range(1, len(self.vall_accuracies) + 1), self.vall_accuracies, label='Validation Accuracy', color='blue')
        self.ax_accuracy.legend()

        self.canvas.draw_idle()

    def log(self, message):
        self.log_output.append(message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = TrainingGUI()
    gui.show()
    sys.exit(app.exec())
