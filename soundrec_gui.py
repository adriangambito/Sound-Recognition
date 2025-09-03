import datetime
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

from train import train_epoch, validate, validate, EarlyStopping
from ESC50Dataset import prepare_esc50_loaders
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim


from models import SoundCNN, ResNet18, SoundCNN_Variable, SoundCNN_MFCCConcat

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
        input_size = int(gui_ref.input_size.text())
        kernel_size = int(gui_ref.kernel_size.text())
        stride = int(gui_ref.stride.text())
        n_blocks = int(gui_ref.n_blocks.text())
        
    except ValueError:
        gui_ref.log("Invalid hyperparameters.")
        return
    
    # if not gui_ref.dataset_dir or not gui_ref.meta_file:
    #     gui_ref.log("Dataset or meta file not loaded.")
    #     return

    #model = SoundCNN(dropout).to(device)
    if gui_ref._loaded_model == True:
        selected_model = gui_ref.model_selector.currentText()

        gui_ref.log(f"Selected model: {selected_model}")
        checkpoint_path = os.path.join("Checkpoints", selected_model)
        if not os.path.exists(checkpoint_path):
            gui_ref.log(f"Selected checkpoint {selected_model} not found.")
            return
        
        
        # Esempio di utilizzo:
        #gui_ref.inspect_checkpoint(checkpoint_path)
        # Carica gli hyper-parameters salvati accanto al modello
        try:
            hyperparams = gui_ref.parse_hyperparams_from_filename(checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
           
            #gui_ref.log(f"Checkpoint: {checkpoint}")
        except Exception as e:
            gui_ref.log(f"Error: {str(e)}")
            return
        

        # Inizializza il modello con i parametri recuperati dal disco
        # model = SoundCNN_Variable(
        #     input_size=hyperparams["input_size"],
        #     kernel_size=hyperparams["kernel_size"],
        #     stride=hyperparams["stride"],
        #     dropout=hyperparams["dropout"],
        #     n_blocks=hyperparams["n_blocks"]
        # ).to(device)
        model = SoundCNN_MFCCConcat(
            input_size=hyperparams["input_size"],
            kernel_size=hyperparams["kernel_size"],
            stride=hyperparams["stride"],
            dropout=hyperparams["dropout"],
            n_blocks=hyperparams["n_blocks"],
            mfcc_feature_size=2000
        ).to(device)

        # Carica i pesi
        model.load_state_dict(checkpoint)
        gui_ref.log(f"✅ Loaded model and hyper-parameters from {checkpoint_path}")
    else:
        model = SoundCNN_MFCCConcat(input_size=input_size, kernel_size=kernel_size, stride=stride, dropout=dropout, n_blocks=n_blocks, mfcc_feature_size=2000).to(device)

    criterion = nn.CrossEntropyLoss()
    # Optimizer
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
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
    early_stopping = EarlyStopping(patience=5)  # Add patience as needed
    best_model_state = None

    for epoch in range(epochs):
        if gui_ref.stop_requested:
            gui_ref.log(f"Training stopped at epoch {epoch+1}.")
            break

        # Train and validate
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, epochs)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Log progress
        gui_ref.log(f"Epoch {epoch+1}/{epochs}")
        gui_ref.log(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        gui_ref.log(f"  Val Loss:   {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        # Update plot
        gui_ref.update_signal.emit(epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc)

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            gui_ref.log(f"Early stopping triggered at epoch {epoch+1}")
            break

        # Save best model based on validation loss
        if best_model_state is None or val_loss < early_stopping.best_loss:
            best_model_state = model.state_dict()
            best_val_acc = val_acc  # Optional, if you also want to track best accuracy
            # Optionally save to disk:
            # torch.save(best_model_state, save_best_path)
            # gui_ref.log(f"Saved best model at epoch {epoch+1} with val accuracy {val_acc:.2f}%")

    # Load best model before finishing
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    gui_ref._trained_model = model
    gui_ref.check_save_model = True
    gui_ref.log("Training completed.")

    # Final test evaluation
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
        self.check_save_model = False
        self._trained_model = None
        self._loaded_model = False
        self.samples_per_class = None

        self.init_ui()
        self.update_signal.connect(self.update_plot)

    
    


    def init_ui(self):
        layout = QVBoxLayout()

        # ------------------------- Hyper Parameters -------------------------
        hyper_params_group = QGroupBox("Hyper Parameters")
        grid_layout = QGridLayout()

        self.learning_rate_input = QLineEdit("0.001")
        self.batch_size_input = QLineEdit("64")
        self.epochs_input = QLineEdit("30")
        self.dropout = QLineEdit("0.2")
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["AdamW", "SGD", "Adam"])
        self.input_size = QLineEdit("128")
        self.kernel_size = QLineEdit("7")
        self.stride = QLineEdit("2")
        self.n_blocks = QLineEdit("3")
        self.samples_per_class = QLineEdit("40")  # New field for samples per class

        grid_layout.addWidget(QLabel("Learning Rate:"), 0, 0)
        grid_layout.addWidget(self.learning_rate_input, 0, 1)
        grid_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        grid_layout.addWidget(self.batch_size_input, 1, 1)
        grid_layout.addWidget(QLabel("Optimizer:"), 2, 0)
        grid_layout.addWidget(self.optimizer_combo, 2, 1)
        grid_layout.addWidget(QLabel("Max Epochs:"), 3, 0)
        grid_layout.addWidget(self.epochs_input, 3, 1)
        grid_layout.addWidget(QLabel("Dropout:"), 4, 0)
        grid_layout.addWidget(self.dropout, 4, 1)

        grid_layout.addWidget(QLabel("Input Size:"), 0, 2)
        grid_layout.addWidget(self.input_size, 0, 3)
        grid_layout.addWidget(QLabel("Kernel Size:"), 1, 2)
        grid_layout.addWidget(self.kernel_size, 1, 3)
        grid_layout.addWidget(QLabel("Stride:"), 2, 2)
        grid_layout.addWidget(self.stride, 2, 3)
        grid_layout.addWidget(QLabel("CNN Blocks:"), 3, 2)
        grid_layout.addWidget(self.n_blocks, 3, 3)        
        grid_layout.addWidget(QLabel("Samples per Class:"), 4, 2)
        grid_layout.addWidget(self.samples_per_class, 4, 3)

        hyper_params_group.setLayout(grid_layout)
        layout.addWidget(hyper_params_group)

        # ------------------------- Buttons -------------------------
        self.load_dataset_btn = QPushButton("Load Dataset")
        self.load_model_btn = QPushButton("Load model")
        self.save_model_btn = QPushButton("Save model")
        self.test_model_btn = QPushButton("Test model")
        self.start_btn = QPushButton("Start Training")
        self.stop_btn = QPushButton("Stop Training")
        self.reset_btn = QPushButton("Reset")

        # ------------------------- Model Selector -------------------------
        self.model_selector = QComboBox()
        self.populate_model_selector()

        # ------------------------- Plots -------------------------
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax_loss = self.figure.add_subplot(211)
        self.ax_accuracy = self.figure.add_subplot(212)

        self.train_losses = []
        self.train_accuracies = []
        self.vall_losses = []
        self.vall_accuracies = []

        # ------------------------- Log Console -------------------------
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        # ------------------------- Connections -------------------------
        self.load_dataset_btn.clicked.connect(self.load_dataset)
        self.load_model_btn.clicked.connect(self.load_model)
        self.save_model_btn.clicked.connect(self.save_model)
        self.test_model_btn.clicked.connect(self.test_model)
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)
        self.reset_btn.clicked.connect(self.reset)


        # ------------------------- Layout Assembly -------------------------

        # Dataset buttons
        dataset_button_layout = QHBoxLayout()
        dataset_button_layout.addWidget(self.load_dataset_btn)
        dataset_button_layout.addStretch(1)

        # Model buttons + dropdown
        model_button_layout = QHBoxLayout()
        model_button_layout.addWidget(self.load_model_btn)
        model_button_layout.addWidget(self.save_model_btn)
        model_button_layout.addWidget(self.model_selector)
        model_button_layout.addWidget(self.reset_btn)
        model_button_layout.addStretch(1)

        # Control buttons
        control_button_layout = QHBoxLayout()
        control_button_layout.addStretch(1)
        control_button_layout.addWidget(self.test_model_btn)
        control_button_layout.addWidget(self.start_btn)
        control_button_layout.addWidget(self.stop_btn)

        # Assemble
        layout.addLayout(dataset_button_layout)
        layout.addLayout(model_button_layout)
        layout.addWidget(self.canvas)
        layout.addWidget(QLabel("Console Log:"))
        layout.addWidget(self.log_output)
        layout.addLayout(control_button_layout)

        self.setLayout(layout)

    # ------------------------- Populate Model Selector -------------------------
    def populate_model_selector(self):
        self.model_selector.clear()
        checkpoints_dir = "Checkpoints"

        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        files = [f for f in os.listdir(checkpoints_dir) if os.path.isfile(os.path.join(checkpoints_dir, f))]
        if files:
            self.model_selector.addItems(files)
        else:
            self.model_selector.addItem("No checkpoints found")



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
                    self.dataset_dir, self.meta_file, batch_size, data_augmentation=True, samples_per_class=int(self.samples_per_class.text())
                )
                self.log(f"Dataset preprocessing completed.")
            except Exception as e:
                self.log(f"Error during dataset preprocessing: {e}")

        else:
            self.log("ESC-50 dataset not found. Check the directory structure.")

            

    def load_model(self):
        self._loaded_model = True
        self.log(f"Status: self._loaded_model -> {self._loaded_model}")
        # file_name, _ = QFileDialog.getSaveFileName(self, "Save Dataset")
        # if file_name:
        #     self.log(f"Dataset saved to: {file_name}")

    # def load_params(self):
    #     file_name, _ = QFileDialog.getOpenFileName(self, "Open Network Parameters")
    #     if file_name:
    #         self.log(f"Network parameters loaded from: {file_name}")

    def save_model(self):
        if self.check_save_model == True:
            self.save_checkpoint()
            self.log("Model saved")
        else:
            self.log("There isn't a model to save")


    
    def test_model(self):
        if self._loaded_model == True:
            selected_model = self.model_selector.currentText()
            self.log(f"Selected model: {selected_model}")
            checkpoint_path = os.path.join("Checkpoints", selected_model)
            if not os.path.exists(checkpoint_path):
                self.log(f"Selected checkpoint {selected_model} not found.")
                return
            
            try:
                hyperparams = self.parse_hyperparams_from_filename(checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
            
                #gui_ref.log(f"Checkpoint: {checkpoint}")
            except Exception as e:
                self.log(f"Error: {str(e)}")
                return

            # Inizializza il modello con i parametri recuperati dal disco
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = SoundCNN_Variable(
                input_size=hyperparams["input_size"],
                kernel_size=hyperparams["kernel_size"],
                stride=hyperparams["stride"],
                dropout=hyperparams["dropout"],
                n_blocks=hyperparams["n_blocks"]
            ).to(device)

            # Carica i pesi
            model.load_state_dict(checkpoint)
            self.log(f"✅ Loaded model and hyper-parameters from {checkpoint_path}")

            if self.test_loader is None:
                self.log("DataLoaders not initialized. Please load the dataset first.")
                return
            
            criterion = nn.CrossEntropyLoss()

            self.log("Starting evaluation model...")
            
            test_loss, test_acc = validate(model, self.test_loader, criterion, device)
            self.log(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

        else:
            self.log(f"The model is not loaded.")


    
    def generate_checkpoint_filename(self):
        # Costruisci il nome base
        filename = (
            f"CNN_{self.learning_rate_input.text()}_{self.batch_size_input.text()}_{self.optimizer_combo.currentText()}_"
            f"{self.epochs_input.text()}_{self.dropout.text()}_{self.input_size.text()}_"
            f"{self.kernel_size.text()}_{self.stride.text()}_{self.n_blocks.text()}.pt"
        )

        checkpoints_dir = "Checkpoints"
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        filepath = os.path.join(checkpoints_dir, filename)

        # Se il file esiste già, aggiungi timestamp
        if os.path.exists(filepath):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filename.replace(".pt", f"_{timestamp}.pt")
            filepath = os.path.join(checkpoints_dir, filename)

        return filepath
    

    def inspect_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict):
                self.log("✅ Checkpoint loaded successfully.")

                has_model_state = "model_state_dict" in checkpoint
                has_hyperparams = "hyperparameters" in checkpoint

                self.log(f"- Contains 'model_state_dict': {has_model_state}")
                self.log(f"- Contains 'hyperparameters': {has_hyperparams}")

                if has_hyperparams:
                    self.log("Hyperparameters contained:")
                    for k, v in checkpoint["hyperparameters"].items():
                        self.log(f"  {k}: {v}")

            else:
                self.log("❌ Checkpoint is not a dictionary. It might be a raw state_dict or model object.")

        except Exception as e:
            self.log(f"❌ Error loading checkpoint: {e}")


    def parse_hyperparams_from_filename(self, filename):
        """
        Estrae gli hyperparameters dal nome del file checkpoint.

        Assumiamo che il formato sia:
        CNN_<lr>_<batch_size>_<optimizer>_<epochs>_<dropout>_<input_size>_<kernel_size>_<stride>_<n_blocks>.pt
        """
        basename = os.path.basename(filename)
        name, _ = os.path.splitext(basename)
        parts = name.split("_")
        
        if len(parts) != 10 or parts[0] != "CNN":
            raise ValueError(f"Invalid checkpoint filename format: {filename}")
        
        hyperparams = {
            "learning_rate": float(parts[1]),
            "batch_size": int(parts[2]),
            "optimizer": parts[3],
            "epochs": int(parts[4]),
            "dropout": float(parts[5]),
            "input_size": int(parts[6]),
            "kernel_size": int(parts[7]),
            "stride": int(parts[8]),
            "n_blocks": int(parts[9])
        }
        return hyperparams

    


    def save_checkpoint(self):
        filepath = self.generate_checkpoint_filename()
        self.log(f"Model checkpoint name: {filepath}")
        torch.save(self._trained_model.state_dict(), filepath)
        
        print(f"Modello salvato in {filepath}")


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


    def reset(self):
        self.log("All settings are restored.")
        self.check_save_model = False
        self._trained_model = None
        self._loaded_model = False
        self.stop_requested = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = TrainingGUI()
    gui.show()
    sys.exit(app.exec())
