# Sound-Recognition: A CNN Model for for ESC-50 Dataset

**A Python application with a graphical user interface (GUI) that allows you to configure parameters and train a Convolutional Neural Network (CNN) on the ESC-50 dataset. Designed to make the training process easy to manage, even for users without coding experience.**

**Main Features:**
*	Automatic loading of the ESC-50 dataset and preparation of DataLoaders.
*	Set hyperparameters (learning rate, batch size, epochs) directly from the GUI.
*	Start and manage training with detailed logs of the progress.
*	Real-time graphs of Loss and Accuracy (via Matplotlib).
*	Automatic evaluation on the test set when training is complete.
*	A button to force stop training and quickly exit the app.
*	Function to export the requirements.txt file.


# Repository structure
# Dataset Structure
The project uses the ESC50 dataset with the following structure:
```
Datasets/
└── ESC50/
    ├── audio/
    │   └── audio/
    └── meta/
    └── esc50.csv
```

This structure includes the audio files in the `audio/` directory and the metadata in the `meta/` directory.

## Project Files

The project contains the following Python files:

- `soundrec_gui.py`: Main GUI application for the training interface.
- `ESC50Dataset.py`: Handles dataset loading and preprocessing.
- `train.py`: Contains the model training logic.
- `esc-50_visualization.py`: Script for visualizing the ESC50 dataset.
- `requirements.txt`: Lists the required dependencies for the project.


# Getting Started

**Clone the repository**
```
https://github.com/adriangambito/Sound-Recognition.git
```

# Install the dependencies
```
pip install -r requirements.txt
```

# Launch the application
```
python soundrec_gui.py
```

The graphical user interface of the application will open.

# Set the initial hyperparameters
* Learning rate
* Batch size
* Epochs

# Press the Load Dataset button
Pressing the the button the ESC50 Dataset will be loaded and preprocessed. 
The audio files from the ESC50 dataset are loaded and processed to extract their Mel spectrograms. Each spectrogram is converted into a 128x128 image representation, which serves as the input for the Convolutional Neural Network (CNN). This transformation allows the model to learn from the time-frequency features of the audio clips in a format suitable for image-based deep learning architectures.

# Press the Start training button
Once the button is pressed, the training process will start. The values of Accuracy and Loss will be displayed dynamically on the graphs, updating in real-time during the execution of each epoch. This allows the user to visually monitor the performance of the model as it trains.

# Press the Stop training button
When the “Stop Training” button is clicked, the system stop the training process and closes the application. Specifically, the stop command does not take effect immediately: the current epoch is allowed to finish before the process is terminated. Once the epoch completes, the application automatically closes the GUI.


# CNN Model
The basic CNN model used for the example training:
```
# CNN model for Sound Recognition
class SoundCNN(nn.Module):
    """CNN Model for ESC50 Dataset."""
    def __init__(self):
        super(SoundCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 50)  # 50 classi per ESC-50
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
```



