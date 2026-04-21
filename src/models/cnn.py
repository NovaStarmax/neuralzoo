import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=6, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            # Bloc 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # 3 canaux d’entrée, le filtre doit être de 3x3x3 + padding 1 pour conserver la taille 32 filtres de sorties
            nn.BatchNorm2d(32), # Normalisation pour accélérer l’entraînement et stabiliser les gradients
            nn.ReLU(),
            nn.MaxPool2d(2),          # 32x32 → 16x16
            nn.Dropout2d(0.2),
            # Bloc 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 32 Features d’entrée, 64 de sortie
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 16x16 → 8x8
            nn.Dropout2d(0.3),
            # Bloc 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 8x8 → 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), # Aplatit les 128 canaux de 4x4 en un vecteur de 2048 éléments
            nn.Linear(128 * 4 * 4, 256), # Couche entièrement connectée de 2048 à 256 neurones  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))