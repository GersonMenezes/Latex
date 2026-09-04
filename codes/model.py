import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # ----------------------------------------------------
        # ENCODER: Compresses the signal and extracts features
        # Input shape: (Batch, 1, 2000)
        # ----------------------------------------------------
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), 
            # Shape after MaxPool: (Batch, 16, 1000)
            
            # Layer 2
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  
            # Shape after MaxPool: (Batch, 32, 500) -> The "Latent Space"
        )
        
        # ----------------------------------------------------
        # DECODER: Reconstructs the signal from the latent space
        # Input shape: (Batch, 32, 500)
        # ----------------------------------------------------
        self.decoder = nn.Sequential(
            # Upsample 1
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            
            # Upsample 2
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            
            # Saída Final
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=5, padding=2)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded