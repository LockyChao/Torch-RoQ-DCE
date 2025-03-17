import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# PyTorch Model
# =============================================================================
class DCEModelFC(nn.Module):
    def __init__(self, in_dim, out_dim, layer_num):
        """
        Args:
            in_dim: length of each input signal (Tissue and AIF). 
                    The concatenated input dimension is 2*in_dim.
            out_dim: dimension of the output (e.g. 3).
            layer_num: number of dense layers in the “deconv” part.
        """
        super(DCEModelFC, self).__init__()
        # First dense layer: from concatenated (2*in_dim) to in_dim.
        layers = [nn.Linear(in_dim * 2, in_dim),
                  nn.ReLU(),
                  nn.LayerNorm(in_dim)]
        # Additional (layer_num-1) layers keep the in_dim.
        for _ in range(layer_num - 1):
            layers += [nn.Linear(in_dim, in_dim),
                       nn.ReLU(),
                       nn.LayerNorm(in_dim)]
        self.feature_extractor = nn.Sequential(*layers)
        # Prediction layers: two hidden layers then final output.
        self.pred_layers = nn.Sequential(
            nn.Linear(in_dim, 160),
            nn.ReLU(),
            nn.Linear(160, 96),
            nn.ReLU(),
            nn.Linear(96, out_dim),
            nn.Sigmoid()
        )

    def forward(self, tissue, aif):
        # tissue and aif are expected to be (batch, dce_dim)
        # Expand and concatenate along feature dimension.
        x = torch.cat([tissue.unsqueeze(-1), aif.unsqueeze(-1)], dim=-1)  # shape (B, dce_dim, 2)
        # Flatten last two dimensions: shape (B, dce_dim*2)
        x = x.view(x.size(0), -1)
        features = self.feature_extractor(x)
        out = self.pred_layers(features)
        return out


class DCEModelCNN(nn.Module):
    def __init__(self, dce_dim, out_dim):
        super(DCEModelCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        # Global average pooling over the temporal dimension
        self.fc = nn.Linear(64, out_dim)

    def forward(self, tissue, aif):
        # tissue and aif: shape (batch, dce_dim)
        # Stack them along a new channel dimension to get shape: (batch, 2, dce_dim)
        x = torch.stack([tissue, aif], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        # Global average pooling (averaging over the time dimension)
        x = torch.mean(x, dim=2)
        out = self.fc(x)
        return out

class DCEModelTransformer(nn.Module):
    def __init__(self, dce_dim, out_dim, num_layers=2, d_model=64, nhead=8,  dim_feedforward=128, dropout=0.1):
        """
        Args:
            dce_dim: Length of the input sequence (number of time points).
            out_dim: Number of output parameters (4 in this case).
            d_model: Dimension of the embedding.
            nhead: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dim_feedforward: Dimension of the feedforward network inside the transformer.
            dropout: Dropout rate.
        """
        super(DCEModelTransformer, self).__init__()
        self.dce_dim = dce_dim
        self.d_model = d_model
        # Input embedding: from 2 (tissue and AIF at a time point) to d_model.
        self.input_embed = nn.Linear(2, d_model)
        # Learnable positional embedding of shape (dce_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(dce_dim, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # Final fully connected layer mapping from d_model to out_dim.
        self.fc = nn.Linear(d_model, out_dim)

    def forward(self, tissue, aif):
        # tissue and aif: shape (batch, dce_dim)
        # Combine into a single tensor of shape (batch, dce_dim, 2)
        x = torch.stack([tissue, aif], dim=2)
        # Apply linear embedding: now shape (batch, dce_dim, d_model)
        x = self.input_embed(x)
        # Add positional encoding
        x = x + self.pos_embed.unsqueeze(0)  # broadcast over batch
        # Transformer expects input shape (sequence_length, batch, embedding_dim)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        # Transpose back: shape (batch, dce_dim, d_model)
        x = x.transpose(0, 1)
        # Global average pooling over the sequence dimension
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        x = self.relu(x)
        out = self.fc(x)
        return out

# =============================================================================
# Define a wrapper module
# =============================================================================
class WrappedDCEModel(nn.Module):
    def __init__(self, base_model, length):
        super(WrappedDCEModel, self).__init__()
        self.base_model = base_model
        self.length = length

    def forward(self, x):
        # x is expected to have shape (batch, 2*length)
        tissue = x[:, :self.length]
        aif = x[:, self.length:]
        return self.base_model(tissue, aif)