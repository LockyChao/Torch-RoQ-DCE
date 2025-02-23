#!/usr/bin/env python
"""
PyTorch version of the DCE model training script.
Hyperparameters are specified via command-line arguments.
MLFlow is used for tracking the experiment.
"""

import argparse
import os
import time
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import mlflow
import mlflow.pytorch

# =============================================================================
# Simulation Dataset (for training/validation)
# =============================================================================
# For demonstration purposes we simulate data.
# In practice you can call your simulation functions from simu.py.
class SimulationDataset(Dataset):
    def __init__(self, sim_size, dce_dim, out_dim):
        self.sim_size = sim_size
        self.dce_dim = dce_dim
        self.out_dim = out_dim

    def __len__(self):
        return self.sim_size

    def __getitem__(self, idx):
        # Replace these with calls to your simulation functions if available.
        # For now we simulate two input signals (“Tissue” and “AIF”) as 1D arrays,
        # and the target “KV” as a 1D vector.
        tissue = np.random.rand(self.dce_dim).astype(np.float32)
        aif = np.random.rand(self.dce_dim).astype(np.float32)
        # For example, KV can be a function of aif and tissue (here simulated randomly)
        kv = np.random.rand(self.out_dim).astype(np.float32)
        # Return as a tuple: (tissue, aif, target)
        return tissue, aif, kv

# =============================================================================
# PyTorch Model
# =============================================================================
class DCEModel(nn.Module):
    def __init__(self, in_dim, out_dim, layer_num):
        """
        Args:
            in_dim: length of each input signal (Tissue and AIF). 
                    The concatenated input dimension is 2*in_dim.
            out_dim: dimension of the output (e.g. 3).
            layer_num: number of dense layers in the “deconv” part.
        """
        super(DCEModel, self).__init__()
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
# =============================================================================
# Loss function: Weighted sum of L1 loss and Mean Absolute Percentage Error (MAPE)
# =============================================================================
class MixedLoss(nn.Module):
    def __init__(self, alpha=0.998, eps=1e-6):
        """
        alpha: weight for L1 loss; (1-alpha) is the weight for MAPE.
        """
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        # Compute MAPE; add epsilon to denominator to avoid division by zero.
        mape_loss = torch.mean(torch.abs((target - pred) / (target + self.eps)))
        return self.alpha * l1_loss + (1 - self.alpha) * mape_loss

# =============================================================================
# Inference function on .mat files
# =============================================================================
def run_inference(model, device, input_folder, output_folder, length, batch_size):
    model.eval()
    os.makedirs(output_folder, exist_ok=True)
    mat_files = glob.glob(os.path.join(input_folder, '*.mat'))
    for mat_path in mat_files:
        mat_data = sio.loadmat(mat_path)
        # Assume that the .mat file contains:
        #   'c_bd' as AIF (vector) and 'c_all' from which we extract tissue curves.
        c_bd = mat_data['c_bd'].flatten().astype(np.float32)
        c_all = mat_data['c_all']  # shape (num_pixels, temporal_length)
        # For demonstration, mimic curve extraction
        # (here we simply crop or pad each row to desired length)
        def extrapolate_to_length(data, target_length):
            pad = target_length - len(data)
            if pad > 0:
                return np.pad(data, (pad, 0), mode='edge')
            return data[:target_length]
        c_bd_extrap = extrapolate_to_length(c_bd, length)
        curves = np.array([extrapolate_to_length(row, length) for row in c_all])
        predictions = []
        # Process in batches
        num_samples = curves.shape[0]
        with torch.no_grad():
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                tissue_batch = torch.tensor(curves[start:end]).to(device)
                # Repeat the AIF for each sample in the batch.
                aif_batch = torch.tensor(np.repeat(c_bd_extrap[np.newaxis, :], end - start, axis=0)).to(device)
                pred = model(tissue_batch, aif_batch)
                predictions.append(pred.cpu().numpy())
        preds_array = np.vstack(predictions)
        # Save predictions to .mat file
        out_path = os.path.join(output_folder, os.path.basename(mat_path))
        sio.savemat(out_path, {'predictions': preds_array})
        print(f'Processed {mat_path} -> {out_path}')
        # Plot histogram distributions
        plt.figure(figsize=(10, 6))
        labels = ['vp', 've', 'ktrans']
        for i in range(preds_array.shape[1]):
            plt.hist(preds_array[:, i], bins=50, alpha=0.5, label=labels[i], density=True)
        plt.title('Distribution of Predictions')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

# =============================================================================
# Main training function
# =============================================================================
def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets and dataloaders
    train_dataset = SimulationDataset(args.sim_size, args.length, args.out_dim)
    val_dataset = SimulationDataset(args.val_size, args.length, args.out_dim)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model, loss, optimizer, scheduler
    model = DCEModel(in_dim=args.length, out_dim=args.out_dim, layer_num=args.layer_num).to(device)
    criterion = MixedLoss(alpha=args.alpha)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    # Set up MLFlow
    mlflow.start_run()
    mlflow.log_params(vars(args))

    best_val_loss = float('inf')
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.prefix)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for i, (tissue, aif, kv) in enumerate(train_loader):
            tissue = tissue.to(device)
            aif = aif.to(device)
            kv = kv.to(device)
            optimizer.zero_grad()
            outputs = model(tissue, aif)
            loss = criterion(outputs, kv)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch} completed in {elapsed:.2f}s, Average Loss: {avg_loss:.4f}")

        mlflow.log_metric("train_loss", avg_loss, step=epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tissue, aif, kv in val_loader:
                tissue = tissue.to(device)
                aif = aif.to(device)
                kv = kv.to(device)
                outputs = model(tissue, aif)
                loss = criterion(outputs, kv)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

        # Save checkpoint if improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch{epoch:04d}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Step the scheduler every 10 epochs (simulate decay schedule)
        if epoch % 10 == 0:
            scheduler.step()
            print(f"Learning rate decayed to: {scheduler.get_last_lr()[0]:.6f}")

    # Log final model with MLFlow

    
    # In your training function, after training is complete:
    # Wrap your trained model (this can be defined as shown previously)
    wrapped_model = WrappedDCEModel(model, args.length)
    
    # Move the wrapped model to CPU for logging
    wrapped_model = wrapped_model.cpu()
    
    # Create a dummy input example as a NumPy array with float32 type.
    dummy_input = np.random.randn(1, args.length * 2).astype(np.float32)
    
    # Log the model with the dummy input example.
    mlflow.pytorch.log_model(wrap pped_model, "model", input_example=dummy_input)

    mlflow.end_run()

    # Optionally run inference on external .mat files
    if args.infer:
        run_inference(model, device, args.input_folder, args.output_folder, args.length, args.infer_batch)

# =============================================================================
# Command-line arguments
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train DCE model in PyTorch with MLFlow tracking")
    parser.add_argument("--length", type=int, default=210, help="Length of input DCE signal")
    parser.add_argument("--out_dim", type=int, default=3, help="Output dimension (KV parameters)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--layer_num", type=int, default=4, help="Number of dense layers in feature extractor")
    parser.add_argument("--sim_size", type=int, default=int(1e6), help="Number of simulated training samples")
    parser.add_argument("--val_size", type=int, default=int(1e4), help="Number of simulated validation samples")
    parser.add_argument("--alpha", type=float, default=0.998, help="Alpha weight for MixedLoss")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--prefix", type=str, default="Simulation_Fitting", help="Prefix for log/checkpoint names")
    # Inference options
    parser.add_argument("--infer", action="store_true", help="Run inference on .mat files after training")
    parser.add_argument("--input_folder", type=str, default="./input_mat", help="Folder with .mat files for inference")
    parser.add_argument("--output_folder", type=str, default="./output_mat", help="Folder to save inference results")
    parser.add_argument("--infer_batch", type=int, default=32, help="Batch size for inference")
    return parser.parse_args()

# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    args = parse_args()
    train(args)
