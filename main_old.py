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
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F 

import mlflow
import mlflow.pytorch

# =============================================================================
# Simulation Dataset (for training/validation)
# =============================================================================
# For demonstration purposes we simulate data.
# In practice you can call your simulation functions from simu.py.
def extrapolate_to_length(data, target_length):
    pad = target_length - len(data)
    if pad > 0:
        return np.pad(data, (pad, 0), mode='edge')
    return data[:target_length]
def generate_kv():
    while True:
        vp = np.random.uniform(0, 0.2)
        ktrans = np.random.uniform(0, 5/60.0)
        # Avoid kep=0 by setting a small positive lower bound.
        kep = np.random.uniform(1e-6, 20/60.0)
        ve = ktrans / kep
        if 0.1 <= ve <= 0.6:
            return vp, ktrans, ve, kep
def extended_tofts(aif, vp, ktrans, kep, dt=2.0608):
    """
    Compute tissue concentration using the extended Tofts model.
    
    Ct(t) = vp * Cp(t) + ktrans * ∫_0^t Cp(u) exp(-kep*(t-u)) du
    
    Here, Cp(t) is the AIF and dt is the sampling interval.
    We implement the integral as a discrete convolution.
    """
    t = np.arange(len(aif)) * dt
    kernel = np.exp(-kep * t)  # kernel: exp(-kep*t)
    conv_result = np.convolve(aif, kernel, mode='full')[:len(aif)] * dt
    tissue = vp * aif + ktrans * conv_result
    return tissue

def extended_tofts_torch(aif, vp, ktrans, kep, dt=2.0608):
    """
    Compute tissue concentration using the extended Tofts model in PyTorch.

    Ct(t) = vp * Cp(t) + ktrans * ∫_0^t Cp(u) exp(-kep*(t-u)) du

    Parameters:
    - aif: Arterial input function (tensor of contrast concentration in plasma over time) [Shape: (N,)]
    - vp: Plasma volume fraction (scalar tensor)
    - ktrans: Transfer constant from plasma to EES (scalar tensor)
    - kep: Rate constant from EES to plasma (scalar tensor)
    - dt: Sampling interval (default=1.0)

    Returns:
    - Tissue contrast concentration Ct as a tensor [Shape: (N,)]
    """
    t = torch.arange(len(aif), dtype=aif.dtype, device=aif.device) * dt
    kernel = torch.exp(-kep * t)  # Exponential kernel: exp(-kep * t)
    
    # Perform discrete convolution using F.conv1d (requires 3D shape: (batch, channels, length))
    aif_reshaped = aif.view(1, 1, -1)  # (1, 1, N)
    kernel_reshaped = kernel.view(1, 1, -1)  # (1, 1, N)

    conv_result = F.conv1d(aif_reshaped, kernel_reshaped, padding=0) * dt  # Apply convolution
    conv_result = conv_result.squeeze()  # Remove batch and channel dimensions

    # Compute tissue concentration
    tissue = vp * aif + ktrans * conv_result
    return tissue
def add_gaussian_noise(signal, snr_dB=20):
    """
    Add zero-mean Gaussian noise to the signal to achieve a desired SNR (in dB).
    
    SNR_dB = 10 * log10(signal_power / noise_power)
    """
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10 ** (snr_dB / 10))
    noise_std = np.sqrt(noise_power)
    noise = np.random.normal(0, noise_std, signal.shape)
    return signal + noise
class SimulationDataset(Dataset):
    def __init__(self, data_folder, num_samples, dce_dim, out_dim, dt):
        """
        Args:
            data_folder: path to the folder containing .mat files.
            num_samples: total number of samples to yield (i.e. dataset length).
            dce_dim: the desired length for the AIF (and other signals).
            out_dim: output dimension (KV parameters).
        """
        self.num_samples = num_samples
        self.dce_dim = dce_dim
        self.out_dim = out_dim
        self.dt = dt
        
        # Load all AIFs (c_bd variable) from .mat files in data_folder.
        self.aif_list = []
        mat_files = glob.glob(os.path.join(data_folder, "*.mat"))
        if not mat_files:
            raise ValueError(f"No .mat files found in {data_folder}")
        for mat_path in mat_files:
            mat_data = sio.loadmat(mat_path)
            if "c_bd" in mat_data:
                # Extract c_bd, flatten, cast to float32
                aif = mat_data["c_bd"].flatten().astype(np.float32)
                # normalize AIF itself so that its max is 1 
                aif = aif/np.max(aif)
                # Ensure the AIF has the desired length by cropping or padding.
                aif = extrapolate_to_length(aif, self.dce_dim)
                self.aif_list.append(aif)
        if not self.aif_list:
            raise ValueError("No 'c_bd' variable found in any .mat files.")
        print(f"Loaded {len(self.aif_list)} AIFs from {data_folder}.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Pick a random AIF from the loaded list.
        aif = random.choice(self.aif_list)
        # Draw a scaling factor from a mixture of two Gaussians:
        # With 50% probability sample from N(1.2, 0.2), and otherwise from N(4.0, 0.5) to simulate low-dose and SD scenario 
        if np.random.rand() < 0.5:
            scale_factor = np.random.normal(1.5, 0.4)
        else:
            scale_factor = np.random.normal(4.0, 0.5)
        # Clip the scaling factor to be between 0.8 and 5.0.
        scale_factor = np.clip(scale_factor, 0.6, 6.0)

        aif_scaled = (aif * scale_factor).astype(np.float32)

        # For demonstration, simulate a dummy tissue signal and target KV.

        vp, ktrans, ve, kep =  generate_kv()
        kv=np.array([vp, ktrans, ve, kep],dtype=np.float32)
        
        # Compute tissue using the extended Tofts model.
        tissue = extended_tofts(aif_scaled, vp, ktrans, kep, dt=self.dt)
        tissue = add_gaussian_noise(tissue, snr_dB=20).astype(np.float32)
        
        return tissue, aif_scaled, kv

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
# Plots and Figures 
# =============================================================================
# Function to create Bland-Altman plots with R², ICC, and CoV annotation
def bland_altman_plot(ax, X, Y, title_str):
    """
    Create a Bland–Altman plot with equal axis scaling.
    """
    avg_val = (X + Y) / 2.0
    diff_val = Y - X

    sns.scatterplot(x=avg_val, y=diff_val, ax=ax, marker='o', edgecolor='none')
    ax.set_title(f'{title_str}', fontsize=12)

    mean_diff = np.mean(diff_val)
    std_diff = np.std(diff_val)
    loa_lower = mean_diff - 1.96 * std_diff
    loa_upper = mean_diff + 1.96 * std_diff

    ax.axhline(mean_diff, color='red', linestyle='--', linewidth=1.5)
    ax.axhline(loa_lower, color='black', linestyle='--')
    ax.axhline(loa_upper, color='black', linestyle='--')
    ax.set_xlabel('Average')
    ax.set_ylabel('Difference (Y - X)')

    # Ensure equal axis scaling
    x_min, x_max = avg_val.min(), avg_val.max()
    y_min, y_max = diff_val.min(), diff_val.max()

    axis_limit = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
    ax.set_xlim([0, axis_limit])
    ax.set_ylim([-axis_limit, axis_limit])

    # Display R², ICC, and CoV instead of mean, LoA, and UpA
    slope, intercept, R_val, p_value, std_err = scipy.stats.linregress(X,Y)
    r_sq = R_val**2

    icc_data = np.concatenate((X,Y),axis=1)
    var = scipy.stats.variation(icc_data, axis=1)
    cov_val = np.mean(var,axis=0)*100
    
    ax.text(axis_limit * 0.4, -axis_limit * 0.9,
            f"R² = {r_sq:.3f}\nCoV = {cov_val:.2f}%",
            fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

# =============================================================================
# Inference function on .mat files
# =============================================================================
def infer(args):
    #to be done 
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
#  generate eval data
# =============================================================================
def generate_eval(args):
    
    # Instantiate the dataset.
    dataset = SimulationDataset(args.input_folder,args.val_size, args.length, args.out_dim, args.dt)
    
    tissues = []
    aifs = []
    kvs = []
    for i in range(len(dataset)):
        tissue, aif, kv = dataset[i]
        tissues.append(tissue)
        aifs.append(aif)
        kvs.append(kv)
    
    tissues = np.array(tissues)  # shape: (num_samples, dce_dim)
    aifs = np.array(aifs)        # shape: (num_samples, dce_dim)
    kvs = np.array(kvs)          # shape: (num_samples, 4)
    
    # Save the generated data.
    np.savez(args.eval_data, tissue=tissues, aif=aifs, kv=kvs)
    print(f"Saved test_data.npz with {args.val_size} samples.")
# =============================================================================
# Main training function
# =============================================================================
def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets and dataloaders
    train_dataset = SimulationDataset(args.input_folder,args.sim_size, args.length, args.out_dim, args.dt)
    val_dataset = SimulationDataset(args.input_folder,args.val_size, args.length, args.out_dim, args.dt)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model, loss, optimizer, scheduler
    if args.model=="FC":
        model = DCEModelFC(in_dim=args.length, out_dim=args.out_dim, layer_num=args.layer_num).to(device)
    elif args.model=="CNN":
        model = DCEModelCNN(dce_dim=args.length, out_dim=args.out_dim).to(device)
    elif args.model=="Transformer":
        model = DCEModelTransformer(dce_dim=args.length, out_dim=args.out_dim, num_layers=args.layer_num).to(device)
    else:
        raise "Unknown Model Type. Try FC, CNN, or Transformer."
    criterion = MixedLoss(alpha=args.alpha)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    # Initiating eval dataset 
    # Evaluate on pre-generated test data.
    # Load test data (assumed stored in 'test_data.npz')
    test_data = np.load(args.eval_data)
    tissue_test = torch.tensor(test_data["tissue"], dtype=torch.float32).to(device)  # shape: (N, dce_dim)
    aif_test = torch.tensor(test_data["aif"], dtype=torch.float32).to(device)      # shape: (N, dce_dim)
    kv_true = torch.tensor(test_data["kv"], dtype=torch.float32).to(device)          # shape: (N, 4)
    test_input = torch.cat([tissue_test, aif_test], dim=1)
    
    
    # Set up MLFlow
    mlflow.start_run()
    mlflow.log_params(vars(args))
    run_id = mlflow.active_run().info.run_id

    best_val_loss = float('inf')
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.prefix,run_id)
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
    wrapped_model.eval()
    
    # Move the wrapped model to CPU for logging
    wrapped_model = wrapped_model.cpu()
    kv_pred = wrapped_model(test_input)

    # Create a plot comparing predictions vs. ground truth for each parameter.
    kv_pred_np = kv_pred.cpu().numpy()
    kv_true_np = kv_true
    
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    param_names = ["vp", "ktrans", "ve", "kep"]

    for p, ax in enumerate(axes):
        bland_altman_plot(ax, kv_pred_np[:, p], kv_true_np[:, p], param_names[p])
            

    # Save the plot in a dedicated artifacts folder, including the run id in the filename.

    
    eval_plot_path = os.path.join(args.eval_result, f"evaluation_plot_run_{run_id}.png")
    plt.savefig(eval_plot_path)
    mlflow.log_artifact(eval_plot_path, artifact_path="evaluation_plots")
    
    # Log the model with the dummy input example.
    mlflow.pytorch.log_model(wrapped_model, "model", input_example=test_input)

    mlflow.end_run()

    # Optionally run inference on external .mat files
    if args.infer:
        run_inference(model, device, args.input_folder, args.output_folder, args.length, args.infer_batch)

# =============================================================================
# Command-line arguments
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train DCE model in PyTorch with MLFlow tracking")
    parser.add_argument("--mode", type=str, default='train', help="train or generate_eval or infer")
    parser.add_argument("--model", type=str, default='FC', help="choice of DL model: FC, CNN, Transformer")
    parser.add_argument("--length", type=int, default=210, help="Length of input DCE signal")
    parser.add_argument("--dt", type=float, default=2.0608, help="Delta t of DCE curve, Ng*TR")
    parser.add_argument("--out_dim", type=int, default=4, help="Output dimension (KV parameters)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--layer_num", type=int, default=4, help="Number of dense layers in feature extractor")
    parser.add_argument("--sim_size", type=int, default=int(1e6), help="Number of simulated training samples")
    parser.add_argument("--val_size", type=int, default=int(1e4), help="Number of simulated validation samples")
    parser.add_argument("--alpha", type=float, default=0.998, help="Alpha weight for MixedLoss")
    parser.add_argument("--checkpoint_dir", type=str, default="/hdd1/chaowei/checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--prefix", type=str, default="Simulation_Fitting", help="Prefix for log/checkpoint names")
    parser.add_argument("--eval_data", type=str, default="/hdd1/chaowei/test_data.npz", help="Where is the eval data")
    parser.add_argument("--eval_result", type=str, default="./artifacts", help="Where is the eval result saved")
    # Inference options
    parser.add_argument("--infer", action="store_true", help="Run inference on .mat files after training")
    parser.add_argument("--input_folder", type=str, default="/mnt/LiDXXLab_Files/Chaowei/Low-dose Study/AI_Fitting/Input_origTofts/", help="Folder with .mat files for inference")
    parser.add_argument("--model_path", type=str, default=None, help="path of trained model")
    parser.add_argument("--output_folder", type=str, default="./output_mat", help="Folder to save inference results")
    parser.add_argument("--infer_batch", type=int, default=32, help="Batch size for inference")

    
    return parser.parse_args()

# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    args = parse_args()
    if args.mode=='generate_eval':
        generate_eval(args)
    elif args.mode=='infer':
        infer(args)
    else:
        train(args)
