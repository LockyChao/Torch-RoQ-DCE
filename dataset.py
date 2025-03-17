import os
import glob
import numpy as np
import scipy.io as sio
import random
import torch
from torch.utils.data import Dataset

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


def generate_kv(dist='uniform', max_vp=0.8, max_ktrans=0.2, max_ve=0.8, max_kep=0.6):
    if dist == 'uniform':
        while True:
            vp = np.random.uniform(1e-4, max_vp)
            ktrans = np.random.uniform(1e-4, max_ktrans)
            kep = np.random.uniform(1e-4, max_kep)  # Avoid kep=0 by setting a small positive lower bound.
            ve = ktrans / kep
            if 0.1 <= ve <= max_ve:
                return vp, ktrans, ve, kep

    elif dist == 'normal':
        while True:
            # Define mean and standard deviation for normal distribution
            mean_vp, std_vp = 0.2, 0.1
            mean_ktrans, std_ktrans = 0.02, 0.01
            mean_ve, std_ve = 0.2, 0.1

            vp = np.random.normal(mean_vp, std_vp)
            ktrans = np.random.normal(mean_ktrans, std_ktrans)
            ve = np.random.normal(mean_ve, std_ve)

            # Ensure values fall within valid ranges
            if vp <= 0 or ktrans <= 0 or ve <= 0.1:
                continue

            kep = ktrans / ve

            if 0.1 <= ve <= max_ve and vp < max_vp and ktrans < max_ktrans and kep < max_kep:
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
    def __init__(self, data_folder, num_samples, dce_dim, out_dim, dt, dist, max_params):
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
        self.dist =dist
        self.max_vp,self.max_ktrans,self.max_ve,self.max_kep = max_params
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

        vp, ktrans, ve, kep =  generate_kv(self.dist,self.max_vp,self.max_ktrans,self.max_ve,self.max_kep)
        kv=np.array([vp, ktrans, ve, kep],dtype=np.float32)
        
        # Compute tissue using the extended Tofts model.
        tissue = extended_tofts(aif_scaled, vp, ktrans, kep, dt=self.dt)
        tissue = add_gaussian_noise(tissue, snr_dB=20).astype(np.float32)
        
        return tissue, aif_scaled, kv

