import torch
import scipy.io as sio
import os
import numpy as np
from models import *
import glob
import matplotlib.pyplot as plt 
from dataset import extended_tofts
from loss import mpae
# =============================================================================
# Inference function on .mat files
# =============================================================================
def infer(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model, loss, optimizer, scheduler
    if args.model=="FC":
        model = DCEModelFC(in_dim=args.length, out_dim=args.out_dim, layer_num=args.layer_num).to(device)
    elif args.model=="CNN":
        model = DCEModelCNN(dce_dim=args.length, out_dim=args.out_dim).to(device)
    elif args.model=="Transformer":
        model = DCEModelTransformer(dce_dim=args.length, out_dim=args.out_dim, num_layers=args.layer_num).to(device)
    else:
        raise "Unknown Model Type. Try FC, CNN, or Transformer."
    checkpoint = torch.load(args.model_path, weights_only=False)  # Ensure it's not just weights_only=True
    model.load_state_dict(checkpoint) 
    run_inference(model,device,args.input_folder,\
                  args.output_folder,args.length,args.batch_size,args.dt, \
                  args.LDAIF,args.subject_mean,args.bias_corr)
    
def run_inference(model, device, input_folder, output_folder, length, batch_size,dt, LDAIF='individual',subject_mean = 'parameter',bias_corr=False): # common or individual? parameter or curve?
    model.eval()
    os.makedirs(output_folder, exist_ok=True)
    mat_files = glob.glob(os.path.join(input_folder, '*.mat'))
    
    
    for mat_path in mat_files:
        mat_data = sio.loadmat(mat_path)
        base_name = os.path.basename(mat_path)
        """
        pre-contrast bias correction 
        """
        if bias_corr and (base_name.endswith('LD2.mat') or base_name.endswith('SD.mat')):
            # Construct corresponding LD1 file name.
            ld1_name = base_name.replace('LD2.mat', 'LD1.mat')
            ld1_path = os.path.join(input_folder, ld1_name)
            if os.path.exists(ld1_path):
                print(f"For {base_name}, using AIF from {ld1_name}.")
                ld1_data = sio.loadmat(ld1_path)
                r1_artery_ld1 = ld1_data['fits_T1_artery'].astype(np.float32)
                n_dce = r1_artery_ld1.shape[-1]-3
                r1_artery_ld1=np.mean(r1_artery_ld1[:,1:n_dce+1],axis=0)
                r1_artery_ld1_0 = np.mean(r1_artery_ld1[:20])

                r1_pan_ld1 = ld1_data['fits_T1_nontumor'].astype(np.float32)
                r1_pan_ld1_0 = np.mean(r1_pan_ld1[:,:20])

                r1_pan_now = mat_data['fits_T1_nontumor'].astype(np.float32)[:,1:n_dce+1]
                r1_artery_now = mat_data['fits_T1_artery'].astype(np.float32)[:,1:n_dce+1]
                r1_artery_now=np.mean(r1_artery_now,axis=0)
                
                c_bd = (r1_artery_now-r1_artery_ld1_0)/(1-0.4)/3.2
                c_bd = c_bd[34:34+196]

                c_all = (r1_pan_now-r1_pan_ld1_0)/3.2
                c_all = c_all[34:34+196]
            else:
                print(f"Warning: {ld1_name} not found; using current file's AIF.")
                mat_data = sio.loadmat(mat_path)
                c_bd = mat_data['c_bd'].flatten().astype(np.float32)


        else:
            
            c_bd = mat_data['c_bd'].flatten().astype(np.float32)
            c_all = mat_data['c_all'][:,:-10].astype(np.float32)  # shape (num_pixels, temporal_length)
            c_all[c_all<0]=0
        ##############################
        
        # if LDAIF == 'common' and base_name.endswith('LD2.mat'):
        #     # Construct corresponding LD1 file name.
        #     ld1_name = base_name.replace('LD2.mat', 'LD1.mat')
        #     ld1_path = os.path.join(input_folder, ld1_name)
        #     if os.path.exists(ld1_path):
        #         print(f"For {base_name}, using AIF from {ld1_name}.")
        #         ld1_data = sio.loadmat(ld1_path)
        #         c_bd = ld1_data['c_bd'].flatten().astype(np.float32)
        #     else:
        #         print(f"Warning: {ld1_name} not found; using current file's AIF.")
        #         mat_data = sio.loadmat(mat_path)
        #         c_bd = mat_data['c_bd'].flatten().astype(np.float32)
        # else:
        #     mat_data = sio.loadmat(mat_path)
        #     c_bd = mat_data['c_bd'].flatten().astype(np.float32)
        
        if subject_mean == 'parameter':

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
    
            # -------------------------- Figure 1 Distribution ----------------------------------------------
            out_fig = os.path.join(output_folder, os.path.basename(mat_path).split('.')[0]+'_inference_dist.png')
            sio.savemat(out_path, {'predictions': preds_array})
            print(f'Processed {mat_path} -> {out_path}')
            # Plot histogram distributions
            plt.figure(figsize=(10, 6))
            labels = ['vp', 've', 'ktrans','kep']
            for i in range(preds_array.shape[1]):
                plt.hist(preds_array[:, i], bins=50, alpha=0.5, label=labels[i], density=True)
            plt.title('Distribution of Predictions')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()
            plt.savefig(out_fig)
    
            # -------------------------- Figure 2: Fitting Goodness ----------------------------------------------
            # Randomly select three sample indices
            selected_indices = np.random.choice(num_samples, 3, replace=False)
            plt.figure(figsize=(15, 5))
            for j, idx in enumerate(selected_indices):
                original_tissue = curves[idx]  # Original tissue curve from c_all
                # Get predicted parameters for this sample (assumed order: vp, ktrans, ve, kep)
                pred_params = preds_array[idx]
                vp, ktrans, ve, kep = pred_params
                # Compute fitted tissue curve using the extended Tofts model.
                fitted_tissue = extended_tofts(c_bd_extrap, vp, ktrans, kep, dt=dt)
                error = mpae(fitted_tissue,original_tissue)
                plt.subplot(1, 3, j+1)
                plt.plot(original_tissue, label="Original Tissue", linewidth=2)
                plt.plot(fitted_tissue, label=f"Fitted Tissue, MAPE={error:.3f}", linestyle="--")
                plt.xlabel("Time Index")
                plt.ylabel("Concentration")
                plt.legend()
            plt.tight_layout()
            out_fig2 = os.path.join(output_folder, os.path.basename(mat_path).split('.')[0] + '_fitting_goodness.png')
            plt.savefig(out_fig2)
            plt.show()
        else:

            c_all = np.mean(c_all,axis=0,keepdims=True)
            # For demonstration, mimic curve extraction
            # (here we simply crop or pad each row to desired length)
            def extrapolate_to_length(data, target_length):
                pad = target_length - len(data)
                if pad > 0:
                    return np.pad(data, (pad, 0), mode='edge')
                return data[:target_length]
            c_bd_extrap = extrapolate_to_length(c_bd, length)
            curves = np.array([extrapolate_to_length(row, length) for row in c_all])
            print(curves.shape)
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
            # -------------------------- Figure 2: Fitting Goodness ----------------------------------------------
            # Randomly select three sample indices
            selected_indices = [0]
            plt.figure(figsize=(15, 5))
            for j, idx in enumerate(selected_indices):
                original_tissue = curves[idx]  # Original tissue curve from c_all
                # Get predicted parameters for this sample (assumed order: vp, ktrans, ve, kep)
                pred_params = preds_array[idx]
                vp, ktrans, ve, kep = pred_params
                # Compute fitted tissue curve using the extended Tofts model.
                fitted_tissue = extended_tofts(c_bd_extrap, vp, ktrans, kep, dt=dt)
                error = mpae(fitted_tissue,original_tissue)
                plt.subplot(1, 3, j+1)
                plt.plot(original_tissue, label="Original Tissue", linewidth=2)
                plt.plot(fitted_tissue, label=f"Fitted Tissue, MAPE={error:.3f}", linestyle="--")
                plt.xlabel("Time Index")
                plt.ylabel("Concentration")
                plt.legend()
            plt.tight_layout()
            out_fig2 = os.path.join(output_folder, os.path.basename(mat_path).split('.')[0] + '_fitting_goodness.png')
            plt.savefig(out_fig2)
            plt.show()            

        
        