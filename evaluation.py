                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
from dataset import extended_tofts

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import DCEModelFC, DCEModelCNN, DCEModelTransformer
from dataset import SimulationDataset  # If needed
from loss import mpae                # If needed

from scipy.optimize import least_squares
from scipy.stats import ttest_rel
from utils import bland_altman_plot

from numpy import ones, kron, mean, eye, hstack, dot, tile
from numpy.linalg import pinv

def generate_eval(args):
    
    # Instantiate the dataset. data_folder, num_samples, dce_dim, out_dim, dt, dist, max_params):
    dataset = SimulationDataset(args.input_folder,args.val_size, args.length, args.out_dim, args.dt, args.sim_dist,[args.max_vp,args.max_ktrans,args.max_ve,args.max_kep])
    
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
    
def fit_pred_ls(c_pl_pred, c_pan_pred, dt=None): 
    numel = len(c_pan_pred)
    
    # Create separate lists to hold the results for each output variable
    ktr_v = []
    kep_v = []
    ve_v = []
    vp_v = []
    fit_time_v = []
    residual_v = []
    
    for ind in range(numel):
        def fun_to_fit(x):
            return np.array(extended_tofts(c_pl_pred[ind], x[0], x[1], x[2],dt=dt) - c_pan_pred[ind]).flatten()

        timenow = time.time()
        if ind % 100 == 0: 
            print(f"Start fitting for prediction index: {ind}") 
        # vp, ktrans, kep 
        result = least_squares(fun_to_fit, x0=np.array([0.1, 0.1, 0.1]), bounds=((1e-4, 1e-6, 1e-6), (0.8, 0.2,0.6)))

        # Calculate residual
        residual_mse = tf.reduce_mean((fun_to_fit(result.x))**2).numpy()
        residual = mse_to_nrmse(residual_mse, c_pan_pred[ind])
        
        # Append results to corresponding lists
        
        vp_v.append(result.x[0])
        ve_v.append(result.x[1]/result.x[2])
        ktr_v.append(result.x[1])
        kep_v.append(result.x[2])
        fit_time_v.append(time.time() - timenow)
        residual_v.append(residual)

        if ind % 20 == 0: 
            print(f"Finished fitting for prediction index: {ind}")

    # Convert the lists to NumPy arrays
    ktr_v = np.array(ktr_v)
    vp_v=np.array(vp_v)
    kep_v = np.array(kep_v)
    ve_v = np.array(ve_v)
    fit_time_v = np.array(fit_time_v)
    residual_v = np.array(residual_v)

    # Return the five variables separately
    return vp_v,ktr_v,  ve_v, kep_v, fit_time_v, residual_v

def significance_stars(p):
    """Return significance stars based on p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."

def compute_nrmse_curve(gt_curve, fitted_curve):
    """
    Compute the NRMSE between two 1D curves.
    NRMSE = sqrt(mean((gt - fitted)^2)) / mean(gt)
    """
    eps = 1e-8
    return np.sqrt(np.mean((gt_curve - fitted_curve)**2)) / (np.mean(gt_curve) + eps)

    

def evaluate(args):
    """
    Evaluate the pre-trained model by comparing the fitted curve (via two routes)
    to the original tissue curve. In particular, for each sample:
      - Compute a fitted curve using the predicted parameters.
      - Refine these parameters via a least-squares fit.
      - Compute the NRMSE between the fitted curve and the original tissue curve.
      - Measure the time needed for the least-squares fitting.
      
    Also, a Bland–Altman plot (using bland_altman_plot) is produced on a scalar summary 
    (here, AUC) of the original tissue vs. the refined fitted curves.
    
    The results (NRMSE and fitting time) are compared between:
      - "Prediction" (direct from NN; fitting time is nearly zero)
      - "Refined Fitting" (after LS refinement)
      
    Significance levels (p-values) from paired t-tests are annotated on the plots.
    """
    # ------------------- 1. Load evaluation data -------------------
    data = np.load(args.eval_data)

    tissues = data['tissue']  # shape: (num_samples, length)
    aifs = data['aif']        # shape: (num_samples, length)
    kvs = data['kv']          # ground-truth kinetic parameters; not used directly here
    num_samples = len(tissues)
    print(f"[INFO] Loaded {num_samples} samples from {args.eval_data}.")

    # ------------------- 2. Load the pre-trained model -------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if args.model == "FC":
        model = DCEModelFC(in_dim=args.length, out_dim=args.out_dim, layer_num=args.layer_num).to(device)
    elif args.model == "CNN":
        model = DCEModelCNN(dce_dim=args.length, out_dim=args.out_dim).to(device)
    elif args.model == "Transformer":
        model = DCEModelTransformer(dce_dim=args.length, out_dim=args.out_dim, num_layers=args.layer_num).to(device)
    else:
        raise ValueError(f"Unknown Model Type '{args.model}'. Try FC, CNN, or Transformer.")

    if args.model_path is None or not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # ------------------- 3. Run inference to obtain predicted parameters -------------------
    batch_size = args.batch_size
    all_preds = []
    start_time = time.time()
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            tissue_batch = torch.tensor(tissues[start_idx:end_idx], dtype=torch.float32).to(device)
            aif_batch    = torch.tensor(aifs[start_idx:end_idx], dtype=torch.float32).to(device)
            pred_params = model(tissue_batch, aif_batch)  # Expected shape: (batch, 4) [vp, ktrans, ve, kep]
            all_preds.append(pred_params.cpu().numpy())
    inference_time = time.time() - start_time
    preds = np.concatenate(all_preds, axis=0)
    print(f"[INFO] Inference completed on {num_samples} samples in {inference_time:.2f} sec.")

    # ------------------- 4. For each sample, compute errors and fitting time -------------------
    # We compare two approaches:
    # (a) "Prediction": directly compute fitted curve from predicted parameters.
    # (b) "Refined Fitting": run a least-squares fit (using predicted parameters as initial guess)
    #      to refine the parameters and compute the fitted curve.
    errors_pred = []
    errors_refined = []
    time_pred = []      # Will be zeros (no iterative fitting)
    time_refined = []   # Fitting times from least squares

    # For Bland–Altman, compute a scalar summary (e.g., area under curve, AUC)
    auc_original = []
    auc_refined = []

    # Loop over each sample (this may be slow if num_samples is very large)
    # for i in range(num_samples):
    vp_nlls = []
    ktrans_nlls=[]
    kep_nlls=[]
    ve_nlls=[]

    vp_gt =  kvs[:args.val_size,0]
    ktrans_gt=kvs[:args.val_size,1]
    kep_gt=kvs[:args.val_size,2]
    ve_gt=kvs[:args.val_size,3]

    vp_pred =[]
    ktrans_pred=[]
    kep_pred=[]
    ve_pred=[]
    for i in range(args.val_size): # use val_size instead of whole validation set 
        aif = aifs[i]       # 1D array of length args.length
        tissue = tissues[i] # Original tissue curve
        
        # --- (a) Direct prediction ---
        # Use predicted parameters: order [vp, ktrans, ve, kep]
        pred_params = preds[i]
        pred_vp, pred_ktrans, pred_ve, pred_kep = pred_params
        # Note: extended_tofts uses [aif, vp, ktrans, kep, dt]; ve is computed internally.
        fitted_curve_pred = extended_tofts(aif, pred_vp, pred_ktrans, pred_kep, args.dt)
        err_pred = compute_nrmse_curve(tissue, fitted_curve_pred)
        errors_pred.append(err_pred)
        time_pred.append(0.0)  # No iterative fitting time

        # --- (b) Refined fitting via least-squares ---
        # Use predicted values for vp, ktrans, and kep as the initial guess.
        # (Assume that ve is computed as ktrans/kep.)
        # x0 = [pred_vp, pred_ktrans, pred_kep]
        x0=[0.1,0.1,0.1] #[vp, ktrans, ve, kep]fit first three
        # Define the residual function: difference between model (fitted curve) and tissue curve.
        def fun_to_fit(x):
            # x: [vp, ktrans, kep]
            return extended_tofts(aif, x[0], x[1], x[1]/x[2], args.dt) - tissue

        t0 = time.time()
        res = least_squares(fun_to_fit, x0, bounds=([1e-4, 1e-6, 0.1], [0.8, 0.2, 0.8]))
        fit_time = time.time() - t0
        time_refined.append(fit_time)

        x_opt = res.x
        refined_vp = x_opt[0]
        refined_ktrans = x_opt[1]
        refined_ve = x_opt[2]
        refined_kep = refined_ktrans / refined_ve  # Consistent with prior code
        # Compute the refined fitted curve.
        fitted_curve_refined = extended_tofts(aif, refined_vp, refined_ktrans, refined_kep, args.dt)
        err_refined = compute_nrmse_curve(tissue, fitted_curve_refined)
        errors_refined.append(err_refined)

        # Compute AUC (area under curve) for Bland–Altman plot.
        auc_original.append(np.trapz(tissue))
        auc_refined.append(np.trapz(fitted_curve_refined))

        vp_nlls.append(refined_vp)
        ktrans_nlls.append(refined_ktrans)
        kep_nlls.append(refined_kep)
        ve_nlls.append(refined_ve)

        vp_pred.append(pred_vp)
        ktrans_pred.append(pred_ktrans)
        ve_pred.append(pred_ve)
        kep_pred.append(pred_kep)

    errors_pred = np.array(errors_pred)
    errors_refined = np.array(errors_refined)
    time_pred = np.array(time_pred)
    time_refined = np.array(time_refined)
    auc_original = np.array(auc_original)
    auc_refined = np.array(auc_refined)

    vp_nlls=np.array(vp_nlls)
    ktrans_nlls=np.array(ktrans_nlls)
    kep_nlls=np.array(kep_nlls)
    ve_nlls=np.array(ve_nlls)

    vp_pred=np.array(vp_pred)
    ktrans_pred=np.array(ktrans_pred)
    ve_pred=np.array(ve_pred)
    kep_pred=np.array(kep_pred)
    # ------------------- 5. Statistical tests -------------------
    # Paired t-test on NRMSE (Prediction vs. Refined Fitting)
    t_err, p_err = ttest_rel(errors_pred, errors_refined)
    # Paired t-test on fitting time (Prediction vs. Refined Fitting)
    t_time, p_time = ttest_rel(time_pred, time_refined)

    print(f"[RESULT] Mean NRMSE (Prediction)  = {np.mean(errors_pred):.4f}")
    print(f"[RESULT] Mean NRMSE (Refined)     = {np.mean(errors_refined):.4f} (p = {p_err:.3e})")
    print(f"[RESULT] Mean fitting time (Refined) = {np.mean(time_refined):.4f} sec (p = {p_time:.3e})")

    # ------------------- 6. Plotting -------------------
    fig, axes = plt.subplots(1,2, figsize=(9, 5))

    # (a) Box plot: NRMSE for Prediction vs. Refined Fitting.
    bp = axes[0].boxplot([errors_pred, errors_refined], patch_artist=True, labels=["Prediction", "Refined"])
    axes[0].set_ylabel("NRMSE")
    axes[0].set_title("Fitted Curve NRMSE")
    for median in bp['medians']:
        median.set(color='black', linewidth=1.5)
    # Annotate significance between groups:
    def annotate_significance(ax, x1, x2, y, p_val):
        ax.plot([x1, x1, x2, x2], [y, y*1.05, y*1.05, y], color='k', linewidth=1.2)
        ax.text((x1+x2)*0.5, y*1.07, significance_stars(p_val), ha='center', va='bottom', color='k')
    y_max_err = max(np.max(errors_pred), np.max(errors_refined))
    annotate_significance(axes[0], 1, 2, y_max_err * 1.1, p_err)



    # (c) Bar plot: Fitting time for Prediction (0) vs. Refined Fitting.
    mean_time_pred = np.mean(time_pred)
    mean_time_refined = np.mean(time_refined)
    axes[1].bar(["Prediction", "Refined"], [mean_time_pred, mean_time_refined])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Fitting Time (sec)")
    axes[1].set_title("Average Fitting Time")
    y_max_time = max(mean_time_pred, mean_time_refined)
    annotate_significance(axes[1], 1, 2, y_max_time * 1.1, p_time)
        
    plt.tight_layout()
    # Save the figure in the directory specified by args.eval_result
    os.makedirs(args.eval_result, exist_ok=True)
    fig_path = os.path.join(args.eval_result, "evaluation_results.png")
    plt.savefig(fig_path, dpi=200)
    plt.show()

    fig, axes = plt.subplots(1,4, figsize=(18, 5))
    bland_altman_plot(axes[0], vp_gt, vp_pred, "Bland–Altman vp: params (Original vs. pred)")
    bland_altman_plot(axes[1], ktrans_gt, ktrans_pred, "Bland–Altman ktrans: params (Original vs. pred)")
    bland_altman_plot(axes[2], ve_gt, ve_pred, "Bland–Altman ve: params (Original vs. pred)")
    bland_altman_plot(axes[3], kep_gt, kep_pred, "Bland–Altman kep: params (Original vs. pred)")
    plt.tight_layout()
    fig_path = os.path.join(args.eval_result, "bland_altman_results_pred.png")
    plt.savefig(fig_path, dpi=200)
    plt.show()

    fig, axes = plt.subplots(1,4, figsize=(18, 5))
    bland_altman_plot(axes[0], vp_gt, vp_nlls, "Bland–Altman vp: params (Original vs. nlls)")
    bland_altman_plot(axes[1], ktrans_gt, ktrans_nlls, "Bland–Altman ktrans: params (Original vs. nlls)")
    bland_altman_plot(axes[2], ve_gt, ve_nlls, "Bland–Altman ve: params (Original vs. nlls)")
    bland_altman_plot(axes[3], kep_gt, kep_nlls, "Bland–Altman kep: params (Original vs. nlls)")
    plt.tight_layout()
    fig_path = os.path.join(args.eval_result, "bland_altman_results_nlls.png")
    plt.savefig(fig_path, dpi=200)
    plt.show()

    
    print(f"[INFO] Evaluation figure saved to {fig_path}")
    print("[INFO] Evaluation complete.")

# Example usage:
# if __name__ == "__main__":
#     args = parse_args()
#     if args.mode == "evaluate":
#         evaluate(args)

