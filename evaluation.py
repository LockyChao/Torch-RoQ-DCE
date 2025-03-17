from dataset import extended_tofts
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

        if ind % 100 == 0: 
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

import os
import glob
import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def evaluate(
    pretrained_folder,
    varpro_folder=None,
    nlls_folder=None,
    gt_folder=None,
    dt=1.0,
    save_fig_path="evaluation_results.png"
):
    """
    Evaluate NRMSE and fitting time for the pre-trained model (and optionally
    VARPRO and NLLS) against ground truth, then plot and mark significance levels.

    Parameters
    ----------
    pretrained_folder : str
        Path to folder containing .mat files with predictions from the pre-trained model.
    varpro_folder : str, optional
        Path to folder containing .mat files with VARPRO predictions.
    nlls_folder : str, optional
        Path to folder containing .mat files with NLLS predictions.
    gt_folder : str, optional
        Path to folder containing ground truth .mat files, or it could be the same .mat
        files that also contain ground truth tissue curves.
    dt : float, optional
        Time spacing (used if you want to re-fit or do additional computations).
    save_fig_path : str, optional
        Where to save the final evaluation figure.

    Returns
    -------
    None
        Plots and saves a figure showing boxplots of NRMSE and bar plots of inference/fitting time,
        with significance levels indicated.

    Notes
    -----
    - This is a skeleton function; adapt data loading, curve extraction, and time measurements
      to your specific use case.
    - If you already have arrays of NRMSE/time for each method, you can skip the .mat loading
      and simply do the plotting and significance testing.
    """

    # Helper function to compute NRMSE
    def compute_nrmse(gt_curve, pred_curve):
        """
        Compute Normalized Root Mean Square Error:
        sqrt(mean((gt - pred)^2)) / mean(gt)
        """
        gt_curve = np.asarray(gt_curve, dtype=np.float32)
        pred_curve = np.asarray(pred_curve, dtype=np.float32)
        if np.mean(gt_curve) < 1e-8:
            return 0.0  # Avoid division by zero if ground truth is near 0
        return np.sqrt(np.mean((gt_curve - pred_curve)**2)) / np.mean(gt_curve)

    # Optionally, a helper to label significance stars
    def significance_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "n.s."

    # Prepare lists to store NRMSEs and times for each method
    nrmse_pretrained = []
    nrmse_varpro    = []
    nrmse_nlls      = []

    time_pretrained = []
    time_varpro     = []
    time_nlls       = []

    # ----------------------------------------------------------------------------
    # (1) Load predictions and ground truth for Pre-trained model
    # ----------------------------------------------------------------------------
    if pretrained_folder is not None:
        pretrained_files = glob.glob(os.path.join(pretrained_folder, '*.mat'))
        for mat_path in pretrained_files:
            mat_data = sio.loadmat(mat_path)
            # Example: if 'predictions' key holds the model predictions
            # If each .mat has shape (#_pixels, #_params) or (#_pixels, #_timepoints), adapt accordingly.
            if 'predictions' not in mat_data:
                continue

            # Load predicted curves or parameters
            pred = mat_data['predictions']

            # Load ground truth from the same or separate .mat file
            # In many setups, ground truth might be in the same file, or you might
            # look up a file with the same base name in gt_folder.
            # Here is an example if ground truth is also in mat_data:
            if gt_folder is not None:
                base_name = os.path.basename(mat_path)
                gt_path = os.path.join(gt_folder, base_name)
                if os.path.exists(gt_path):
                    gt_data = sio.loadmat(gt_path)
                    # Suppose ground-truth tissue curve is "c_tissue_gt"
                    # and you want to compare them with the predicted curve "pred".
                    # In practice, adapt to your actual naming and shapes.
                    if 'c_tissue_gt' in gt_data:
                        c_tissue_gt = gt_data['c_tissue_gt']
                        # Example: compute NRMSE across all pixels
                        # If shape mismatch, adapt accordingly
                        for i in range(len(c_tissue_gt)):
                            # Just a demonstration. Adjust to match your data shape:
                            c_gt = c_tissue_gt[i]
                            c_pd = pred[i]
                            # Compute NRMSE
                            nrmse_val = compute_nrmse(c_gt, c_pd)
                            nrmse_pretrained.append(nrmse_val)
            else:
                # Or skip if ground truth is in the same file:
                if 'c_tissue_gt' in mat_data:
                    c_tissue_gt = mat_data['c_tissue_gt']
                    for i in range(len(c_tissue_gt)):
                        nrmse_val = compute_nrmse(c_tissue_gt[i], pred[i])
                        nrmse_pretrained.append(nrmse_val)

            # Measure or retrieve inference time (if recorded separately)
            # For example, if your code logs time in mat_data['inference_time'] or
            # if you measure it externally, adapt accordingly:
            if 'inference_time' in mat_data:
                time_pretrained.append(float(mat_data['inference_time'].squeeze()))
            else:
                # Otherwise, do a placeholder or measure externally
                pass

    # ----------------------------------------------------------------------------
    # (2) Load results for VARPRO
    # ----------------------------------------------------------------------------
    if varpro_folder is not None:
        varpro_files = glob.glob(os.path.join(varpro_folder, '*.mat'))
        for mat_path in varpro_files:
            mat_data = sio.loadmat(mat_path)
            if 'predictions' not in mat_data:
                continue

            pred = mat_data['predictions']
            # Suppose you do the same approach for ground truth
            # ...
            # Example skeleton:
            if gt_folder is not None:
                base_name = os.path.basename(mat_path)
                gt_path = os.path.join(gt_folder, base_name)
                if os.path.exists(gt_path):
                    gt_data = sio.loadmat(gt_path)
                    if 'c_tissue_gt' in gt_data:
                        c_tissue_gt = gt_data['c_tissue_gt']
                        for i in range(len(c_tissue_gt)):
                            nrmse_val = compute_nrmse(c_tissue_gt[i], pred[i])
                            nrmse_varpro.append(nrmse_val)
            if 'fitting_time' in mat_data:
                time_varpro.append(float(mat_data['fitting_time'].squeeze()))

    # ----------------------------------------------------------------------------
    # (3) Load results for NLLS
    # ----------------------------------------------------------------------------
    if nlls_folder is not None:
        nlls_files = glob.glob(os.path.join(nlls_folder, '*.mat'))
        for mat_path in nlls_files:
            mat_data = sio.loadmat(mat_path)
            if 'predictions' not in mat_data:
                continue

            pred = mat_data['predictions']
            # Suppose you do the same approach for ground truth
            # ...
            if gt_folder is not None:
                base_name = os.path.basename(mat_path)
                gt_path = os.path.join(gt_folder, base_name)
                if os.path.exists(gt_path):
                    gt_data = sio.loadmat(gt_path)
                    if 'c_tissue_gt' in gt_data:
                        c_tissue_gt = gt_data['c_tissue_gt']
                        for i in range(len(c_tissue_gt)):
                            nrmse_val = compute_nrmse(c_tissue_gt[i], pred[i])
                            nrmse_nlls.append(nrmse_val)
            if 'fitting_time' in mat_data:
                time_nlls.append(float(mat_data['fitting_time'].squeeze()))

    # Convert to numpy arrays for convenience
    nrmse_pretrained = np.array(nrmse_pretrained)
    nrmse_varpro     = np.array(nrmse_varpro)
    nrmse_nlls       = np.array(nrmse_nlls)

    time_pretrained = np.array(time_pretrained) if len(time_pretrained) else np.array([1])  # fallback
    time_varpro     = np.array(time_varpro)     if len(time_varpro) else np.array([10])    # fallback
    time_nlls       = np.array(time_nlls)       if len(time_nlls) else np.array([100])     # fallback

    # ----------------------------------------------------------------------------
    # (4) Plot NRMSE (box plot) and fitting time (bar plot)
    # ----------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left: Box plot for NRMSE
    data_for_box = []
    labels_for_box = []
    if len(nrmse_pretrained):
        data_for_box.append(nrmse_pretrained)
        labels_for_box.append("Pre-trained")
    if len(nrmse_varpro):
        data_for_box.append(nrmse_varpro)
        labels_for_box.append("VARPRO")
    if len(nrmse_nlls):
        data_for_box.append(nrmse_nlls)
        labels_for_box.append("NLLS")

    box = axes[0].boxplot(data_for_box, patch_artist=True, labels=labels_for_box)
    axes[0].set_ylabel("NRMSE")

    # Optionally mark the mean with an 'X'
    for median in box['medians']:
        median.set(color='black', linewidth=1.5)
    for mean_idx, dataset in enumerate(data_for_box, start=1):
        mean_val = np.mean(dataset)
        axes[0].plot(mean_idx, mean_val, 'x', color='white', markersize=10, markeredgewidth=2)

    # Right: Bar plot of average time (on log scale, if desired)
    mean_times = []
    bar_labels = []
    if len(time_pretrained):
        mean_times.append(np.mean(time_pretrained))
        bar_labels.append("Pre-trained")
    if len(time_varpro):
        mean_times.append(np.mean(time_varpro))
        bar_labels.append("VARPRO")
    if len(time_nlls):
        mean_times.append(np.mean(time_nlls))
        bar_labels.append("NLLS")

    axes[1].bar(bar_labels, mean_times)
    axes[1].set_yscale("log")  # comment this out if you prefer linear scale
    axes[1].set_ylabel("Fitting/Inference Time (s)")

    # ----------------------------------------------------------------------------
    # (5) Mark significance among methods (pairwise)
    # ----------------------------------------------------------------------------
    # If you only have two methods, just do a single test. For three, do pairwise.
    # Example: do t-tests for each pair
    if len(data_for_box) >= 2:
        # Indices: 0 -> Pre-trained, 1 -> VARPRO, 2 -> NLLS
        # Do pairwise significance
        # We only demonstrate if each group has data:
        x_positions = np.arange(1, len(data_for_box) + 1)

        # A simple function to place significance text above the boxes
        def annotate_significance(ax, x1, x2, y, p_val):
            """
            Draw a bracket and place significance text between boxes at x1 and x2.
            """
            ax.plot([x1, x1, x2, x2], [y, y+0.001, y+0.001, y], color='k', linewidth=1.2)
            ax.text((x1+x2)*0.5, y+0.002, significance_stars(p_val), ha='center', va='bottom', color='k')

        # We'll place each significance bracket slightly above the max NRMSE
        y_max = max(np.concatenate(data_for_box))
        height_step = 0.005 * y_max  # spacing for each bracket

        current_height = y_max + height_step

        # Example pairwise comparisons:
        if len(data_for_box) == 3:
            # Pre-trained vs VARPRO
            p_12 = ttest_ind(data_for_box[0], data_for_box[1], equal_var=False).pvalue
            annotate_significance(axes[0], 1, 2, current_height, p_12)
            current_height += height_step

            # Pre-trained vs NLLS
            p_13 = ttest_ind(data_for_box[0], data_for_box[2], equal_var=False).pvalue
            annotate_significance(axes[0], 1, 3, current_height, p_13)
            current_height += height_step

            # VARPRO vs NLLS
            p_23 = ttest_ind(data_for_box[1], data_for_box[2], equal_var=False).pvalue
            annotate_significance(axes[0], 2, 3, current_height, p_23)
            current_height += height_step

        elif len(data_for_box) == 2:
            # Just compare data_for_box[0] and data_for_box[1]
            p_12 = ttest_ind(data_for_box[0], data_for_box[1], equal_var=False).pvalue
            annotate_significance(axes[0], 1, 2, current_height, p_12)

    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=200)
    plt.show()

    print(f"Evaluation figure saved to: {save_fig_path}")
    print("Done.")


# -----------------------------------------------------------------------------
# USAGE EXAMPLE (pseudocode)
# -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     # Suppose you have run your inference for each method and saved .mat results
#     # to separate folders: 'pretrained_out', 'varpro_out', 'nlls_out'.
#     # And your ground-truth curves are in 'gt_folder'.
#     evaluate(
#         pretrained_folder="path/to/pretrained_out",
#         varpro_folder="path/to/varpro_out",
#         nlls_folder="path/to/nlls_out",
#         gt_folder="path/to/gt_folder",
#         dt=1.0,
#         save_fig_path="evaluation_results.png"
#     )
