import time
import os
import numpy as np
import torch
import torch.optim as optim
import wandb  # **REPLACED mlflow with wandb**
from dataset import SimulationDataset
from torch.utils.data import DataLoader,Dataset
from models import *
from loss import *
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# Main training function
# =============================================================================
def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets and dataloaders
    train_dataset = SimulationDataset(
        args.input_folder, args.sim_size, args.length,
        args.out_dim, args.dt, args.sim_dist,
        [args.max_vp, args.max_ktrans, args.max_ve, args.max_kep]
    )
    val_dataset = SimulationDataset(
        args.input_folder, args.val_size, args.length,
        args.out_dim, args.dt, args.sim_dist,
        [args.max_vp, args.max_ktrans, args.max_ve, args.max_kep]
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model, loss, optimizer, scheduler
    if args.model == "FC":
        model = DCEModelFC(in_dim=args.length, out_dim=args.out_dim, layer_num=args.layer_num).to(device)
    elif args.model == "CNN":
        model = DCEModelCNN(dce_dim=args.length, out_dim=args.out_dim).to(device)
    elif args.model == "Transformer":
        model = DCEModelTransformer(dce_dim=args.length, out_dim=args.out_dim, num_layers=args.layer_num).to(device)
    else:
        raise ValueError("Unknown Model Type. Choose from 'FC', 'CNN', or 'Transformer'.")

    # Loss functions
    if args.loss == "mixedV1":
        criterion = MixedLoss(alpha=args.alpha)
    elif args.loss == "mixedV2":
        criterion = MixedLossV2(
            alpha=args.alpha,
            penalty_weight=args.penalty,
            min_bound=[0, 0, 0.1, 0],
            max_bound=[args.max_vp, args.max_ktrans, args.max_ve, args.max_kep]
        )
    elif args.loss == "MPAE":
        criterion = MPAE()
    else:
        raise ValueError("Unknown Loss. Choose from 'mixedV1', 'mixedV2', or 'MPAE'.")
    criterion_mape = MPAE()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    # Load evaluation data
    test_data = np.load(args.eval_data)
    tissue_test = torch.tensor(test_data["tissue"], dtype=torch.float32).to(device)
    aif_test = torch.tensor(test_data["aif"], dtype=torch.float32).to(device)
    kv_true = torch.tensor(test_data["kv"], dtype=torch.float32).to(device)
    test_input = torch.cat([tissue_test, aif_test], dim=1)

    # Setup Weights & Biases
    wandb.init(config=vars(args), project=getattr(args, 'project_name', 'DCE_Training'))  # **INIT W&B instead of mlflow.start_run**
    config = wandb.config  # **LOG params loaded via config**
    wandb.watch(model, log="all")  # **TRACK model**
    run_id = wandb.run.id  # **RETRIEVE run ID**

    best_val_loss = float('inf')
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.prefix, run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for i, (tissue, aif, kv) in enumerate(train_loader, 1):
            tissue, aif, kv = tissue.to(device), aif.to(device), kv.to(device)
            optimizer.zero_grad()
            outputs = model(tissue, aif)
            loss = criterion(outputs, kv)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch [{epoch}/{args.epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch} completed in {elapsed:.2f}s, Average Loss: {avg_loss:.4f}")

        # Log training metrics
        wandb.log({"train_loss": avg_loss}, step=epoch)  # **REPLACED mlflow.log_metric**

        # Validation
        model.eval()
        val_loss, val_loss_mape = 0.0, 0.0
        with torch.no_grad():
            for i, (tissue, aif, kv) in enumerate(val_loader, 1):
                tissue, aif, kv = tissue.to(device), aif.to(device), kv.to(device)
                outputs = model(tissue, aif)
                val_loss += criterion(outputs, kv).item()
                val_loss_mape += criterion_mape(outputs, kv).item()
        avg_val_loss = val_loss / len(val_loader)
        avg_val_loss_mape = val_loss_mape / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation MAPE: {avg_val_loss_mape:.4f}")

        # Log validation metrics
        wandb.log({"val_loss": avg_val_loss, "val_mape": avg_val_loss_mape}, step=epoch)  # **REPLACED mlflow.log_metric**

        # Save checkpoint if improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch{epoch:04d}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)  # **SAVE checkpoint with W&B**
            print(f"Saved checkpoint: {checkpoint_path}")

        # Scheduler step
        if epoch % 10 == 0:
            scheduler.step()
            print(f"Learning rate decayed to: {scheduler.get_last_lr()[0]:.6f}")

    # Evaluation and artifacts
    wrapped_model = WrappedDCEModel(model, args.length)
    wrapped_model.eval()
    wrapped_model = wrapped_model.cpu()
    kv_pred = wrapped_model(test_input)

    kv_pred_np = kv_pred.cpu().numpy()
    kv_true_np = kv_true.cpu().numpy()

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    param_names = ["vp", "ktrans", "ve", "kep"]
    for p, ax in enumerate(axes):
        bland_altman_plot(ax, kv_pred_np[:, p], kv_true_np[:, p], param_names[p])

    eval_plot_path = os.path.join(args.eval_result, f"evaluation_plot_{run_id}.png")
    plt.savefig(eval_plot_path)

    # Log plot as artifact
    plot_artifact = wandb.Artifact('evaluation_plots', type='plot')  # **REPLACED mlflow.log_artifact**
    plot_artifact.add_file(eval_plot_path)
    wandb.log_artifact(plot_artifact)

    # Log final model
    model_path = os.path.join(checkpoint_dir, 'wrapped_model.pth')
    torch.save(wrapped_model.state_dict(), model_path)
    model_artifact = wandb.Artifact('trained_model', type='model')  # **REPLACED mlflow.pytorch.log_model**
    model_artifact.add_file(model_path)
    wandb.log_artifact(model_artifact)

    wandb.finish()  # **REPLACED mlflow.end_run**

    # Optionally run inference
    if getattr(args, 'infer', False):
        run_inference(model, device, args.input_folder, args.output_folder, args.length, args.infer_batch)
