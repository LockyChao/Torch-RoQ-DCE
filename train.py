import time
import torch
import torch.optim as optim
import mlflow
import mlflow.pytorch
from dataset import SimulationDataset
from torch.utils.data import Dataset, DataLoader
from models import *
from loss import *
import os
# =============================================================================
# Main training function
# =============================================================================
def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets and dataloaders
    train_dataset = SimulationDataset(args.input_folder,args.sim_size, args.length, args.out_dim, args.dt,args.sim_dist,[args.max_vp,args.max_ktrans,args.max_ve,args.max_kep] )
    val_dataset = SimulationDataset(args.input_folder,args.val_size, args.length, args.out_dim, args.dt, args.sim_dist,[args.max_vp,args.max_ktrans,args.max_ve,args.max_kep])
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

    if args.loss == "mixedV1":
        criterion = MixedLoss(alpha=args.alpha)
    elif args.loss == "mixedV2":
        criterion = MixedLossV2(alpha=args.alpha, penalty_weight=args.penalty, min_bound=[0,0,0.1,0], max_bound=[args.max_vp,args.max_ktrans,args.max_ve,args.max_kep]) 
    elif args.loss == "MPAE":
        criterion = MPAE()
    else: 
        raise "Unknown Loss."
    criterion_mape = MPAE()
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
        val_loss_mape = 0.0
        with torch.no_grad():
            for tissue, aif, kv in val_loader:
                tissue = tissue.to(device)
                aif = aif.to(device)
                kv = kv.to(device)
                outputs = model(tissue, aif)
                loss = criterion(outputs, kv)
                loss_mape = criterion_mape(outputs,kv)
                val_loss += loss.item()
                val_loss_mape+=loss_mape.item()
        avg_val_loss = val_loss / len(val_loader)
        avg_val_loss_mape = val_loss_mape / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation MAPE: {avg_val_loss_mape:.4f}")
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

        mlflow.log_metric("val_mape", avg_val_loss_mape, step=epoch)

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
