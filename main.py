import argparse
from train import train
from inference import infer,run_inference
from evaluation import evaluate
# from evaluation import generate_eval
import torch
# =============================================================================
# Command-line arguments
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train DCE model in PyTorch with MLFlow tracking")
    parser.add_argument("--mode", type=str, default='train', help="train or generate_eval or infer")
    parser.add_argument("--model", type=str, default='FC', help="choice of DL model: FC, CNN, Transformer")
    parser.add_argument("--loss", type=str, default='mixedV1', help="choice of loss, MPAE, mixedV1 or mixedV2")
    parser.add_argument("--penalty", type=float, default=1, help="penalty for loss term in mixedV2")
    parser.add_argument("--length", type=int, default=210, help="Length of input DCE signal")
    parser.add_argument("--dt", type=float, default=2.0608, help="Delta t of DCE curve, Ng*TR")
    parser.add_argument("--out_dim", type=int, default=4, help="Output dimension (KV parameters)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--layer_num", type=int, default=4, help="Number of dense layers in feature extractor")
    parser.add_argument("--sim_size", type=int, default=int(1e6), help="Number of simulated training samples")
    parser.add_argument("--sim_dist", type=str, default="uniform", help="Normal or Uniform distributed")
    parser.add_argument("--max_vp", type=float, default=0.8, help="Maximum vp simulated")
    parser.add_argument("--max_ktrans", type=float, default=0.2, help="Maximum ktrans simulated")
    parser.add_argument("--max_kep", type=float, default=0.6, help="Maximum kep simulated")
    parser.add_argument("--max_ve", type=float, default=0.8, help="Maximum ve simulated")
    parser.add_argument("--val_size", type=int, default=int(1e4), help="Number of simulated validation samples")
    parser.add_argument("--alpha", type=float, default=0.998, help="Alpha weight for MixedLoss")
    parser.add_argument("--checkpoint_dir", type=str, default="/hdd1/chaowei/checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--prefix", type=str, default="Simulation_Fitting", help="Prefix for log/checkpoint names")
    parser.add_argument("--eval_data", type=str, default="/hdd1/chaowei/test_new.npz", help="Where is the eval data")
    parser.add_argument("--eval_result", type=str, default="./artifacts", help="Where is the eval result saved")
    # Inference options
    parser.add_argument("--infer", action="store_true", help="Run inference on .mat files after training")
    parser.add_argument("--input_folder", type=str, default=None, help="Folder with .mat files for inference")
    parser.add_argument("--model_path", type=str, default=None, help="path of trained model")
    parser.add_argument("--output_folder", type=str, default="./output_mat", help="Folder to save inference results")
    parser.add_argument("--infer_batch", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--LDAIF", type=str, default='individual', help="common AIF or individual AIF for LD data")
    parser.add_argument("--subject_mean", type=str, default='parameter', help="subject-wise parameter or curve")
    parser.add_argument("--bias_corr", type=bool, default='False', help="bias correction")

    
    return parser.parse_args()

# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    args = parse_args()
    if args.mode=='generate_eval':
        # generate_eval(args)
        pass
    elif args.mode=='infer':
        infer(args)
    elif args.mode == "evaluate":
        evaluate(args)
    else:
        train(args)