#!/bin/bash
#SBATCH --job-name=DCEQuantification
#SBATCH --output=logs/%j.log
#SBATCH --error=errors/%j.log
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mail-user=chaowei.wu@cshs.org
#SBATCH --mail-type=END,FAIL


source activate dce

export CWROOT=/common/lidxxlab/chaowei/Torch-RoQ-DCE2


echo "Detailed GPU Information:"
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

wandb login b893a6d433af71e3cdb71002caa7dc90d57d4587

cd $CWROOT
python main.py \
       --mode infer \
       --model Transformer \
       --infer_batch 256 \
       --model_path $CWROOT/checkpoints/Simulation_Fitting/cg3hinwn/epoch0048.pth \
       --input_folder $CWROOT/Input_extTofts_fitkep_wholePan/ \
       --output_folder $CWROOT/output_mat/Output_extTofts_fitkep_wholePan/ \
       --subject_mean parameter 

python main.py \
       --mode infer \
       --model Transformer \
       --infer_batch 256 \
       --model_path $CWROOT/checkpoints/Simulation_Fitting/cg3hinwn/epoch0048.pth \
       --input_folder $CWROOT/Input_extTofts_fitkep/ \
       --output_folder $CWROOT/output_mat/Output_extTofts_fitkep/ \
       --subject_mean parameter 

# python main.py \
#        --mode infer \
#        --model Transformer \
#        --infer_batch 256 \
#        --model_path $CWROOT/checkpoints/Simulation_Fitting/cg3hinwn/epoch0048.pth \
#        --input_folder $CWROOT/Input_extTofts_fitkep/ \
#        --output_folder $CWROOT/output_mat/Output_extTofts_fitkep_curve/ \
#        --subject_mean curve 

# python main.py \
#        --mode infer \
#        --model Transformer \
#        --infer_batch 256 \
#        --model_path $CWROOT/checkpoints/dl01/epoch0090.pth \
#        --input_folder $CWROOT/Input_extTofts_fitkep/ \
#        --output_folder $CWROOT/output_mat/Output_extTofts_fitkep_dl01model/ \
#        --subject_mean parameter 

# python main.py \
#        --mode infer \
#        --model Transformer \
#        --infer_batch 256 \
#        --model_path $CWROOT/checkpoints/dl01/epoch0090.pth \
#        --input_folder $CWROOT/Input_extTofts_fitkep/ \
#        --output_folder $CWROOT/output_mat/Output_extTofts_fitkep_curve_dl01model/ \
#        --subject_mean curve 