python main.py --mode infer --model Transformer --infer_batch 2048 --input_folder "/mnt/LiDXXLab_Files/Chaowei/Low-dose-Study/AI_Fitting/Input_extTofts_fitkep_wholePan/" --output_folder "./output_mat/Output_extTofts_fitkep_wholePan_biasCorr_oldmodel/" --model_path "/hdd1/chaowei/checkpoints/Simulation_Fitting/8f804f4d6f7c4c1a937a45dd3674689b/epoch0090.pth" --subject_mean parameter --bias_corr True

# python main.py --mode infer --model Transformer --infer_batch 2048 --input_folder "/mnt/LiDXXLab_Files/Chaowei/Low-dose-Study/AI_Fitting/Input_extTofts_fitkep_wholePan/" --output_folder "./output_mat/Output_extTofts_fitkep_wholePan_oldmodel/" --model_path "/hdd1/chaowei/checkpoints/Simulation_Fitting/8f804f4d6f7c4c1a937a45dd3674689b/epoch0090.pth" --subject_mean parameter

# python main.py --mode infer --model Transformer --infer_batch 2048 --input_folder "/mnt/LiDXXLab_Files/Chaowei/Low-dose-Study/AI_Fitting/Input_extTofts_fitkep_wholePan/" --output_folder "./output_mat/Output_extTofts_fitkep_wholePan_mCA_oldmodel/" --model_path "/hdd1/chaowei/checkpoints/Simulation_Fitting/8f804f4d6f7c4c1a937a45dd3674689b/epoch0090.pth" --subject_mean curve

# python main.py --mode infer --model Transformer --infer_batch 2048 --input_folder "/mnt/LiDXXLab_Files/Chaowei/Low-dose-Study/AI_Fitting/Input_extTofts_fitkep/" --output_folder "./output_mat/Output_extTofts_fitkep_mCA_oldmodel/" --model_path "/hdd1/chaowei/checkpoints/Simulation_Fitting/8f804f4d6f7c4c1a937a45dd3674689b/epoch0090.pth" --subject_mean curve

# python main.py --mode infer --model FC --infer_batch 2048 --input_folder "/mnt/LiDXXLab_Files/Chaowei/Low-dose-Study/AI_Fitting/Input_extTofts_fitkep/" --output_folder "./output_mat/Output_extTofts_fitkep_mCA_newmodel/" --model_path "/hdd1/chaowei/checkpoints/Simulation_Fitting/b8a4b89fee374dc7a69f430eb5f64201/epoch0093.pth" --subject_mean curve


# python main.py --mode infer --model FC --infer_batch 2048 --input_folder "/mnt/LiDXXLab_Files/Chaowei/Low-dose-Study/AI_Fitting/Input_extTofts_fitkep/" --output_folder "./output_mat/Output_extTofts_fitkep_newmodel/" --model_path "/hdd1/chaowei/checkpoints/Simulation_Fitting/b8a4b89fee374dc7a69f430eb5f64201/epoch0093.pth" --subject_mean parameters

# python main.py --mode infer --model FC --infer_batch 2048 --input_folder "/mnt/LiDXXLab_Files/Chaowei/Low-dose-Study/AI_Fitting/Input_extTofts_fitkep_wholePan/" --output_folder "./output_mat/Output_extTofts_fitkep_wholePan/" --model_path "/hdd1/chaowei/checkpoints/Simulation_Fitting/b8a4b89fee374dc7a69f430eb5f64201/epoch0093.pth" --subject_mean parameters

# python main.py --mode infer --model FC --infer_batch 2048 --input_folder "/mnt/LiDXXLab_Files/Chaowei/Low-dose-Study/AI_Fitting/Input_extTofts_fitkep_wholePan/" --output_folder "./output_mat/Output_extTofts_fitkep_wholePan_Curve/" --model_path "/hdd1/chaowei/checkpoints/Simulation_Fitting/b8a4b89fee374dc7a69f430eb5f64201/epoch0093.pth" --subject_mean curve


# python main.py --mode infer --model FC --infer_batch 2048 --input_folder "/mnt/LiDXXLab_Files/Chaowei/Low-dose-Study/AI_Fitting/Input_extTofts_fitkep_wholePan/" --output_folder "./output_mat/Output_extTofts_fitkep_wholePan/" --model_path "/hdd1/chaowei/checkpoints/Simulation_Fitting/8f804f4d6f7c4c1a937a45dd3674689b/epoch0021.pth" --subject_mean parameters
