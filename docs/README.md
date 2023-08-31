## Predicting Ovarian Cancer Treatment Response in Histopathology using Hierarchical Vision Transformers and Multiple Instance Learning 
<img src="CISTIB logo.png" align="right" width="240"/>

*HIPT-ABMIL is a transformer-based approach to classifying histopathology slides which leverages spatial information for better prognostication.* 

<img src="HIPT-AMBIL-ModelDiagram-Background-min.png" align="centre" width="900"/>

This repo was made as part of an entry to the MICCAI 2023 challenge [*automated prediction of treatment effectiveness in ovarian cancer using histopathological images* (ATEC23)](https://github.com/cwwang1979/MICCAI_ATEC23challenge). The training data was made available through [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=83593077).

HIPT-CLAM is a multiple instance learning (MIL) approach in which features are extracted from 4096x4096 pixel regions using the pretrained hierarchical transformer model [HIPT_4K](https://github.com/mahmoodlab/HIPT) and these features are aggregated to generate a slide-level representation using the attention-based multiple instance learning (ABMIL) approach [CLAM](https://github.com/mahmoodlab/CLAM). 


## Code Runs
The following code was used in producing the results submitted as part of the ATEC23 challenge.

<details>
<summary>
Data acquisition
</summary>
Before running any code, we downloaded the training data from TCIA, and turned the single-level svs files into multi-level (pyramidal) svs files using libvips. Some level of compression was necessary here to reduce file sizes, though we found compression Q90 indistinguishable from uncompressed images. Single-slide example:
  
``` shell
vips tiffsave "I:\treatment_data\2-1613704B.svs" "I:\treatment_data\pyramid_jpeg90compress\2-1613704B.svs" --compression jpeg --Q 90 --tile --pyramid
```
</details>

<details>
<summary>
Tissue region extraction
</summary>
We segmented tissue using Otsu thresholding and extracted non-overlapping 4096x4096 tissue regions:
  
``` shell
python create_patches_fp.py --source "../mount_i/treatment_data/pyramid_jpeg90compress" --save_dir "../mount_outputs/extracted_mag20x_patch4096_fp_updated" --patch_size 4096 --step_size 4096 --seg --patch --stitch --sthresh 15 --mthresh 5 --use_otsu --closing 100
``` 
</details>

<details>
<summary>
Feature extraction
</summary>
We extracted [1,192] features from each 4096x4096 region using HIPT_4K:
  
``` shell
python extract_features_fp.py --use_transforms 'HIPT' --model_type 'HIPT_4K' --data_h5_dir "../mount_outputs/extracted_mag20x_Q90_patch4096_fp_updated" --data_slide_dir "../mount_i/treatment_data/pyramid_jpeg90compress" --csv_path "dataset_csv/set_treatment.csv" --feat_dir "../mount_outputs/features/treatment_Q90_hipt4096_features_normalised_updatedsegmentation" --batch_size 1 --slide_ext .svs 
```
</details>

<details>
<summary>
Hyperparameter tuning
</summary>
Grid tuning was performed using RayTune with hyperparameter options defined within main.py. This example is from tuning fold 0 of the 5-fold cross-validation using HIPT-ABMIL: 
  
``` shell
python main.py --tuning --hardware DGX --tuning_output_file /mnt/results/tuning_results/main_treatment_Q90_betterseg_patience30mineverloss_3reps_noaugs_DGX_moreoptions_fold0.csv --num_tuning_experiments 3 --data_slide_dir "/mnt/data/ATEC_jpeg90compress" --min_epochs 0 --early_stopping --split_dir "treatment_5fold_100" --k 1 --results_dir /mnt/results --exp_code treatment_HIPTnormalised_Q90_betterseg_patience30mineverloss_3reps_noaugs_tuning_moreoptions_fold0 --subtyping --weighted_sample --bag_loss ce --task treatment --max_epochs 200 --model_type clam_sb --no_inst_cluster --log_data --csv_path 'dataset_csv/set_treatment.csv' --data_root_dir "/mnt/data" --features_folder treatment_Q90_hipt4096_features_normalised_updatedsegmentation
```
</details>

<details>
<summary>
Model training
</summary>
The best model from the 5-fold cross-validation experiment (as judged by averaged validation set cross-entropy loss across three repeats and five folds) was trained:
  
``` shell
python main.py --hardware DGX --max_patches_per_slide 15 --data_slide_dir "/mnt/data/ATEC_jpeg90compress" --min_epochs 0 --early_stopping --drop_out 0.0 --lr 0.0005 --reg 0.0001 --model_size hipt_smaller --split_dir "treatment_5fold_100" --k 5 --results_dir /mnt/results --exp_code treatment_HIPTnormalised_Q90_betterseg_15patches_drop0lr0005reg0001_modelhiptsmaller_ABMILsb_ce_20x_5fold_noaugs_bestfromsecondbigtuning --subtyping --weighted_sample --bag_loss ce --task treatment --max_epochs 1000 --model_type clam_sb --no_inst_cluster --csv_path 'dataset_csv/set_treatment.csv' --data_root_dir "/mnt/data" --features_folder treatment_Q90_hipt4096_features_normalised_updatedsegmentation
```
</details>

<details>
<summary>
Model evaluation
</summary>
The model was evaluated on the test sets of the five-fold cross validation with 100,000 iterations of bootstrapping:
  
``` shell
python eval.py --drop_out 0.0 --model_size hipt_smaller --models_exp_code treatment_HIPTnormalised_Q90_betterseg_15patches_drop0lr0005reg0001_modelhiptsmaller_ABMILsb_ce_20x_5fold_noaugs_bestfromsecondbigtuning_s1 --save_exp_code treatment_HIPTnormalised_Q90_betterseg_15patches_drop0lr0005reg0001_modelhiptsmaller_ABMILsb_ce_20x_5fold_noaugs_bestfromsecondbigtuning_bootstrapping --task treatment --model_type clam_sb --results_dir /mnt/results --data_root_dir "/mnt/data" --k 5 --features_folder "treatment_Q90_hipt4096_features_normalised_updatedsegmentation" --csv_path 'dataset_csv/set_treatment.csv' 
python bootstrapping.py --num_classes 2 --model_names  treatment_HIPTnormalised_Q90_betterseg_15patches_drop0lr0005reg0001_modelhiptsmaller_ABMILsb_ce_20x_5fold_noaugs_bestfromsecondbigtuning_bootstrapping --bootstraps 100000 --run_repeats 1 --folds 5
```

The cross-validation results for this optimal HIPT-ABMIL model were as follows:

``` shell
 Confusion Matrix:
 [[ 76  49]
 [ 29 128]]

 average ce loss:  0.4858174402095372 (not bootstrapped)
 AUC mean:  [0.8206680412411297]  AUC std:  [0.02530094639907452]
 F1 mean:  [0.7659177381223935]  F1 std:  [0.02579712919409385]
 accuracy mean:  [0.7234604255319149]  accuracy std:  [0.02667653193254119]
 balanced accuracy mean:  [0.7117468943178861]  balanced accuracy std:  [0.026864606981070703]
```
</details>

## Reference
This code is an extension of our [previous repository](), which itself was forked from the [CLAM repository](https://github.com/mahmoodlab/CLAM) with corresponding [paper](https://www.nature.com/articles/s41551-020-00682-w). Code is also used from the [HIPT repository](https://github.com/mahmoodlab/HIPT), including pretrained model weights. This repository and the original CLAM repository are both available for non-commercial academic purposes under the GPLv3 License.
