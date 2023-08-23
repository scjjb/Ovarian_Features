from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *
import cProfile, pstats
from datasets.dataset_h5 import Dataset_All_Bags

from functools import partial
from ray import tune
from ray.air.config import RunConfig
import ray

#from streamlit import legacy_caching as caching

# Evaluation settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--csv_path', type=str, default=None, help='path to dataset_csv file')
parser.add_argument('--coords_path', type=str, default=None,help='path to coords pt files if needed')
parser.add_argument('--pretraining_dataset',type=str,choices=['ImageNet','Histo'],default='ImageNet')
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--eval_features',default=False, action='store_true',help='extract features during sampling')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='directory containing features folders')
parser.add_argument('--features_folder', type=str, default=None,
                    help='folder within data_root_dir containing the features - must contain pt_files/h5_files subfolder')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big','hipt_small','hipt_medium','hipt_big','hipt_smaller'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout p=0.25')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['ovarian_5class','ovarian_1vsall','nsclc','treatment'])
parser.add_argument('--profile', action='store_true', default=False, 
                    help='show profile of longest running code sections')
parser.add_argument('--profile_rows', type=int, default=10, help='number of rows to show from profiler (requires --profile to show any)')
parser.add_argument('--label_dict',type=str,help='Convert labels to numbers')
parser.add_argument('--cpu_only',action='store_true',default=False,help='Use CPU only')

## Sampling args
parser.add_argument('--sampling', action='store_true', default=False, help='sampling for faster evaluation')
parser.add_argument('--sampling_type', type=str, choices=['spatial','textural'],default='spatial',help='type of sampling to use')
parser.add_argument('--samples_per_iteration', type=int, default=100, help='number of patches to sample per sampling iteration')
parser.add_argument('--resampling_iterations', type=int, default=10, help='number of resampling iterations (not including the initial sample)')
parser.add_argument('--sampling_random', type=float, default=0.2, help='proportion of samples which are completely random per iteration')
parser.add_argument('--sampling_random_delta',type=float, default=0.02, help='reduction in sampling_random per iteration')
parser.add_argument('--sampling_neighbors', type=int, default=20, help='number of nearest neighbors to consider when resampling')
parser.add_argument('--sampling_neighbors_delta', type=int, default=0, help='reduction in number of nearest neighbors per iteration')
parser.add_argument('--texture_model',type=str, choices=['resnet50','levit_128s'], default='resnet50',help='model to use for feature extraction in textural sampling')
parser.add_argument('--plot_sampling',action='store_true',default=False,help='Save an image showing the samples taken at each at last iteration')
parser.add_argument('--plot_sampling_gif',action='store_true',default=False,help='Save a gif showing the evolution of the samples taken')
parser.add_argument('--plot_weighting',action='store_true',default=False,help='Save an image showing each patches sample weighting in the final iteration')
parser.add_argument('--plot_weighting_gif',action='store_true',default=False,help='Save a gif showing the evolution of sample weighting')
parser.add_argument('--use_all_samples',action='store_true',default=False,help='Use every previous sample for final iteration')
parser.add_argument('--final_sample_size',type=int,default=100,help='number of patches for final iteration sample')
parser.add_argument('--retain_best_samples',type=int,default=100,help='number of highest-attention previous samples to retain for final iteration sample')
parser.add_argument('--initial_grid_sample',action='store_true',default=False,help='Take the initial sample to be spaced out in a grid')
parser.add_argument('--sampling_average',action='store_true',default=False,help='Take the sampling weights as averages rather than maxima to leverage more learned information')
parser.add_argument('--weight_smoothing',type=float,default=0.15,help='Power applied to attention scores to generate sampling weights')
parser.add_argument('--fully_random',action='store_true', default=False, help='Take entirely random samples (no active sampling)')

## tuning options
parser.add_argument('--tuning', action='store_true', default=False, help='run hyperparameter tuning')
parser.add_argument('--tuning_output_file',type=str,default="tuning_results/tuning_output.csv",help="where to save tuning outputs")
parser.add_argument('--num_tuning_experiments',type=int,default=100,help="number of tuning experiments")
parser.add_argument('--same_slide_repeats',type=int,default=1,help='number of times to repeat evaluating each slide to help account for randomness')
parser.add_argument('--hardware',type=str, choices=['DGX','PC'], default='DGX',help='sets amount of CPU and GPU to use per experiment')
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.cpu_only:
    torch.cuda.is_available = lambda : False
    device=torch.device("cpu")

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)

if args.task == 'ovarian_5class':
    args.n_classes=5
    args.label_dict = {'high_grade':0,'low_grade':1,'clear_cell':2,'endometrioid':3,'mucinous':4}

elif args.task == 'ovarian_1vsall':
    args.n_classes=2
    args.label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1}

elif args.task == 'nsclc':
    args.n_classes=2
    args.label_dict = {'luad':0,'lusc':1}

elif args.task =='treatment':
    args.n_classes=2
    args.label_dict = {'invalid':0,'effective':1}

else:
    raise NotImplementedError

dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                                coords_path=args.coords_path,
                                data_dir= os.path.join(args.data_root_dir, args.features_folder),
                                shuffle = False,
                                print_info = True,
                                label_dict = args.label_dict,
                                patient_strat=False,
                                ignore=[])

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

def count_patches(dataset,args,ckpt):
    loader = get_simple_loader(dataset)
    patches=0
    for batch_idx, (data, label,coords,slide_id) in enumerate(loader):
        patches=patches+len(data)
        print(len(data))
        print(data)
        assert 1==2,"testing"
    return patches


def main():
    all_results = []
    all_auc = []
    all_acc = []
    all_loss = []

    if args.tuning:
        ray.init(num_gpus=1)
                             
        if args.hardware=='DGX':
            hardware={"cpu":10,"gpu":0.07}
        else:
            hardware={"cpu":2,"gpu":0.5}
    
        search_space = {
            "weight_smoothing":  tune.loguniform(0.2,0.20000001),
            "resampling_iterations": tune.choice([2,4,6,8,10,12,16]),
            "sampling_neighbors": tune.choice([4,8,16,32,48,64]),
            "sampling_random": tune.uniform(0.00,0.75),
            "sampling_random_delta": tune.loguniform(0.0001,0.5)
        }

        scheduler = tune.schedulers.ASHAScheduler(
            metric="auc",
            mode="max",
            grace_period=1,
            reduction_factor=3,
            max_t=1)

        reporter = tune.CLIReporter(
            metric_columns=["auc", "training_iteration"],
            max_report_frequency=5,
            max_progress_rows=20,
            metric="auc",
            mode="max",
            sort_by_metric=True)


    for ckpt_idx in range(len(ckpt_paths)):
        if args.eval_features:
            split_dataset=Dataset_All_Bags(args.csv_path)
        else:
            if datasets_id[args.split] < 0:
                split_dataset = dataset	
            else:	
                csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])	
                datasets = dataset.return_splits(from_id=False, csv_path=csv_path)	
                split_dataset = datasets[datasets_id[args.split]]
            
        if args.tuning:
            #if args.sampling:
            tuner = tune.Tuner(tune.with_resources(partial(eval,dataset=split_dataset,args=args,ckpt_path=ckpt_paths[ckpt_idx]),hardware),param_space=search_space, run_config=RunConfig(name="test_run", progress_reporter=reporter),tune_config=tune.TuneConfig(scheduler=scheduler,num_samples=args.num_tuning_experiments))
        
            results = tuner.fit()
            results_df=results.get_dataframe()
            results_df.to_csv(args.tuning_output_file,index=False)
            
            best_trial = results.get_best_result("auc", "max","all")
            print("best trial:", best_trial)
            print("Best trial config: {}".format(best_trial.config))
            print("Best trial final auc: {}".format(best_trial.metrics["auc"]))
            print("Best trial final acuracy: {}".format(best_trial.metrics["accuracy"]))

        else:
            model, patient_results, test_error, auc, df, loss = eval(None,split_dataset, args, ckpt_paths[ckpt_idx])
            #all_results.append(all_results)
            all_auc.append(auc)
            print("all auc", all_auc)
            all_acc.append(1-test_error)
            print("all acc", all_acc)
            all_loss.append(loss)
            print("all loss", all_loss)
            if not args.eval_features:
                df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)	
    if not args.tuning:
        final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc, 'loss': all_loss})	
        print(final_df)
        if len(folds) != args.k:	
            save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])	
        else:	
            save_name = 'summary.csv'	
        final_df.to_csv(os.path.join(args.save_dir, save_name))
    
if __name__ == "__main__":
    if args.profile:
        caching.clear_cache()
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        print("max gpu mem usage:",torch.cuda.max_memory_allocated())
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(args.profile_rows)
    else:
        main()
    
