from __future__ import print_function

import argparse
import os
import torch
import pandas as pd
import numpy as np

from functools import partial
from ray import tune
from ray.air.config import RunConfig
import ray
import cProfile, pstats

# internal imports
from utils.core_utils import train
from utils.core_utils_tuning import train_tuning
from utils.core_utils_sampling import train_sampling
from datasets.dataset_generic import Generic_MIL_Dataset
from utils.tuning_utils import TrialPlateauStopper

## set maximum number of raytune trials pending at once to 20
os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = "20"

def main():
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end
    
    if args.tuning:
        ray.init(num_gpus=1,runtime_env={"TUNE_MAX_PENDING_TRIALS_PG": 7})
            
        if args.hardware=='DGX':
            hardware={"cpu":32,"gpu":0.2}
        else:
            if args.task =='treatment':
                hardware={"cpu":0.8,"gpu":0.2}
            else:
                hardware={"cpu":2,"gpu":0.5}

        if args.sampling:
            if args.no_inst_cluster:
                search_space = {
                    "reg": tune.loguniform(1e-10,1e-2),
                    "drop_out": tune.uniform(0.00,0.99),
                    "lr": tune.loguniform(1e-5,1e-2),
                    "no_sample": tune.choice([0,10,20,30,40]),
                    "weight_smoothing": tune.loguniform(0.0001,0.5),
                    "resampling_iterations": tune.choice([2,4,6,8,10,12,16]),
                    "sampling_neighbors": tune.choice([4,8,16,32,64]),
                    "sampling_random": tune.uniform(0.25,0.95),
                    "sampling_random_delta": tune.loguniform(0.0001,0.5)
                    }
            else:
                search_space = {
                    "reg": tune.loguniform(1e-10,1e-2),
                    "drop_out": tune.uniform(0.00,0.99),
                    "lr": tune.loguniform(1e-5,1e-2),
                    "B": tune.choice([4,6,16,32,64]),
                    "no_sample": tune.choice([0,10,20,30,40]),
                    "weight_smoothing": tune.loguniform(0.0001,0.5),
                    "resampling_iterations": tune.choice([2,4,6,8,10,12,16]),
                    "sampling_neighbors": tune.choice([4,8,16,32,64]),
                    "sampling_random": tune.uniform(0.25,0.95),
                    "sampling_random_delta": tune.loguniform(0.0001,0.5)
                }
        else:
            if args.no_inst_cluster:
                if args.model_size in ["hipt_big","hipt_medium","hipt_small","hipt_smaller","hipt_smallest"]:
                    search_space={
                        ## HIPT-ABMIL first tuning
                        #"A_model_size": tune.grid_search(["hipt_medium","hipt_small","hipt_smaller"]),
                        #"lr": tune.grid_search([0.01,0.001,0.0001]),
                        #"patches": tune.grid_search([25,50, 75,100]),
                        #"drop_out": tune.grid_search([0.25,0.5, 0.75]),
                        #"reg": tune.grid_search([0.1, 0.01, 0.001, 0.0001]),

                        ## HIPT-ABMIL second tuning
                        "A_model_size": tune.grid_search(["hipt_small","hipt_smaller","hipt_smallest"]),
                        "lr": tune.grid_search([0.005,0.001,0.0005]),
                        "patches": tune.grid_search([15, 25, 35, 45]),
                        "drop_out": tune.grid_search([0.0, 0.2,0.4,0.6]),
                        "reg": tune.grid_search([0.001, 0.0001, 0.00001]),
                        }
                else:
                    ## first ResNet-ABMIL tuning:
                    #search_space={
                    #    "reg": tune.grid_search([0.01, 0.001, 0.0001]),
                    #    "drop_out": tune.grid_search([0.25, 0.5, 0.75]),
                    #    "lr": tune.grid_search([0.001,0.0001, 0.00001]),
                    #    "A_patches": tune.grid_search([7500, 5000, 2500 ]),
                    #    "model_size": tune.grid_search(["small","tiny","tinier"])
                    #    }

                    ## second ResNet-ABMIL tuning:
                    search_space={
                            "reg": tune.grid_search([0.001, 0.0001, 0.00001]),
                            "drop_out": tune.grid_search([0.15, 0.35, 0.55]),
                            "lr": tune.grid_search([0.005,0.001,0.0005]),
                            "A_patches": tune.grid_search([6000, 5000, 4000 ]),
                            "model_size": tune.grid_search(["small","tiny","tinier"])
                            }


            else:
                if args.model_size in ["hipt_big","hipt_medium","hipt_small","hipt_smaller","hipt_smallest"]:
                    search_space={
                            ## first HIPT-CLAM:
                            #"reg": tune.grid_search([0.1, 0.01, 0.001, 0.0001]),
                            #"drop_out": tune.grid_search([0.25, 0.5, 0.75]),
                            #"lr": tune.grid_search([0.01,0.001,0.0001]),
                            #"patches": tune.grid_search([25, 50, 75, 100]),
                            #"B": tune.grid_search([4,6,8]),
                            #"A_model_size": tune.grid_search(["hipt_medium","hipt_small","hipt_smaller"]),
                            
                            ## second HIPT-CLAM tuning:
                            "reg": tune.grid_search([0.001, 0.0001, 0.00001]),
                            "drop_out": tune.grid_search([0.0, 0.2, 0.4, 0.6]),
                            "lr": tune.grid_search([0.005,0.001,0.0005]),
                            "patches": tune.grid_search([15,25,35,45]),
                            "B": tune.grid_search([6,8,10]),
                            "A_model_size": tune.grid_search(["hipt_smaller","hipt_smallest"])
                            }
                else:
                    search_space = {
                        "reg": tune.loguniform(1e-10,1e-2),
                        "drop_out": tune.uniform(0.00,0.99),
                        "lr": tune.loguniform(1e-5,1e-2),
                        "B": tune.choice([4,6,16,32,64]),
                    }
            

        scheduler = tune.schedulers.ASHAScheduler(
            metric="loss",
            mode="min",
            grace_period=min(50,args.max_epochs),
            reduction_factor=3,
            max_t=args.max_epochs)


        reporter = tune.CLIReporter(
            metric_columns=["loss", "auc", "training_iteration"],
            max_report_frequency=5,
            max_progress_rows=20,
            metric="loss",
            mode="min",
            sort_by_metric=True)
        

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        if args.perturb:
            train_dataset.perturb_features(True)
        if args.use_augs:
            train_dataset.use_augmentations(True)
        datasets = (train_dataset, val_dataset, test_dataset)

        ##class_counts to be used in balanced cross entropy if enabled
        class_counts=0
        if args.bag_loss == 'balanced_ce':
            class_counts_train=dataset.count_by_class(csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
            class_counts_val=dataset.count_by_class(csv_path='{}/splits_{}.csv'.format(args.split_dir, i),split='val')
            class_counts=[class_counts_train[i]+class_counts_val[i] for i in range(len(class_counts_train))]

        if args.tuning:
            seed_torch(args.seed)
            stopper=TrialPlateauStopper(metric="loss",mode="min",num_results=30,grace_period=40)
            if args.sampling:
                tuner = tune.Tuner(tune.with_resources(partial(train_sampling,datasets=datasets,cur=i,class_counts=class_counts,args=args),hardware),param_space=search_space, run_config=RunConfig(name="test_run",stop=stopper, progress_reporter=reporter),tune_config=tune.TuneConfig(scheduler=scheduler,num_samples=args.num_tuning_experiments))
            else:
                tuner = tune.Tuner(tune.with_resources(partial(train_tuning,datasets=datasets,cur=i,class_counts=class_counts,args=args),hardware),param_space=search_space, run_config=RunConfig(name="test_run",stop=stopper, progress_reporter=reporter),tune_config=tune.TuneConfig(scheduler=scheduler,num_samples=args.num_tuning_experiments))
            results = tuner.fit()
            results_df=results.get_dataframe(filter_metric="loss", filter_mode="min")
            results_df.to_csv(args.tuning_output_file,index=False)

            ## if the tuning has already run and saved, can look at the best trial using the following code:
            ## tuner = tune.Tuner.restore(
            ##        path="~/ray_results/test_run"
            ##          )
            ## results = tuner.fit()

            best_trial = results.get_best_result("loss", "min","last-10-avg")
            print("best trial:", best_trial)
            print("Best trial config: {}".format(best_trial.config))
            print("Best trial final loss: {}".format(best_trial.metrics["loss"]))
            print("Best trial final auc: {}".format(best_trial.metrics["auc"]))
            print("Best trial final acuracy: {}".format(best_trial.metrics["accuracy"]))
            

        else:
            if args.sampling:
                test_auc, val_auc, test_acc, val_acc  = train_sampling(None,datasets, i, class_counts, args)
            else:
                test_auc, val_auc, test_acc, val_acc  = train(datasets, i, class_counts, args)
        
            all_test_auc.append(test_auc)
            all_val_auc.append(val_auc)
            all_test_acc.append(test_acc)
            all_val_acc.append(val_acc)

    
    if not args.tuning:
        final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
            'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

        if len(folds) != args.k:
            save_name = 'summary_partial_{}_{}.csv'.format(start, end)
        else:
            save_name = 'summary.csv'
        final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default="/", 
                    help='directory containing features folders')
parser.add_argument('--features_folder', type=str, default="/",
                    help='folder within data_root_dir containing the features - must contain pt_files/h5_files subfolder')
parser.add_argument('--coords_path', type=str, default=None,
                    help='path to coords pt files if needed')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--min_epochs', type=int, default=20,
                    help='minimum number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--continue_training', action='store_true', default=False, help='Continue model training from latest checkpoint')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout p=0.25')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'balanced_ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['256','tinier3','tinier_resnet18','tinier2_resnet18','tiny_resnet18','tinier','tiny','small', 'big','hipt_big','hipt_medium','hipt_small','hipt_smaller','hipt_smallest'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['ovarian_5class','ovarian_1vsall','nsclc','treatment','treatment_switched'])
parser.add_argument('--profile', action='store_true', default=False, 
                    help='show profile of longest running code sections')
parser.add_argument('--profile_rows', type=int, default=10, help='number of rows to show from profiler (requires --profile to show any)')
parser.add_argument('--csv_path',type=str,default=None,help='path to dataset_csv file')
parser.add_argument('--perturb', action='store_true', default=False, help='perturb features during training')
parser.add_argument('--perturb_variance', type=float, default=0.1, help='variance of feature perturbations')
parser.add_argument('--use_augs', action='store_true', default=False, help='use augmented versions of the training slides during training. The features to be saved in the same place as the non-augmented features, with the addition of "aug0", "aug1" etc. before the .pt in each filename')
parser.add_argument('--number_of_augs', type=int, default=1, help='number of augmented versions of each real image that are available')

## feature extraction options
parser.add_argument('--extract_features', action='store_true', default=False, help='extract features during training')
parser.add_argument('--augment_features', action='store_true', default=False, help='if extracting features, whether to apply augmentations before feature extraction')
parser.add_argument('--max_patches_per_slide', type=int, default=100000, help='number of patches to use per slide each iteration when extracting features during training')
parser.add_argument('--model_architecture',type=str,choices=['resnet18','resnet50','levit_128s'],default='resnet50')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--pretraining_dataset',type=str,choices=['ImageNet','Histo'],default='ImageNet')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)

## sampling options
parser.add_argument('--sampling', action='store_true', default=False, help='sampling for faster training')
parser.add_argument('--sampling_type', type=str, choices=['spatial','textural','newest'],default='spatial',help='type of sampling to use')
parser.add_argument('--samples_per_iteration', type=int, default=100, help='number of patches to sample per sampling iteration')
parser.add_argument('--resampling_iterations', type=int, default=10, help='number of resampling iterations (not including the initial sample)')
parser.add_argument('--sampling_random', type=float, default=0.2, help='proportion of samples which are completely random per iteration')
parser.add_argument('--sampling_random_delta',type=float, default=0.02, help='reduction in sampling_random per iteration')
parser.add_argument('--sampling_neighbors', type=int, default=20, help='number of nearest neighbors to consider when resampling')
parser.add_argument('--final_sample_size',type=int,default=100,help='number of patches for final sample')
parser.add_argument('--texture_model',type=str, choices=['resnet50','levit_128s'], default='resnet50',help='model to use for feature extraction in textural sampling')
parser.add_argument('--sampling_average',action='store_true',default=False,help='Take the sampling weights as averages rather than maxima to leverage more learned information')
parser.add_argument('--weight_smoothing',type=float,default=0.15,help='Power applied to attention scores to generate sampling weights')
parser.add_argument('--use_all_samples',action='store_true', default=False, help='Use all previous samples in the final sample step')
parser.add_argument('--no_sampling_epochs',type=int,default=20,help='number of epochs to complete full slide processing before beginning sampling')
parser.add_argument('--fully_random',action='store_true', default=False, help='Take entirely random samples (no active sampling)')


## tuning options
parser.add_argument('--tuning', action='store_true', default=False, help='run hyperparameter tuning')
parser.add_argument('--tuning_output_file',type=str,default="tuning_results/tuning_output.csv",help="where to save tuning outputs")
parser.add_argument('--num_tuning_experiments',type=int,default=100,help="number of tuning experiments")
parser.add_argument('--hardware',type=str, choices=['DGX','PC'], default='DGX',help='sets amount of CPU and GPU to use per experiment')

### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')

## debugging arg
parser.add_argument('--debug_loader', action='store_true', default=False,
                        help='debugger arg which runs through the loader but doesnt train models')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "drop_out": args.drop_out,
            "use_early_stopping": args.early_stopping,
            "use_sampling": args.sampling,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

if args.sampling:
    settings.update({'sampling_type': args.sampling_type})

print('\nLoad Dataset')
    
if args.task == 'ovarian_5class':
    args.n_classes=5
    args.label_dict = {'high_grade':0,'low_grade':1,'clear_cell':2,'endometrioid':3,'mucinous':4}
    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping

elif args.task == 'ovarian_1vsall':
    args.n_classes=2
    args.label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1}

elif args.task =='treatment':
    args.n_classes=2
    args.label_dict = {'invalid':0,'effective':1}

elif args.task == 'nsclc':
    args.n_classes=2
    args.label_dict = {'luad':0,'lusc':1}

else:
    raise NotImplementedError

dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                            data_dir= os.path.join(args.data_root_dir, args.features_folder),
                            max_patches_per_slide=args.max_patches_per_slide,
                            perturb_variance=args.perturb_variance,
                            number_of_augs=args.number_of_augs,
                            coords_path = args.coords_path,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = args.label_dict,
                            patient_strat=False,
                            data_h5_dir=args.data_h5_dir,
                            data_slide_dir=args.data_slide_dir,
                            slide_ext=args.slide_ext,
                            pretrained=True, 
                            custom_downsample=args.custom_downsample, 
                            target_patch_size=args.target_patch_size,
                            model_architecture = args.model_architecture,
                            batch_size = args.batch_size,
                            ignore=[])

    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    #torch.multiprocessing.set_start_method('spawn')
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        results = main()
        print("max gpu mem usage:",torch.cuda.max_memory_allocated())
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(args.profile_rows)
    else:
        results = main()
    print("finished!")
    print("end script")

