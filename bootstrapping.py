import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score,balanced_accuracy_score, roc_auc_score, roc_curve, auc, precision_score, recall_score
import matplotlib.pyplot as plt
import os
import numpy as np
import torch

parser = argparse.ArgumentParser(description='Model names input split by commas')
parser.add_argument('--model_names', type=str, default=None,help='models to plot')
parser.add_argument('--bootstraps', type=int, default=100000,
                    help='Number of bootstraps to calculate')
parser.add_argument('--run_repeats', type=int, default=10,
                            help='Number of model repeats')
parser.add_argument('--folds', type=int, default=10,
                            help='Number of cross-validation folds')
parser.add_argument('--data_csv', type=str, default='set_all_714.csv')
parser.add_argument('--model_folder', type=str, default='/mnt/results/eval_results')
parser.add_argument('--num_classes',type=int,default=2)
parser.add_argument('--plot_roc_curves', action='store_true', default=False, help="Plot an ROC curve for each run repeat")
parser.add_argument('--roc_plot_dir', type=str, default='../mount_outputs/roc_plots/',help='directory to plot ROC curves')
parser.add_argument('--ensemble', action='store_true', default=False, help="Ensemble the predictions from different folds into one prediction. Only works if all folds test sets are identical.")
args = parser.parse_args()
model_names=args.model_names.split(",")
bootstraps=args.bootstraps

model_folder = args.model_folder
for model_name in model_names:
    full_model_name=model_folder+'/EVAL_'+model_name

    all_auc_means=[]
    all_f1_means=[]
    all_accuracy_means=[]
    all_balanced_accuracy_means=[]
    all_auc_sds=[]
    all_f1_sds=[]
    all_accuracy_sds=[]
    all_balanced_accuracy_sds=[]
    all_auc_cis=[]
    all_f1_cis=[]
    all_accuracy_cis=[]
    all_balanced_accuracy_cis=[]

    for run_no in range(args.run_repeats):
            
        all_Yhats=[]
        all_Ys=[]
        all_p1s=[]
        all_probs=[]
        all_losses=list(pd.read_csv(full_model_name+'/summary.csv')['loss'])
        print("run: ",run_no)
        for fold_no in range(args.folds):
            if args.run_repeats>1:
                full_df = pd.read_csv(full_model_name+'_run{}/fold_{}.csv'.format(run_no,fold_no))
            else:
                full_df = pd.read_csv(full_model_name+'/fold_{}.csv'.format(fold_no))
            all_Yhats=all_Yhats+list(full_df['Y_hat'])
            all_Ys=all_Ys+list(full_df['Y'])
            if args.num_classes==2:
                all_p1s=all_p1s+list(full_df['p_1'])
            else:
                if len(all_probs)<1:
                    all_probs=full_df.iloc[:,-args.num_classes:]
                else:
                    all_probs=all_probs.append(full_df.iloc[:,-args.num_classes:])


        if args.ensemble:
            num_of_samples = int(len(all_Ys)/args.folds)
            all_Ys=all_Ys[:num_of_samples]
            ensemble_probs = all_probs.head(num_of_samples)
            ensemble_Yhats = ['None'] * num_of_samples
            for i in range(num_of_samples):
                ensemble_probs.iloc[i] = all_probs.iloc[i::num_of_samples].mean(axis=0)
                ensemble_Yhats[i] = torch.topk(torch.tensor(ensemble_probs.iloc[i]), 1, dim = 0)[1].item()
            all_probs = ensemble_probs
            all_Yhats = ensemble_Yhats


        AUC_scores=[]
        err_scores=[]
        accuracies=[]
        f1s=[]
        balanced_accuracies=[]
        
        print("confusion matrix (predicted x axis, true y axis): \n")
        print(confusion_matrix(all_Ys,all_Yhats),"\n")

        for i in range(len(np.unique(all_Ys))):
            #indxs = [index for index, value in enumerate(all_Ys) if value == i]
            #precision = precision_score([all_Ys[index] for index in indxs],[all_Yhats[index] for index in indxs])
            print("class {} precision: {:.5f} recall: {:.5f} f1: {:.5f}".format(i,precision_score(all_Ys,all_Yhats,labels=[i],average='macro'),recall_score(all_Ys,all_Yhats,labels=[i],average='macro'),f1_score(all_Ys,all_Yhats,labels=[i],average='macro')))
            #print("class {} recall: {}".format(i,recall_score(all_Ys,all_Yhats,labels=[i],average='macro')))        
            #print("class {} F1: {}".format(i,f1_score(all_Ys,all_Yhats,labels=[i],average='macro')))
        print("\naverage loss: ",np.mean(all_losses), "(not bootstrapped)")

        if args.plot_roc_curves:
            fpr, tpr, threshold = roc_curve(all_Ys, all_p1s)
            roc_auc = auc(fpr, tpr)
            if args.run_repeats>1:
                plt.plot(fpr, tpr, label = 'Repeat '+str(run_no+1))
            else:
                plt.plot(fpr, tpr)
        
        bootstrap_failure_resamples = 0
        for _ in range(bootstraps):
            idxs=np.random.choice(range(len(all_Ys)),len(all_Ys))
            classes_sampled = len(np.unique([all_Ys[idx] for idx in idxs]))
            while classes_sampled < args.num_classes:
                bootstrap_failure_resamples += 1
                print("resampling because of failed sample",bootstrap_failure_resamples)
                idxs=np.random.choice(range(len(all_Ys)),len(all_Ys))
                classes_sampled = len(np.unique([all_Ys[idx] for idx in idxs]))
            
            if args.num_classes==2:
                f1s=f1s+[f1_score([all_Ys[idx] for idx in idxs],[all_Yhats[idx] for idx in idxs])]
                AUC_scores=AUC_scores+[roc_auc_score([all_Ys[idx] for idx in idxs],[all_p1s[idx] for idx in idxs])]
            else:
                f1s=f1s+[f1_score([all_Ys[idx] for idx in idxs],[all_Yhats[idx] for idx in idxs],average='macro')]
                AUC_scores=AUC_scores+[roc_auc_score([all_Ys[idx] for idx in idxs],[all_probs.iloc[idx,:] for idx in idxs],multi_class='ovr')]
            accuracies=accuracies+[accuracy_score([all_Ys[idx] for idx in idxs],[all_Yhats[idx] for idx in idxs])]
            balanced_accuracies=balanced_accuracies+[balanced_accuracy_score([all_Ys[idx] for idx in idxs],[all_Yhats[idx] for idx in idxs])]
            

        if args.plot_roc_curves:
            os.makedirs(args.roc_plot_dir, exist_ok=True)
            print("saving ROC curves to {}{}.png \n".format(args.roc_plot_dir,model_name))
            plt.savefig("{}{}.png".format(args.roc_plot_dir,model_name),dpi=300)

        all_auc_means=all_auc_means+[np.mean(AUC_scores)]
        all_auc_sds=all_auc_sds+[np.std(AUC_scores)]
        all_auc_cis=all_auc_cis+[list(np.percentile(AUC_scores, [2.5,97.5]))]
        all_f1_means=all_f1_means+[np.mean(f1s)]
        all_f1_sds=all_f1_sds+[np.std(f1s)]
        all_f1_cis=all_f1_cis+[list(np.percentile(f1s, [2.5,97.5]))]
        all_accuracy_means=all_accuracy_means+[np.mean(accuracies)]
        all_accuracy_sds=all_accuracy_sds+[np.std(accuracies)]
        all_accuracy_cis=all_accuracy_cis+[list(np.percentile(accuracies, [2.5,97.5]))]
        all_balanced_accuracy_means=all_balanced_accuracy_means+[np.mean(balanced_accuracies)]
        all_balanced_accuracy_sds=all_balanced_accuracy_sds+[np.std(balanced_accuracies)]
        all_balanced_accuracy_cis=all_balanced_accuracy_cis+[list(np.percentile(balanced_accuracies, [2.5,97.5]))]

        print("AUC mean: ", all_auc_means," AUC std: ",all_auc_sds, "AUC 95%CI: ",all_auc_cis)
        if args.num_classes==2:
            print("F1 mean: ",all_f1_means," F1 std: ",all_f1_sds, "F1 95%CI: ",all_f1_cis)
        else:
            print("Macro F1 mean: ",all_f1_means," F1 std: ",all_f1_sds, "F1 95%CI: ",all_f1_cis)
        print("accuracy mean: ",all_accuracy_means," accuracy std: ",all_accuracy_sds, "accuracy 95%CI: ",all_accuracy_cis)
        print("balanced accuracy mean: ",all_balanced_accuracy_means," balanced accuracy std: ",all_balanced_accuracy_sds, "balanced accuracy 95%CI: ",all_balanced_accuracy_cis)
        
    plot_CIs=False
    if plot_CIs:
        plot_to = "/mnt/results/CIs/"
        plt.hist(AUC_scores, bins=50)
        plt.axvline(np.percentile(AUC_scores, [2.5]), color="red")
        plt.axvline(np.percentile(AUC_scores, [97.5]), color="red")
        plt.savefig(plot_to+"AUC.png")

    df=pd.DataFrame([[all_auc_means],[all_accuracy_means],[all_balanced_accuracy_means],[all_f1_means],[all_auc_sds],[all_accuracy_sds],[all_balanced_accuracy_sds],[all_f1_sds]])
    df.to_csv("metric_results/"+model_name+".csv",index=False)

