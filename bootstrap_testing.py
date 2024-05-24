import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.metrics import f1_score

model_folder = "/mnt/results/eval_results"
model_names = "stagingandids_bce_uni_graph5x10x_sepzero_bestfrom8tuning_externaltest"
model_names=model_names.split(",")

bootstraps = 100

for model_name in model_names:
    full_model_name=model_folder+'/EVAL_'+model_name
    all_Yhats=[]
    all_Ys=[]
    all_p1s=[]
    all_probs=[]
    f1s = []
    all_losses=list(pd.read_csv(full_model_name+'/summary.csv')['loss'])
    for fold_no in range(5):
        full_df = pd.read_csv(full_model_name+'/fold_{}.csv'.format(fold_no))
        all_Yhats=all_Yhats+list(full_df['Y_hat'])
        all_Ys=all_Ys+list(full_df['Y'])
        if len(all_probs)<1:
            all_probs=full_df.iloc[:,-5:]
        else:
            all_probs=all_probs.append(full_df.iloc[:,-5:])
    for _ in range(bootstraps):
        idxs=np.random.choice(range(len(all_Ys)),len(all_Ys))
        classes_sampled = len(np.unique([all_Ys[idx] for idx in idxs]))
        while classes_sampled < 5:
            bootstrap_failure_resamples += 1
            print("resampling because of failed sample",bootstrap_failure_resamples)
            idxs=np.random.choice(range(len(all_Ys)),len(all_Ys))
            classes_sampled = len(np.unique([all_Ys[idx] for idx in idxs]))
        f1s=f1s+[f1_score([all_Ys[idx] for idx in idxs],[all_Yhats[idx] for idx in idxs],average='macro')]            

model_names2 = "stagingandids_bce_uni_graph5x10x_sepavg_bestfrom8tuning_externaltest"
model_names2=model_names2.split(",")

for model_name in model_names2:
    full_model_name=model_folder+'/EVAL_'+model_name
    all_Yhats2=[]
    all_Ys2=[]
    all_p1s2=[]
    all_probs2=[]
    f1s2 = []
    all_losses2=list(pd.read_csv(full_model_name+'/summary.csv')['loss'])
    for fold_no in range(5):
        full_df = pd.read_csv(full_model_name+'/fold_{}.csv'.format(fold_no))
        all_Yhats2=all_Yhats2+list(full_df['Y_hat'])
        all_Ys2=all_Ys2+list(full_df['Y'])
        if len(all_probs2)<1:
            all_probs2=full_df.iloc[:,-5:]
        else:
            all_probs2=all_probs2.append(full_df.iloc[:,-5:])

    for _ in range(bootstraps):
        idxs=np.random.choice(range(len(all_Ys2)),len(all_Ys2))
        classes_sampled = len(np.unique([all_Ys2[idx] for idx in idxs]))
        while classes_sampled < 5:
            bootstrap_failure_resamples += 1
            print("resampling because of failed sample",bootstrap_failure_resamples)
            idxs=np.random.choice(range(len(all_Ys2)),len(all_Ys2))
            classes_sampled = len(np.unique([all_Ys2[idx] for idx in idxs]))
        f1s2=f1s2+[f1_score([all_Ys2[idx] for idx in idxs],[all_Yhats2[idx] for idx in idxs],average='macro')] 

print(f1s)
print(f1s2)

stat, p1 = ttest_ind(f1s,f1s2,equal_var=False)
print('f1 stat=%.4f, p=%.4f' % (stat, p1))

print("COME BACK TO THIS IDEA BUT DO IT BETTER!!! search for 'bootstrapped p-values python' and look at 'https://www.datatipz.com/blog/hypothesis-testing-with-bootstrapping-python' ")

