import torch
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Basic k-nearest neighbors to test whether extracted features are sensible. This should easily achieve good performance as the train-test splits do not stratify patients, so the same patients will be in train and test')
parser.add_argument('--task',type=str,choices=['subtyping','subtyping_binary','treatment'],default='subtyping')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--data_root_dir', type=str, default="/",help='directory containing features folders')
parser.add_argument('--features_folder', type=str, default="/",help='folder within data_root_dir containing the features - must contain pt_files/h5_files subfolder')
parser.add_argument('--k', type=int, default=10,help='number of nearest neighbors in knn classification')
parser.add_argument('--splits', type=int, default=10,help='number of cross-validation splits (not currently stratified by patient)')
args = parser.parse_args()

df = pd.read_csv(args.csv_path,header=0)

def agg_slide_feature(region_features):
    h_WSI = region_features.mean(axis=0)
    return h_WSI

df = pd.read_csv(args.csv_path,header=0)
data_dir = os.path.join(args.data_root_dir, args.features_folder)

x=None
labels=[]
for row in df.iterrows():
    slide_id = row[1]['slide_id']
    labels = labels +  [row[1]['label']]
    full_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id))
    features = torch.load(full_path)
    averaged_features = features.mean(axis=0)
    if x is None:
        x = torch.unsqueeze(averaged_features, dim=0)
    else:
        x = torch.cat((x,torch.unsqueeze(averaged_features, dim=0)),0)

if args.task == "subtyping":
    label_dict = {'high_grade':0,'low_grade':1,'clear_cell':2,'endometrioid':3,'mucinous':4}
elif args.task == "subtyping_binary":
    label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1}
elif args.task == "treatment":
    label_dict = {'invalid':0,'effective':1}
else:
    raise NotImplementedError

labels = [label_dict[label] for label in labels]
assert 0 < sum(labels) < len(labels), "all labels are identical"

print("starting knn with k={} and {}-fold CV".format(args.k,args.splits))

## building on the code from https://github.com/mahmoodlab/HIPT/blob/master/3-Self-Supervised-Eval/slide_extraction-evaluation.ipynb
embeddings_all = x.detach().squeeze(1)
labels = np.array(labels)             
                              
clf = KNeighborsClassifier(args.k)
skf = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=0)

if len(label_dict.keys()) > 2:
    scores = cross_val_score(clf, embeddings_all, labels, cv=skf, scoring='roc_auc_ovr')
else:
    scores = cross_val_score(clf, embeddings_all, labels, cv=skf, scoring='roc_auc')
print("all auc scores:",scores)
print("mean auc score across folds:",round(scores.mean(),6))
print("std auc score across folds:",round(scores.std(),6))
