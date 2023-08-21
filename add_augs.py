import pandas as pd
import argparse
import os
import numpy as np
parser = argparse.ArgumentParser(description='Script to add augmentations to train fold')
parser.add_argument('--split_name', type=str, default=None)
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--num_augs', type=int, default=1, help='number of augs per image')
args = parser.parse_args()

os.mkdir('splits/{}_{}trainaugs'.format(args.split_name,args.num_augs))

for i in range(args.k):
    df = pd.read_csv('splits/{}/splits_{}.csv'.format(args.split_name,i),keep_default_na=False)
    train=df['train']
    train=list(train)
    for train_sample in train:
        if len(train_sample)>0:
            for aug in range(args.num_augs):    
                #print(len(train))
                #print(train)
                train=train+[train_sample.strip()+'aug{}'.format(aug+1)]
                #print(train)
                #print(len(train))
                #assert 1==2, "testing"
    #assert 1==2, "testing"
    val=df['val'].reindex(range(len(train)),fill_value='')
    test=df['test'].reindex(range(len(train)),fill_value='')
    print(train)
    print(val)
    print(test)
    dic={"train": train, "val": val, "test": test}
    new_df=pd.DataFrame(dic)
    new_df=new_df.fillna('')
    new_df.reset_index(drop=True,inplace=True)
    print(new_df)
    new_df.to_csv('splits/{}_{}trainaugs/splits_{}.csv'.format(args.split_name,args.num_augs,i))
