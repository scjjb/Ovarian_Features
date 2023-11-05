import pandas as pd

import argparse
parser = argparse.ArgumentParser(description='taking the results from 5-fold CV tuning and combining them to a single file')
parser.add_argument('--file_base_name', type=str, default="/mnt/results/tuning_results/staging_only_resnet50_40x_thirdtuning_bce", help='path and file name of tuning results file without the final _fold0.csv etc.')
args = parser.parse_args()

file_name_base = args.file_base_name

files=[file_name_base+"_fold{}.csv".format(i) for i in range(5)]

combined_df = pd.read_csv(files[0])
combined_df.rename(columns={'loss': 'fold0loss'}, inplace=True)

# Loop through the other files and concatenate their first columns to the combined DataFrame
for i in range(1,5):
    file = files[i]
    df = pd.read_csv(file, usecols=[0])
    df.columns=["fold{}loss".format(i)]
    combined_df = pd.concat([combined_df, df], axis=1)

column_names=["fold{}loss".format(i) for i in range(5)]
combined_df['avgloss'] = combined_df[column_names].mean(axis=1)

print(combined_df)
combined_df.to_csv(file_name_base+"_allfolds.csv", index=False)
