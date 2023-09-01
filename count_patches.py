import os
import numpy as np
from utils.utils import get_simple_loader
from datasets.dataset_generic import Generic_MIL_Dataset
import argparse


parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None,help='directory containing features folders')
parser.add_argument('--features_folder', type=str, default=None, help='folder within data_root_dir containing the features - must contain pt_files/h5_files subfolder')
parser.add_argument('--csv_path',type=str,default=None,help='path to dataset_csv file')
parser.add_argument('--coords_path',type=str,default=None,help='path to coords folder')
parser.add_argument('--n_classes',type=int,default=2,help='number of classes') 
args = parser.parse_args()

assert args.n_classes == 2, "currently only implemented for binary classification"

def count_patches(dataset):
    dataset.load_from_h5(True)
    loader = get_simple_loader(dataset)
    patches0=0
    patches1=0
    print("slides: ",len(loader))
    patch_counts=[]
    all_counts=[]
    for batch_idx, (data, label, coords, ids) in enumerate(loader):
        count=len(coords)
        if label==0:
            patches0=patches0+count
        elif label==1:
            patches1=patches1+count
        patch_counts=patch_counts+[[ids,count]]
        all_counts=all_counts+[count]
        print("number", batch_idx, "   slide",ids,"  class 0 patches: ",patches0, "  class 1 patches: ",patches1)
    patches=patches0+patches1
    #pd.DataFrame(patch_counts,columns=["slide","patches"]).to_csv("results/patch_counts/ESGO_available_staging.csv",index=False)
    return patches, all_counts


csv_path=args.csv_path
data_root_dir=args.data_root_dir
features_folder=args.features_folder
coords_path=args.coords_path

n_classes = args.n_classes
label_dict = {'invalid': 0,'effective': 1}
#label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1}
dataset = Generic_MIL_Dataset(csv_path = csv_path,
                        data_dir= os.path.join(data_root_dir, features_folder),
                        coords_path = coords_path,
                        shuffle = False, 
                        seed = 0, 
                        print_info = True,
                        label_dict = label_dict,
                        patient_strat=False,
                        ignore=[],
                        max_patches_per_slide=100000000)
split_dataset=dataset
patches, all_counts  = count_patches(split_dataset)
print("{} patches".format(patches))

print("min patches:",min(all_counts))
print("max patches:",max(all_counts))
print("mean patches:",np.mean(all_counts))
print("sd patches:",np.std(all_counts))
