from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats
import random

from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth

## newly added for the online augmentations 
from models.resnet_custom import resnet18_baseline,resnet50_baseline
from torchvision import transforms
from torch.utils.data import DataLoader
import openslide
import timm
from datasets.dataset_h5 import Whole_Slide_Bag_FP
from utils.utils import collate_features

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
        splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
        if not boolean_style:
                df = pd.concat(splits, ignore_index=True, axis=1)
                df.columns = column_keys
        else:
                df = pd.concat(splits, ignore_index = True, axis=0)
                index = df.values.tolist()
                one_hot = np.eye(len(split_datasets)).astype(bool)
                bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
                df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

        df.to_csv(filename)
        print()

class Generic_WSI_Classification_Dataset(Dataset):
        def __init__(self,
                csv_path = 'dataset_csv/ccrcc_clean.csv',
                coords_path = None,
                shuffle = False, 
                seed = 7, 
                print_info = True,
                label_dict = {},
                filter_dict = {},
                ignore=[],
                patient_strat=False,
                label_col = None,
                patient_voting = 'max',
                perturb_variance=0.0,
                number_of_augs=0,
                slide_ext=None,
                data_h5_dir=None,
                data_slide_dir=None,
                max_patches_per_slide=None
                ):
                """
                Args:
                        csv_file (string): Path to the csv file with annotations.
                        shuffle (boolean): Whether to shuffle
                        seed (int): random seed for shuffling the data
                        print_info (boolean): Whether to print a summary of the dataset
                        label_dict (dict): Dictionary with key, value pairs for converting str labels to int
                        ignore (list): List containing class labels to ignore
                """
                self.label_dict = label_dict
                self.num_classes = len(set(self.label_dict.values()))
                self.seed = seed
                self.print_info = print_info
                self.patient_strat = patient_strat
                self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
                self.data_dir = None
                self.coords_path = coords_path
                self.perturb_variance = perturb_variance
                self.number_of_augs = number_of_augs
                self.slide_ext = slide_ext
                self.data_h5_dir = data_h5_dir
                self.data_slide_dir = data_slide_dir
                self.max_patches_per_slide = max_patches_per_slide
                if not label_col:
                        label_col = 'label'
                self.label_col = label_col

                slide_data = pd.read_csv(csv_path)
                slide_data = self.filter_df(slide_data, filter_dict)
                slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)
                self.pretrained = None
                self.custom_downsample = None
                self.target_patch_size = None
                self.model_architecture = None
                self.batch_size = None

                ###shuffle data
                if shuffle:
                        np.random.seed(seed)
                        np.random.shuffle(slide_data)

                self.slide_data = slide_data

                self.patient_data_prep(patient_voting)
                self.cls_ids_prep()

                if print_info:
                        self.summarize()

        def cls_ids_prep(self):
                # store ids corresponding each class at the patient or case level
                self.patient_cls_ids = [[] for i in range(self.num_classes)]            
                for i in range(self.num_classes):
                        self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

                # store ids corresponding each class at the slide level
                self.slide_cls_ids = [[] for i in range(self.num_classes)]
                for i in range(self.num_classes):
                        self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        def patient_data_prep(self, patient_voting='max'):
                patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
                patient_labels = []
                
                for p in patients:
                        locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
                        assert len(locations) > 0
                        label = self.slide_data['label'][locations].values
                        if patient_voting == 'max':
                                label = label.max() # get patient label (MIL convention)
                        elif patient_voting == 'maj':
                                label = stats.mode(label)[0]
                        else:
                                raise NotImplementedError
                        patient_labels.append(label)
                
                self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

        @staticmethod
        def df_prep(data, label_dict, ignore, label_col):
                if label_col != 'label':
                        data['label'] = data[label_col].copy()

                mask = data['label'].isin(ignore)
                data = data[~mask]
                data.reset_index(drop=True, inplace=True)
                for i in data.index:
                        key = data.loc[i, 'label']
                        data.at[i, 'label'] = label_dict[key]

                return data

        def filter_df(self, df, filter_dict={}):
                if len(filter_dict) > 0:
                        filter_mask = np.full(len(df), True, bool)
                        # assert 'label' not in filter_dict.keys()
                        for key, val in filter_dict.items():
                                mask = df[key].isin(val)
                                filter_mask = np.logical_and(filter_mask, mask)
                        df = df[filter_mask]
                return df

        def __len__(self):
                if self.patient_strat:
                        return len(self.patient_data['case_id'])

                else:
                        return len(self.slide_data)

        def summarize(self):
                print("label column: {}".format(self.label_col))
                print("label dictionary: {}".format(self.label_dict))
                print("number of classes: {}".format(self.num_classes))
                print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
                for i in range(self.num_classes):
                        print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
                        print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

        def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
                settings = {
                                        'n_splits' : k, 
                                        'val_num' : val_num, 
                                        'test_num': test_num,
                                        'label_frac': label_frac,
                                        'seed': self.seed,
                                        'custom_test_ids': custom_test_ids
                                        }

                if self.patient_strat:
                        settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
                else:
                        settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

                self.split_gen = generate_split(**settings)

        def set_splits(self,start_from=None):
                if start_from:
                        ids = nth(self.split_gen, start_from)

                else:
                        ids = next(self.split_gen)

                if self.patient_strat:
                        slide_ids = [[] for i in range(len(ids))] 

                        for split in range(len(ids)): 
                                for idx in ids[split]:
                                        case_id = self.patient_data['case_id'][idx]
                                        slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
                                        slide_ids[split].extend(slide_indices)

                        self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

                else:
                        self.train_ids, self.val_ids, self.test_ids = ids

        def get_split_from_df(self, all_splits, split_key='train'):
                split = all_splits[split_key]
                split = split.dropna().reset_index(drop=True)

                if len(split) > 0:
                        mask = self.slide_data['slide_id'].isin(split.tolist())
                        df_slice = self.slide_data[mask].reset_index(drop=True)
                        split = Generic_Split(df_slice, data_dir=self.data_dir, coords_path=self.coords_path, num_classes=self.num_classes,perturb_variance=self.perturb_variance,number_of_augs=self.number_of_augs,slide_ext=self.slide_ext,data_h5_dir=self.data_h5_dir, data_slide_dir=self.data_slide_dir,pretrained=self.pretrained, custom_downsample=self.custom_downsample, target_patch_size=self.target_patch_size,model_architecture = self.model_architecture, batch_size = self.batch_size,max_patches_per_slide=self.max_patches_per_slide)
                else:
                        split = None
                
                return split

        def get_merged_split_from_df(self, all_splits, split_keys=['train']):
                merged_split = []
                for split_key in split_keys:
                        split = all_splits[split_key]
                        split = split.dropna().reset_index(drop=True).tolist()
                        merged_split.extend(split)

                if len(split) > 0:
                        mask = self.slide_data['slide_id'].isin(merged_split)
                        df_slice = self.slide_data[mask].reset_index(drop=True)
                        split = Generic_Split(df_slice, data_dir=self.data_dir, coords_path=self.coords_path, num_classes=self.num_classes,perturb_variance=self.perturb_variance,number_of_augs=self.number_of_augs,slide_ext=self.slide_ext,data_h5_dir=self.data_h5_dir, data_slide_dir=self.data_slide_dir,pretrained=self.pretrained, custom_downsample=self.custom_downsample, target_patch_size=self.target_patch_size,model_architecture = self.model_architecture, batch_size = self.batch_size,max_patches_per_slide=self.max_patches_per_slide)
                else:
                        split = None
                
                return split


        def return_splits(self, from_id=True, csv_path=None):


                if from_id:
                        if len(self.train_ids) > 0:
                                train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
                                train_split = Generic_Split(train_data, data_dir=self.data_dir, coords_path=self.coords_path, num_classes=self.num_classes,perturb_variance=self.perturb_variance,number_of_augs=self.number_of_augs,slide_ext=self.slide_ext,data_h5_dir=self.data_h5_dir, data_slide_dir=self.data_slide_dir,pretrained=self.pretrained, custom_downsample=self.custom_downsample, target_patch_size=self.target_patch_size,model_architecture = self.model_architecture, batch_size = self.batch_size,max_patches_per_slide=self.max_patches_per_slide)

                        else:
                                train_split = None
                        
                        if len(self.val_ids) > 0:
                                val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
                                val_split = Generic_Split(val_data, data_dir=self.data_dir, coords_path=self.coords_path, num_classes=self.num_classes,slide_ext=self.slide_ext,data_h5_dir=self.data_h5_dir, data_slide_dir=self.data_slide_dir,pretrained=self.pretrained, custom_downsample=self.custom_downsample, target_patch_size=self.target_patch_size,model_architecture = self.model_architecture, batch_size = self.batch_size,max_patches_per_slide=self.max_patches_per_slide)

                        else:
                                val_split = None
                        
                        if len(self.test_ids) > 0:
                                test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
                                test_split = Generic_Split(test_data, data_dir=self.data_dir, coords_path=self.coords_path, num_classes=self.num_classes,slide_ext=self.slide_ext,data_h5_dir=self.data_h5_dir, data_slide_dir=self.data_slide_dir,pretrained=self.pretrained, custom_downsample=self.custom_downsample, target_patch_size=self.target_patch_size,model_architecture = self.model_architecture, batch_size = self.batch_size,max_patches_per_slide=self.max_patches_per_slide)
                        
                        else:
                                test_split = None
                        
                
                else:
                        assert csv_path 
                        all_splits = pd.read_csv(csv_path)#, dtype=self.slide_data['slide_id'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
                        all_splits.astype('str')
                        train_split = self.get_split_from_df(all_splits, 'train')
                        val_split = self.get_split_from_df(all_splits, 'val')
                        test_split = self.get_split_from_df(all_splits, 'test')
                        
                return train_split, val_split, test_split

        def get_list(self, ids):
                return self.slide_data['slide_id'][ids]

        def getlabel(self, ids):
                return self.slide_data['label'][ids]

        def __getitem__(self, idx):
                return None

        def test_split_gen(self, return_descriptor=False):

                if return_descriptor:
                        index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
                        columns = ['train', 'val', 'test']
                        df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
                                                        columns= columns)

                count = len(self.train_ids)
                print('\nnumber of training samples: {}'.format(count))
                labels = self.getlabel(self.train_ids)
                unique, counts = np.unique(labels, return_counts=True)
                for u in range(len(unique)):
                        print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
                        if return_descriptor:
                                df.loc[index[u], 'train'] = counts[u]
                
                count = len(self.val_ids)
                print('\nnumber of val samples: {}'.format(count))
                labels = self.getlabel(self.val_ids)
                unique, counts = np.unique(labels, return_counts=True)
                for u in range(len(unique)):
                        print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
                        if return_descriptor:
                                df.loc[index[u], 'val'] = counts[u]

                count = len(self.test_ids)
                print('\nnumber of test samples: {}'.format(count))
                labels = self.getlabel(self.test_ids)
                unique, counts = np.unique(labels, return_counts=True)
                for u in range(len(unique)):
                        print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
                        if return_descriptor:
                                df.loc[index[u], 'test'] = counts[u]

                assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
                assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
                assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

                if return_descriptor:
                        return df

        def save_split(self, filename):
                train_split = self.get_list(self.train_ids)
                val_split = self.get_list(self.val_ids)
                test_split = self.get_list(self.test_ids)
                df_tr = pd.DataFrame({'train': train_split})
                df_v = pd.DataFrame({'val': val_split})
                df_t = pd.DataFrame({'test': test_split})
                df = pd.concat([df_tr, df_v, df_t], axis=1) 
                df.to_csv(filename, index = False)

        
        def count_by_class(self, split='train', csv_path=None):
                assert csv_path 
                all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)
                chosen_split = self.get_split_from_df(all_splits, split)
                count_list = [len(cls_ids) for cls_ids in chosen_split.slide_cls_ids]
                return count_list
                                                                


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
        def __init__(self,
                data_dir, 
                coords_path,
                perturb_variance=0.1,
                number_of_augs=1,
                max_patches_per_slide=float('inf'),
                data_h5_dir=None,
                data_slide_dir=None,
                slide_ext=None,
                pretrained=None, 
                custom_downsample=None, 
                target_patch_size=None,
                model_architecture=None,
                batch_size=None,
                debug_loader=False,
                **kwargs):
        
                super(Generic_MIL_Dataset, self).__init__(**kwargs)
                self.data_dir = data_dir
                self.coords_path = coords_path
                self.use_h5 = False
                self.extract_features = False
                self.augment_features = False
                self.transforms = None
                self.max_patches_per_slide = max_patches_per_slide
                self.data_h5_dir = data_h5_dir
                self.data_slide_dir = data_slide_dir
                self.slide_ext = slide_ext
                self.use_perturbs = False
                self.use_augs = False
                self.perturb_variance = perturb_variance
                self.number_of_augs = number_of_augs
                self.model = None
                self.pretrained = pretrained 
                self.custom_downsample = custom_downsample
                self.target_patch_size = target_patch_size
                self.model_architecture = model_architecture
                self.batch_size = batch_size
                self.debug_loader = debug_loader
        def load_from_h5(self, toggle):
                self.use_h5 = toggle
                print("use_h5 is currently not set to use h5 but to instead get coords from pt")

        def set_debug_loader(self, toggle):
                self.debug_loader = toggle
                if toggle:
                    print("debugging loader - model will not train")

        def set_extract_features(self, toggle):
                self.extract_features = toggle
                if toggle:
                    print("extracting features from {} patches per slide".format(self.max_patches_per_slide))

        def set_augment_features(self, toggle):
                self.augment_features = toggle    
                assert(self.extract_features, "augment_features requires extract_features")
                if toggle:
                    print("augmenting features")

        def perturb_features(self, toggle):
                self.use_perturbs = toggle
                print("perturbing features")

        def use_augmentations(self, toggle):
                self.use_augs = toggle
                print("using augmentations")

        def set_transforms(self):
                if self.augment_features:
                    self.transforms = transforms.Compose(
                                            [transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomVerticalFlip(p=0.5),
                                            transforms.RandomAffine(degrees=5,translate=(0.1,0.1), scale=(0.9,1.1),shear=0.1),
                                            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                                            transforms.ToTensor()
                                            ])
                else:
                    self.transforms = transforms.Compose(
                                            [transforms.ToTensor()])

        #def initialise_model(self, model_architecture, pretraining_dataset):
        #        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #        print('loading {} pretrained model {}'.format(pretraining_dataset, model_architecture))
        #        if model_architecture=='resnet18':
        #            self.model = resnet18_baseline(pretrained=True,dataset=pretraining_dataset)
        #        elif model_architecture=='resnet50':
        #            self.model = resnet50_baseline(pretrained=True,dataset=pretraining_dataset)    
        #        elif model_architecture=='levit_128s':
        #            self.model=timm.create_model('levit_256',pretrained=True, num_classes=0)
        #        self.model.to(device)

        def __getitem__(self, idx):
                slide_id = self.slide_data['slide_id'][idx]
                label = self.slide_data['label'][idx]
                if type(self.data_dir) == dict:
                        source = self.slide_data['source'][idx]
                        data_dir = self.data_dir[source]
                else:
                        data_dir = self.data_dir

                if self.extract_features:
                    #torch.multiprocessing.set_start_method('spawn')
                    h5_file_path = os.path.join(self.data_h5_dir, 'patches', slide_id+".h5")
                    file_path = os.path.join(self.data_slide_dir, slide_id+self.slide_ext)
                    wsi = openslide.open_slide(file_path)
                    dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, wsi=wsi, custom_transforms=self.transforms, pretrained=self.pretrained,custom_downsample=self.custom_downsample, target_patch_size=self.target_patch_size, max_patches_per_slide = self.max_patches_per_slide,model_architecture = self.model_architecture, batch_size = self.batch_size, extract_features = self.extract_features)
                    #print("IN GETITEM len dataset before update",len(dataset))
                    dataset.update_sample(np.random.choice(len(dataset),self.max_patches_per_slide))
                    #print("IN GETITEM len dataset after update",len(dataset))
                    #print("dataset len",len(dataset))
                    
                    #x, y = dataset[0]
                    #device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    ## cannot split num workers as this is an inner loop
                    #if self.model_architecture=='resnet18':
                    #    kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == "cuda" else {}
                    #elif self.model_architecture=='resnet50':
                    #    kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == "cuda" else {}
                    #elif self.model_architecture=='levit_128s':
                    #    kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == "cuda" else {}
                    #    tfms=torch.nn.Sequential(transforms.CenterCrop(224))
                    #kwargs = {}
                    #loader = DataLoader(dataset=dataset, batch_size=self.batch_size, **kwargs, collate_fn=collate_features,multiprocessing_context='spawn')
                    #all_features=[]
                    #for count, (batch, coords) in enumerate(loader):
                    #    with torch.no_grad():   
                    #        batch = batch.to(device, non_blocking=True)
                    #        if args.model_type=='levit_128s':
                    #            batch=tfms(batch)
                    #        features = model(batch)
                    #        features = features.cpu().numpy()
                    #        all_features = all_features+features
                    #print("IN GETITEM after update dataset",dataset)
                    patches = [data for data in dataset]
                    #print(patches[0][0].shape)
                    #print(patches[0][1])
                    label = torch.tensor(label)
                    return patches, label

                
                if self.use_augs:
                    assert not self.use_h5, "augmentations not currently setup with h5 files, only pt files"
                    ## aug numbers start at 0, -1 is the original with no augmentation 
                    aug_number = random.randint(-1,self.number_of_augs-1)
                    if aug_number>-1:
                        slide_id=slide_id+"aug{}".format(aug_number)
                    #print(self.number_of_augs)
                    #print(slide_id)
                if not self.use_h5:
                        if self.data_dir:
                                full_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id))
                                #print("loading :",full_path)
                                if self.debug_loader:
                                    print(slide_id)
                                try:
                                    features = torch.load(full_path)
                                except:
                                    assert 1==2, "Error caused by slide {}".format(slide_id)
                                #print("max patches per slide:",self.max_patches_per_slide)
                                #print("len features:",len(features))
                                if self.max_patches_per_slide < len(features):
                                    sampled_idxs=np.random.choice(len(features),self.max_patches_per_slide)
                                    features = features[sampled_idxs]
                                #print("len features after:",len(features))
                                if self.use_perturbs:
                                    #print("features",features)
                                    #print("variance:",self.perturb_variance)
                                    noise = torch.randn_like(features)*self.perturb_variance
                                    features = features + noise
                                    #print("perturbed",features)
                                #print("loaded :",full_path)
                                return features, label
                        
                        else:
                                return slide_id, label

                else:
                    if self.coords_path is not None:
                        #h5_path = os.path.join(data_dir,'h5_files','{}.h5'.format(slide_id))
                        ## Next line is not a replacement for line above as the coords change from the patches to the h5 file
                        #h5_path = os.path.join('../mount_outputs/patches','{}.h5'.format(slide_id))
                        
                        
                        #with h5py.File(h5_path,'r') as hdf5_file:
                                #features_h5 = hdf5_file['features'][:]
                                #coords = hdf5_file['coords'][:]
                                #print(hdf5_file)
                                #print(hdf5_file.keys())
                                #assert 1==2,"testing"
                                #coords = hdf5_file['coords'][:]
                                #print(coords)
                                #assert 1==2,"testing"
                        #coords=torch.load(h5_path)
                        
                        full_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id))
                        features = torch.load(full_path)
                        
                        
                        #coords_path=os.path.join("../mount_outputs/coords","{}.pt".format(slide_id))
                        coords_path=os.path.join(self.coords_path,"{}.pt".format(slide_id))
                        coords=torch.load(coords_path)
                        
                        #print(coords)
                        #assert 1==2,"testing"

                        #features = torch.from_numpy(features_h5)
                    else:
                        full_path = os.path.join(data_dir,'h5_files','{}.h5'.format(slide_id))
                        with h5py.File(full_path,'r') as hdf5_file:
                            features = hdf5_file['features'][:]
                            coords = hdf5_file['coords'][:]
                        features = torch.from_numpy(features)
                    #print("max patches per slide:",self.max_patches_per_slide)
                    #print("len features:",len(features))
                    if self.max_patches_per_slide < len(features):
                        sampled_idxs=np.random.choice(len(features),self.max_patches_per_slide)
                        features = features[sampled_idxs]
                        coords = coords[sampled_idxs]
                    if self.use_perturbs:
                        noise = torch.randn_like(features) * 0.1
                        features = features + noise
                    return features, label, coords, slide_id


class Generic_Split(Generic_MIL_Dataset):
        def __init__(self, slide_data, data_dir=None, coords_path=None, num_classes=2, perturb_variance=0.1, number_of_augs = 1, max_patches_per_slide=None,data_h5_dir=None,data_slide_dir=None,slide_ext=None, pretrained=None, custom_downsample=None, target_patch_size=None, model_architecture=None, batch_size = None, extract_features = False):
                self.debug_loader = False
                self.use_h5 = False
                self.use_perturbs = False
                self.use_augs = False
                self.perturb_variance = perturb_variance
                self.number_of_augs = number_of_augs
                self.slide_data = slide_data
                self.data_dir = data_dir
                self.coords_path = coords_path
                self.num_classes = num_classes
                self.max_patches_per_slide = max_patches_per_slide
                self.slide_cls_ids = [[] for i in range(self.num_classes)]
                self.data_h5_dir = data_h5_dir
                self.data_slide_dir = data_slide_dir
                self.slide_ext = slide_ext
                self.pretrained = pretrained
                self.custom_downsample = custom_downsample
                self.target_patch_size = target_patch_size
                self.model_architecture = model_architecture
                self.batch_size = batch_size
                print("max patches per slide",self.max_patches_per_slide)
                self.extract_features = extract_features
                for i in range(self.num_classes):
                        self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        def __len__(self):
                return len(self.slide_data)
                

