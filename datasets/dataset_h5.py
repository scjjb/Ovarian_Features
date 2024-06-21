from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py

import random
from random import randrange

def eval_transforms(pretrained=False):
        if pretrained:
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)

        else:
                mean = (0.5,0.5,0.5)
                std = (0.5,0.5,0.5)

        trnsfrms_val = transforms.Compose(
                                        [
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean = mean, std = std)
                                        ]
                                )

        return trnsfrms_val

class Whole_Slide_Bag(Dataset):
        def __init__(self,
                file_path,
                pretrained=False,
                custom_transforms=None,
                target_patch_size=-1,
                ):
                """
                Args:
                        file_path (string): Path to the .h5 file containing patched data.
                        pretrained (bool): Use ImageNet transforms
                        custom_transforms (callable, optional): Optional transform to be applied on a sample
                """
                self.pretrained=pretrained
                if target_patch_size > 0:
                        self.target_patch_size = (target_patch_size, target_patch_size)
                else:
                        self.target_patch_size = None

                if not custom_transforms:
                        self.roi_transforms = eval_transforms(pretrained=pretrained)
                else:
                        self.roi_transforms = custom_transforms

                self.file_path = file_path

                with h5py.File(self.file_path, "r") as f:
                        dset = f['imgs']
                        self.length = len(dset)

                self.summary()
                        
        def __len__(self):
                return self.length

        def summary(self):
                hdf5_file = h5py.File(self.file_path, "r")
                dset = hdf5_file['imgs']
                for name, value in dset.attrs.items():
                        print(name, value)

                print('pretrained:', self.pretrained)
                print('transformations:', self.roi_transforms)
                if self.target_patch_size is not None:
                        print('target_size: ', self.target_patch_size)

        def __getitem__(self, idx):
                with h5py.File(self.file_path,'r') as hdf5_file:
                        img = hdf5_file['imgs'][idx]
                        coord = hdf5_file['coords'][idx]
                
                img = Image.fromarray(img)
                if self.target_patch_size is not None:
                        img = img.resize(self.target_patch_size)
                img = self.roi_transforms(img).unsqueeze(0)
                return img, coord

class Whole_Slide_Bag_FP(Dataset):
        def __init__(self,
                file_path,
                wsi,
                pretrained=False,
                custom_transforms=None,
                custom_downsample=None,
                target_patch_size=-1,
                selected_idxs=None,
                max_patches_per_slide=None,
                model_architecture=None,
                batch_size=None,
                extract_features=False
                ):
                """
                Args:
                        file_path (string): Path to the .h5 file containing patched data.
                        pretrained (bool): Use ImageNet transforms
                        custom_transforms (callable, optional): Optional transform to be applied on a sample
                        custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
                        target_patch_size (int): Custom defined image size before embedding
                """
                self.pretrained = pretrained
                self.wsi = wsi
                self.max_patches_per_slide = max_patches_per_slide
                if not custom_transforms:
                        self.roi_transforms = eval_transforms(pretrained=pretrained)
                else:
                        self.roi_transforms = custom_transforms

                self.file_path = file_path
                self.extract_features = extract_features
                #print("file path:",self.file_path)
                with h5py.File(self.file_path, "r") as f:
                        self.coords=f['coords'][:len(f['coords'])]
                        if selected_idxs is None:
                            self.selected_coords=self.coords
                            #print(self.selected_coords)
                            #print(self.selected_coords[0])
                        else:
                            self.selected_coords = self.coords[sorted(list(set(selected_idxs)))]
                        #print("max patches per slide",self.max_patches_per_slide)
                        #print(self.selected_coords)
                        #print(self.selected_coords.dtype)
                        #if self.max_patches_per_slide:
                            #if self.max_patches_per_slide<len(self.selected_coords):
                                #self.selected_coords = random.sample(self.selected_coords,self.max_patches_per_slide)
                                #sample_keys = random.sample(list(self.selected_coords.keys()), self.max_patches_per_slide)
                                #self.selected_coords = {key: self.selected_coords[key] for key in sample_keys}
                                
                                ## below was working but turned off
                                #sample_idxs = random.sample(range(len(self.selected_coords)),self.max_patches_per_slide)
                        #        sample_idxs = np.random.choice(len(self.selected_coords),self.max_patches_per_slide)
                        #        self.selected_coords = torch.tensor(self.selected_coords[sorted(list(set(sample_idxs)))])
                        
                        #print("len selected_coords",len(self.selected_coords))
                        #print("selected coords:",self.selected_coords)
                        #print(self.selected_coords)
                        self.patch_level = f['coords'].attrs['patch_level']
                        self.patch_size = f['coords'].attrs['patch_size']
                        self.length = len(self.selected_coords)
                        if target_patch_size > 0:
                                self.target_patch_size = (target_patch_size, ) * 2
                        elif custom_downsample > 1:
                                self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
                        else:
                                self.target_patch_size = None
        
        def __len__(self):
                return self.length

        def summary(self):
                hdf5_file = h5py.File(self.file_path, "r")
                dset = hdf5_file['coords']
                for name, value in dset.attrs.items():
                        print(name, value)

                print('\nfeature extraction settings')
                print('target patch size: ', self.target_patch_size)
                print('pretrained: ', self.pretrained)
                print('transformations: ', self.roi_transforms)
        
        def update_sample(self,selected_idxs):
                #print("updating sample to length ",len(selected_idxs))
                #with h5py.File(self.file_path, "r") as f:
                #print("self.coords",self.coords)
                self.selected_coords=self.coords[sorted(list(set(selected_idxs)))]
                #print("selected coords in update sample",self.selected_coords)
                #print("selected_coords.shape in update_sample",self.selected_coords.shape)
                #print("selected coords",self.selected_coords)
                self.length = len(self.selected_coords)
                #print("self.length",self.length)
        
        def coords(self,num):
                with h5py.File(self.file_path,'r') as hdf5_file:
                    coords = hdf5_file['coords'][:num]
                return coords

        def __getitem__(self, idx):
                coord=self.selected_coords[idx]
                #print("coord entered into read_region",(coord))
                #print("selected_coords before read_region",self.selected_coords)
                
                #print("transforms:",self.roi_transforms)
                img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
                if self.target_patch_size is not None:
                        img = img.resize(self.target_patch_size)
                print(img)
                transform = transforms.Compose([transforms.ToTensor()])
                #print("before transforms",transform(img))
                img = self.roi_transforms(img).unsqueeze(0)
                #print("after transforms",img)
                return img, coord

class Dataset_All_Bags(Dataset):

        def __init__(self, csv_path):
                self.df = pd.read_csv(csv_path)
        
        def __len__(self):
                return len(self.df)

        def __getitem__(self, idx):
                return self.df['slide_id'][idx]
