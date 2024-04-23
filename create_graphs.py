import argparse
import os 
import time
import numpy as np
import pandas as pd
import torch
import h5py
from scipy.spatial.distance import cdist, pdist, squareform


parser = argparse.ArgumentParser(description='Configurations for WSI Training')
## Folders
parser.add_argument('--data_root_dir', type=str, default="/", 
                    help='directory containing features folders')
parser.add_argument('--features_folder', type=str, default="/",
                    help='folder within data_root_dir containing the features - must contain pt_files/h5_files subfolder')
parser.add_argument('--small_features_folder', type=str, default="/",
                    help='folder within data_root_dir containing the small features if needed (only used in graph_ms)- must contain pt_files/h5_files subfolder')
parser.add_argument('--coords_path', type=str, default=None,
                    help='path to coords pt files if needed')
parser.add_argument('--small_coords_path', type=str, default=None,
                    help='path to small coords pt files if needed (only used in graph_ms)')
parser.add_argument('--csv_path',type=str,default=None,help='path to dataset_csv file')
parser.add_argument('--save_graph_path',type=str,default=None,help='path to folder for saving graphs')
## Graph settings
parser.add_argument('--model_type', type=str, choices=['graph', 'graph_ms'], default='graph', help='type of model')
parser.add_argument('--graph_edge_distance',type=int,default=750,help="Maximum distance between nodes in graph to add edges.")
parser.add_argument('--offset',type=int,default=512,help="The offset applied to the larger patches in graph_ms, which will typically be half of the size of the smaller magnification patches. This is needed due to coords being top-left rather than centre")
parser.add_argument('--ms_features',choices=["naive","seperate_zero","seperate_avg"],default="naive",help="whether to assume all patch features are the same (naive) or keep them separate across magnifications")
parser.add_argument('--no_auto_skip', default=False, action='store_true')
args = parser.parse_args()



def extract_graphs():
    ## Generate the graphs and adjacency matrices for WSIs and save to files to avoid repeating this process during training/testing
    ## These will be saved in .pt files containing the features and adjacencies 
    
    slide_data = pd.read_csv(args.csv_path)
    slide_ids = slide_data['slide_id']

    if not os.path.isdir(args.save_graph_path):
        os.mkdir(args.save_graph_path)
        os.mkdir(args.save_graph_path+"/features")
        os.mkdir(args.save_graph_path+"/adj")

    dest_files = os.listdir(args.save_graph_path+"/features")

    data_dir = os.path.join(args.data_root_dir, args.features_folder)
    small_data_dir = os.path.join(args.data_root_dir, args.small_features_folder)

    total_time_elapsed = 0.0

    total = len(slide_ids)
    for i in range(total):
        slide_id = str(slide_ids[i])
        
        if slide_id+'_features.pt' in dest_files and not args.no_auto_skip:
            print('skipped {}'.format(slide_id))
            continue
            
        time_start = time.time()

        full_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id))
        features = torch.load(full_path)
        
        if args.model_type == 'graph':
            raise NotImplementedError("still need to copy this over from datasets/dataset_generic.py")

        elif args.model_type == 'graph_ms':
            small_full_path = os.path.join(small_data_dir, 'pt_files', '{}.pt'.format(slide_id))
            small_features = torch.load(small_full_path)

            with h5py.File(os.path.join(args.coords_path, slide_id+".h5"),'r') as hdf5_file:
                coordinates = hdf5_file['coords'][:]
            with h5py.File(os.path.join(args.small_coords_path, slide_id+".h5"),'r') as hdf5_file:
                small_coordinates = hdf5_file['coords'][:]

            ## naive features treats high-mag and low-mag features as interchangable, seperate features separate the two with either zero padding or mean padding
            if args.ms_features == 'naive':
                x_big = features.clone().detach()
                x_small = small_features.clone().detach()
            elif args.ms_features == 'seperate_zero':
                features = torch.nn.functional.pad(features, (0, 1024), mode='constant', value=0)
                x_big = features.clone().detach()
                small_features = torch.nn.functional.pad(small_features,(1024,0),mode='constant', value=0)
                x_small = small_features.clone().detach()
            elif args.ms_features == 'seperate_avg':
                big_avg = torch.mean(features,dim=0)
                small_avg = torch.mean(small_features,dim=0)
                features = torch.cat((features,small_avg.repeat(features.shape[0],1)),1)
                x_big = features.clone().detach()
                small_features = torch.cat((big_avg.repeat(small_features.shape[0],1),small_features),1)
                x_small = small_features.clone().detach()
            else:
                raise NotImplementedError("Didn't get a correct input for ms_features")

            total_coords = len(coordinates)+len(small_coordinates)

            ## calculate edges for big patches
            distances = pdist(coordinates, 'euclidean')
            dist_matrix = squareform(distances)
            adj = (dist_matrix <= args.graph_edge_distance).astype(np.float32)
            adj = (adj - np.identity(adj.shape[0])).astype(np.float32)
            edge_indices = np.transpose(np.triu(adj,k=1).nonzero())
            adj_big = torch.from_numpy(edge_indices).t().contiguous()
        
            ## calculate edges for small patches - maximum distance for neighbours is halved for this
            distances = pdist(small_coordinates, 'euclidean')
            dist_matrix = squareform(distances)
            adj = (dist_matrix <= (args.graph_edge_distance/2)).astype(np.float32)
            adj = (adj - np.identity(adj.shape[0])).astype(np.float32)
            edge_indices = np.transpose(np.triu(adj,k=1).nonzero())
            adj_small = torch.from_numpy(edge_indices).t().contiguous()
            adj_small = torch.add(adj_small,coordinates.shape[0])

            ## calculate edges between magnifications with offset used to properly overlay them
            distances = cdist(coordinates+args.offset,small_coordinates, 'euclidean')
            adj = (distances <= 2*args.offset).astype(np.float32)
            edge_indices = np.transpose(adj.nonzero())
            adj_between = torch.from_numpy(edge_indices).t().contiguous()
            adj_between[1] += coordinates.shape[0]

            ## combine the edges and features
            adj = torch.cat((adj_big, adj_small, adj_between),dim=1)
            x = torch.cat((x_big,x_small),dim=0)

            adj_checker = [str(digit.item()) for digit in adj[0]]
            adj_checker += [str(digit.item()) for digit in adj[1]]
            x_checker = [str(digit) for digit in range(len(x))]
            solo_nodes = [node for node in x_checker if node not in adj_checker]
        
    
            ## save the features and adj as a single .np file. Need to make sure theres a folder made with mkdir before this part 
            torch.save(x, os.path.join(args.save_graph_path, 'features', str(slide_id)+'_features.pt'))
            torch.save(adj, os.path.join(args.save_graph_path, 'adj', str(slide_id)+'_adj.pt'))
        
        time_elapsed = time.time() - time_start
        total_time_elapsed += time_elapsed
        print("Saved graph {} with features shape {} and adj shape {}, taking {} s. Progress {}/{}".format(slide_id,x.shape,adj.shape, round(time_elapsed,4),i+1,total))
    print("total time: {}".format(total_time_elapsed))

if __name__ == "__main__":
    extract_graphs()


