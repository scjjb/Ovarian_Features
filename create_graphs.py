import argparse
import os 
import numpy as np
import pandas as pd
import torch
from datasets.dataset_generic import Generic_MIL_Dataset


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
args = parser.parse_args()


def extract_graphs():
    ## Generate the graphs and adjacency matrices for WSIs and save to files to avoid repeating this process during training/testing
    ## These will be saved in .pt files containing the features and adjacencies 
    
    dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                            data_dir = os.path.join(args.data_root_dir, args.features_folder),
                            small_data_dir = os.path.join(args.data_root_dir, args.small_features_folder),
                            max_patches_per_slide=np.inf,
                            perturb_variance=0,
                            number_of_augs=0,
                            coords_path = args.coords_path,
                            small_coords_path = args.small_coords_path,
                            shuffle = False,
                            seed = 1,
                            print_info = True,
                            label_dict = {'high_grade':0,'low_grade':1,'clear_cell':2,'endometrioid':3,'mucinous':4},
                            patient_strat=False,
                            data_h5_dir=None,
                            data_slide_dir=None,
                            slide_ext=None,
                            pretrained=True,
                            custom_downsample=None,
                            target_patch_size=None,
                            model_architecture = None,
                            model_type = args.model_type,
                            batch_size = None,
                            graph_edge_distance = args.graph_edge_distance,
                            offset = args.offset,
                            plot_graph = False,
                            ms_features = args.ms_features,
                            ignore=[])

    slide_data = pd.read_csv(args.csv_path)
    slide_ids = slide_data['slide_id']

    if not os.path.isdir(args.save_graph_path):
        os.mkdir(args.save_graph_path)
        os.mkdir(args.save_graph_path+"/features")
        os.mkdir(args.save_graph_path+"/adj")

    total = len(dataset)
    for i in range(total):
        features, adj, label = dataset[i]
        ## save the features and adj as a single .np file. Need to make sure theres a folder made with mkdir before this part 
        slide_id = slide_ids[i]
        torch.save(features, os.path.join(args.save_graph_path, 'features', str(slide_id)+'_features.pt'))
        torch.save(adj, os.path.join(args.save_graph_path, 'adj', str(slide_id)+'_adj.pt'))
        print("Saved graph {} with features shape {} and adj shape {}. Progress {}/{}".format(slide_id,features.shape,adj.shape,i+1,total))


if __name__ == "__main__":
    extract_graphs()


