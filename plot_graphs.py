import argparse
import os
import pandas as pd
import numpy as np
import torch
import h5py
import torch_geometric
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Configurations for plotting graphs')
parser.add_argument('--graph_path',type=str,default=None,help='path to folder containing graphs')
parser.add_argument('--csv_path',type=str,default=None,help='path to dataset_csv file')
parser.add_argument('--coords_path', type=str, default=None,help='path to coords pt files')
parser.add_argument('--small_coords_path', type=str, default=None,help='path to small coords pt files (only used in graph_ms)')
parser.add_argument('--model_type', type=str, choices=['graph', 'graph_ms'], default='graph', help='type of model')
parser.add_argument('--offset',type=int,default=512,help="The offset applied to the larger patches in graph_ms, which will typically be half of the size of the smaller magnification patches. This is needed due to coords being top-left rather than centre")
parser.add_argument('--plot_path',type=str,default=None,help='path to folder for saving plots')
parser.add_argument('--plot_graph',choices=["together","seperate"],help="whether to indicate the magnifications of nodes/edges")
parser.add_argument('--max_plots',type=int,default=100,help="maximum number of graphs to plot")
args = parser.parse_args()


def plot_graphs():
    slide_data = pd.read_csv(args.csv_path)
    slide_ids = slide_data['slide_id']
    
    total = min(len(slide_ids),args.max_plots)

    if not os.path.isdir(args.plot_path):
        os.mkdir(args.plot_path)

    for i in range(total):
        slide_id = str(slide_ids[i])

        if args.model_type == 'graph_ms':
            with h5py.File(os.path.join(args.coords_path, slide_id+".h5"),'r') as hdf5_file:
                coordinates = hdf5_file['coords'][:]
            with h5py.File(os.path.join(args.small_coords_path, slide_id+".h5"),'r') as hdf5_file:
                small_coordinates = hdf5_file['coords'][:]

        features = torch.load(os.path.join(args.graph_path, 'features', str(slide_id)+'_features.pt'))
        adj = torch.load(os.path.join(args.graph_path, 'adj', str(slide_id)+'_adj.pt'))


        fig, ax = plt.subplots()
        all_coordinates = np.append(coordinates+args.offset,small_coordinates,axis=0)
        x_big = features[:len(coordinates)]
        x_small = features[-len(small_coordinates):]
        
        if args.plot_graph=="together": 
            fig, ax = plt.subplots()
            all_coordinates = np.append(coordinates+args.offset,small_coordinates,axis=0)
            
            ## convert coordinates to dictionary for nx
            nodes = list(range(len(all_coordinates)))
            pos = {node: tuple(coord) for node, coord in zip(nodes, all_coordinates)}
            data = features
            plot_data = torch_geometric.data.Data(x=data, edge_index=adj)
            g = torch_geometric.utils.to_networkx(plot_data, to_undirected=True)
            options = {"node_size": 2, "node_color": "black", "width": 0.8, "style":"-"}
            nx.draw(g,pos=pos, ax=ax,**options)
            matplotlib.use("Agg")
            # Reflect the plot along the y-axis
            matplotlib.pyplot.gca().invert_yaxis()
            fig.savefig(os.path.join(args.plot_path,slide_id+".png"),bbox_inches="tight")
            plt.close(fig)

        elif args.plot_graph=="seperate":
            raise NotImplementedError("Need to make code to go through adj and assign each edge to groups based on big->big, small->small, and big->small")



        print("Plotted graph {}. Progress {}/{}".format(slide_id,i+1,total))

if __name__ == "__main__":
    plot_graphs()
