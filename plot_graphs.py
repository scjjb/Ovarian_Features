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
parser.add_argument('--graph_path',type=str,default=None,help='path to folder containing graphs. Plots will be saved in a subfolder here.')
parser.add_argument('--csv_path',type=str,default=None,help='path to dataset_csv file')
parser.add_argument('--coords_path', type=str, default=None,help='path to coords pt files')
parser.add_argument('--small_coords_path', type=str, default=None,help='path to small coords pt files (only used in graph_ms)')
parser.add_argument('--model_type', type=str, choices=['graph', 'graph_ms'], default='graph', help='type of model')
parser.add_argument('--offset',type=int,default=512,help="The offset applied to the larger patches in graph_ms, which will typically be half of the size of the smaller magnification patches. This is needed due to coords being top-left rather than centre")
parser.add_argument('--plot_graph',choices=["together","seperate"],help="whether to indicate the magnifications of nodes/edges")
parser.add_argument('--max_plots',type=int,default=100,help="maximum number of graphs to plot")
args = parser.parse_args()


def plot_graphs():
    slide_data = pd.read_csv(args.csv_path)
    slide_ids = slide_data['slide_id']
    
    total = min(len(slide_ids),args.max_plots)

    assert os.path.isdir(args.graph_path)
    plot_path = os.path.join(args.graph_path,"plots")
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)

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
            plt.close('all')

        elif args.plot_graph=="seperate":
            # have to reverse engineer which edges were big, small, and between magnifications
            # using the fact that the edges were ordered big, small, between when extracted, with big patches being indexed 0 to len(coordinates)-1, and small patches indexed higher
            # could equally have been done by looping through to see which edges had nodes above/below len(coordinates)
            big_indices = torch.tensor(range(len(coordinates)))
            from_big_adj = torch.nonzero(torch.isin(adj[0], big_indices)).squeeze(1)
            from_big_adj2 = torch.concat((torch.tensor([-1]),from_big_adj))
            idx_gaps = torch.concat((from_big_adj,from_big_adj[-1:]+1))-from_big_adj2
            
            last_big_idx = np.where(idx_gaps != 1)[0][0]-1
            first_between_idx = from_big_adj[last_big_idx+1]

            adj_big = torch.stack((adj[0][:last_big_idx],adj[1][:last_big_idx]))
            adj_small = torch.stack((adj[0][last_big_idx+1:first_between_idx],adj[1][last_big_idx+1:first_between_idx]))
            adj_between = torch.stack((adj[0][first_between_idx:],adj[1][first_between_idx:]))

            big_nodes = list(range(len(coordinates)))
            
            ## add the offset
            big_pos = {node: tuple(coord+args.offset) for node, coord in zip(big_nodes, coordinates)}
            plot_data = torch_geometric.data.Data(x=x_big, edge_index=adj_big)
            g = torch_geometric.utils.to_networkx(plot_data, to_undirected=True)
            options = {"node_size": 2, "node_color": "black", "edge_color": "red", "width": 0.8, "style":"-."}
            nx.draw(g,pos=big_pos, ax=ax,**options)
            
            small_nodes = list(range(len(small_coordinates)))
            small_pos = {node: tuple(coord) for node, coord in zip(small_nodes, small_coordinates)}
            #print("shape",x_small.shape)
            adj_small = torch.add(adj_small,-coordinates.shape[0])
            plot_data = torch_geometric.data.Data(x=x_small, edge_index=adj_small)
            g = torch_geometric.utils.to_networkx(plot_data, to_undirected=True)
            options = {"node_size": 2, "node_color": "black", "edge_color": "black", "width": 0.8, "style": "-"}
            nx.draw(g,pos=small_pos, ax=ax,**options)

            all_nodes = list(range(len(all_coordinates)))
            all_pos = {node: tuple(coord) for node, coord in zip(all_nodes, all_coordinates)}
            plot_data = torch_geometric.data.Data(x=features, edge_index=adj_between)
            g = torch_geometric.utils.to_networkx(plot_data, to_undirected=True)
            options = {"node_size": 0.1, "node_color": "black", "edge_color":"tab:blue", "width": 0.8, "style": "-"}
            nx.draw(g,pos=all_pos, ax=ax,**options)
            
            matplotlib.use("Agg")
            plt.gca().invert_yaxis()
        
            fig.savefig(plot_path+"/graph_{}.png".format(slide_id),bbox_inches="tight")
            plt.close()
        print("Plotted graph {}. Progress {}/{}".format(slide_id,i+1,total))

if __name__ == "__main__":
    plot_graphs()
