import torch
from torch_geometric.nn import GraphConv, GATv2Conv 
from torch_geometric.nn import TopKPooling, SAGPooling 
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from torch import nn
import torch.nn.functional as F

"""
TopK Pooling Graph Classification Network (3 fc layers)
args:
    pooling_factor: proportion of nodes after each pooling layer
    embedding_size: size of graph node embeddings
    max_nodes: CURRENTLY NOT IMPLEMENTED
    pooling_layers: number of graph message passing and pooling layers
model adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_topk_pool.py
"""
class Graph_Model(torch.nn.Module):
    def __init__(self, pooling_factor = 0.8, pooling_layers = 3, message_passings = 1, gat_heads = 1, embedding_size = 128, num_features=196, num_classes=2, max_nodes=250, drop_out=0.5, message_passing = 'standard', pooling = 'topk'):
        super().__init__()
        
        self.drop_out = drop_out
        self.message_passings = message_passings
        graph_layers=[]
        
        if message_passing == 'standard':
            graph_layers.append(GraphConv(num_features, embedding_size))
            for _ in range(message_passings-1):
                 graph_layers.append(GraphConv(embedding_size, embedding_size))
        elif message_passing == 'gatv2':
            graph_layers.append(GATv2Conv(num_features, embedding_size, heads = gat_heads))
            for _ in range(message_passings-1):
                graph_layers.append(GATv2Conv(embedding_size, embedding_size))
        else:
            raise NotImplementedError

        if pooling == 'topk':
            graph_layers.append(TopKPooling(embedding_size, ratio=pooling_factor))
        elif pooling == 'sag':
            graph_layers.append(SAGPooling(embedding_size, ratio=pooling_factor))
        else:
            raise NotImplementedError

        for _ in range(pooling_layers-1):
            for _ in range(message_passings):
                if message_passing == 'standard':
                    graph_layers.append(GraphConv(embedding_size, embedding_size))
                elif message_passing == 'gatv2':
                    graph_layers.append(GATv2Conv(embedding_size, embedding_size))
            if pooling == 'topk':
                graph_layers.append(TopKPooling(embedding_size, ratio=pooling_factor))
            elif pooling == 'sag':
                graph_layers.append(SAGPooling(embedding_size, ratio=pooling_factor))
        
        self.graph_layers=nn.ModuleList(graph_layers)
        self.lin1 = torch.nn.Linear(2*embedding_size,embedding_size)
        self.lin2 = torch.nn.Linear(embedding_size,int(embedding_size/2))
        self.lin3 = torch.nn.Linear(int(embedding_size/2), num_classes)
    
    def forward(self, x, adj, training=False):
        x, edge_index = x.squeeze(), adj.squeeze()
        xhidden = None
        for i in range(len(self.graph_layers)):
            if ((i+1) % (self.message_passings+1)) != 0:
                x = F.relu(self.graph_layers[i](x, edge_index))
            else:
                x, edge_index, _, batch, _, _ = self.graph_layers[i](x, edge_index)
                if xhidden is None:
                    xhidden = torch.cat([gmp(x,batch), gap(x,batch)], dim=1)
                else:
                    xhidden = xhidden + torch.cat([gmp(x,batch), gap(x,batch)], dim=1)

        x = xhidden 
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.drop_out, training=training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.drop_out, training=training)
        x = self.lin3(x)

        logits = x
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        return logits, Y_prob, Y_hat, {}, {}

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.conv1.to(device)
        #self.pool1.to(device)
        self.graph_layers.to(device)
        #self.convN.to(device)
        #self.poolN.to(device)
        self.lin1.to(device)
        self.lin2.to(device)
        self.lin3.to(device)
