import torch
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
import torch.nn.functional as F

"""
TopK Pooling Graph Classification Network (3 fc layers)
args:
    pooling_factor: CURRENTLY NOT IMPLEMENTED
    embedding_size: FILL THESE IN 
    max_nodes: CURRENTLY NOT IMPLEMENTED
    pooling_layers: NOT CURRENTLY IMPLEMENTED
model adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_topk_pool.py
"""
class Graph_Model(torch.nn.Module):
    def __init__(self, pooling_factor = 0.6, embedding_size = 256, max_nodes = 250, pooling_layers = 3,num_features=196, num_classes=2, drop_out=0.5):
        super().__init__()
        
        self.drop_out = drop_out

        self.conv1 = GraphConv(num_features, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.lin1 = torch.nn.Linear(256,128)
        self.lin2 = torch.nn.Linear(128,64)
        self.lin3 = torch.nn.Linear(64, num_classes)
    
    def forward(self, x, adj, training=False):
        x, edge_index = x.squeeze(), adj.squeeze()

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index)
        x2 = torch.cat([gmp(x,batch), gap(x,batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3
        x = F.dropout(x, p=self.drop_out, training=training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.drop_out, training=training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.drop_out, training=training)
        x = self.lin3(x)

        logits = x
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        #output = F.log_softmax(self.lin3(x), dim=-1)
        return logits, Y_prob, Y_hat, {}, {}

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1.to(device)
        self.pool1.to(device)
        self.conv2.to(device)
        self.pool2.to(device)
        self.conv3.to(device)
        self.pool3.to(device)
        self.lin1.to(device)
        self.lin2.to(device)
        self.lin3.to(device)
