import os.path as osp
import time
from math import ceil
import os 
import numpy as np
import pandas as pd
import sklearn

import torch
from torch import nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, GCNConv, dense_diff_pool, dense_mincut_pool
from torch_geometric.data import Batch, Dataset, Data, DataLoader
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

import h5py

from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, roc_auc_score
import warnings

import argparse
parser = argparse.ArgumentParser(description='Graph neural network classifier for subtyping')
parser.add_argument('--epochs', type=int, default=2,help='training epochs')
parser.add_argument('--max_nodes', type=int, default=5000,help='max nodes per graph')
parser.add_argument('--pooling_factor', type=float, default=0.6,help='proportion of graph nodes retained in each graph pooling layer')
parser.add_argument('--pooling_layers',type=int,default=3,help='number of graph pooling layers')
parser.add_argument('--learning_rate', type=float, default=0.001,help='model learning rate')
parser.add_argument('--data_root_dir', type=str, default="../mount_outputs/features",help='directory containing features folders')
parser.add_argument('--features_folder', type=str, default="graph_ovarian_leeds_resnet50imagenet_256patch_features_5x/pt_files",help='folder within data_root_dir containing the features - must contain pt_files/h5_files subfolder')
parser.add_argument('--coords_dir', type=str, default="../mount_outputs/patches/ovarian_leeds_mag40x_patchgraph2048and1024_fp/patches/big",help="directory containing coordinates files")
parser.add_argument('--csv_path',type=str,default='dataset_csv/miniprototype.csv',help='path to dataset_csv label file')
parser.add_argument('--graph_pooling', type=str,choices=["diff","mincut"],default="diff",help="type of graph pooling to use - dense_diff_pool or dense_mincut_pool")
args = parser.parse_args()

warnings.simplefilter(action='ignore', category=FutureWarning)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

if args.graph_pooling == 'mincut':
    print("mincut not implemented yet, will need to change pooling layers to be linear rather than GNN layers as in https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_mincut_pool.py")
    raise NotImplementedError

class GraphDataset(Dataset):
    def __init__(self, node_features_dir, coordinates_dir, labels_file, max_nodes = 250, transform=None, pre_transform=None):
        #super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.node_features_dir = node_features_dir
        self.coordinates_dir = coordinates_dir
        self.labels_file = labels_file
        self.max_nodes = max_nodes
        self.max_nodes_in_dataset = 0

    @property
    def dir_names(self):
        return [self.node_features_dir, self.coordinates_dir]
    
    def process(self):
        labels_df = pd.read_csv(self.labels_file)
        slides = labels_df['slide_id']
        # Create a list of Data objects, each representing a graph
        data_list = []
        label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1}

        print("processing dataset")
        total_slides = len(slides)
        for i in tqdm(range(total_slides)):
            slide_name = slides[i]
            node_features = torch.load(os.path.join(self.node_features_dir, str(slide_name)+".pt"))
            with h5py.File(os.path.join(self.coordinates_dir, str(slide_name)+".h5"),'r') as hdf5_file:
                coordinates = hdf5_file['coords'][:]
            if len(node_features)>self.max_nodes:
                node_features=node_features[:self.max_nodes]
                coordinates=coordinates[:self.max_nodes]
            if len(coordinates)>self.max_nodes_in_dataset:
                self.max_nodes_in_dataset=len(coordinates)
            adjacency_matrix = to_dense_adj(torch.tensor(coordinates), max_num_nodes=min(len(coordinates),self.max_nodes), edge_attr=None)
            x = node_features.clone().detach()
            adj = adjacency_matrix.clone().detach().squeeze(0)
            label_name = labels_df[labels_df['slide_id']==slide_name]['label'].values[0]
            label = torch.tensor(int(label_dict[label_name]))
            data = Data(x=x, adj=adj, y=label)
            data_list.append(data)
        self.data = data_list
        self.y = [data['y'] for data in data_list]

    def num_classes(self):
        return len(np.unique(self.y))

    def collate(self, batch):
        return Batch.from_data_list(batch)

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


data_dir = os.path.join(args.data_root_dir, args.features_folder)
dataset = GraphDataset(node_features_dir=data_dir, coordinates_dir=args.coords_dir, labels_file = args.csv_path,max_nodes=args.max_nodes)

dataset.process()

n = (len(dataset) + 4) // 5
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DenseDataLoader(test_dataset, batch_size=1)
val_loader = DenseDataLoader(val_dataset, batch_size=1)
train_loader = DenseDataLoader(train_dataset, batch_size=1)

print("train slides:",len(train_loader))
print("val slides:",len(val_loader))
print("test slides",len(test_loader))

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        #self.conv1 = GCNConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        #self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        ## removed the layer which keeps the same dimension because that seems ridiculous

        #self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        #self.conv3 = GCNConv(hidden_channels, out_channels, normalize)
        #self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        #x1 = self.conv1(x0, adj)
        #x2 = self.conv2(x1, adj)
        #x3 = self.conv3(x2, adj)
        x1 = self.bn(1, self.conv1(x0, adj).relu())
        x2 = self.bn(2, self.conv2(x1, adj).relu())
        #x3 = self.bn(3, self.conv3(x2, adj).relu())
        
        ## this appending of the different features levels is a bit weird and requires larger layers in the network
        #x = torch.cat([x1, x2, x3], dim=-1)
        x = torch.cat([x1, x2], dim=-1)
        
        if self.lin is not None:
            x = self.lin(x).relu()

        return x


class Net(torch.nn.Module):
    def __init__(self, pooling_factor = 0.6, embedding_size = 256, max_nodes = 250, pooling_layers = 3):
        super().__init__()
        print("largest graph layer size",max_nodes)
        print("dataset.num_features",dataset.num_features)
        num_nodes = ceil(pooling_factor * max_nodes)
        self.gnn1_pool = GNN(dataset.num_features, embedding_size, num_nodes)
        self.gnn1_embed = GNN(dataset.num_features, embedding_size, embedding_size, lin=False)
        
        hidden_layers=[]
        for _ in range(pooling_layers-1):
            num_nodes = ceil(pooling_factor * num_nodes)
            hidden_layers.append(GNN(2 * embedding_size, embedding_size, num_nodes))
            hidden_layers.append(GNN(2 * embedding_size, embedding_size, embedding_size, lin=False))
        
        self.hidden_layers=nn.ModuleList(hidden_layers)
        print("smallest graph layer size",num_nodes)

        self.lin1 = torch.nn.Linear(2 * embedding_size, embedding_size)
        self.lin2 = torch.nn.Linear(embedding_size, dataset.num_classes())
        
        if args.graph_pooling == 'diff':
            self.pooling_layer = dense_diff_pool 
        elif args.graph_pooling == 'mincut':
            self.pooling_layer = dense_mincut_pool
        else: 
            raise NotImplementedError

    def forward(self, x, adj):
        s = self.gnn1_pool(x, adj)
        x = self.gnn1_embed(x, adj)

        x, adj, l1, e1 = self.pooling_layer(x, adj, s)
        
        for i in range(len(self.hidden_layers)):
            if (i % 2) == 0:
                s = self.hidden_layers[i](x, adj)
            else:
                x = self.hidden_layers[i](x, adj)
                x, adj, l2, e2 = self.pooling_layer(x, adj, s)
        
        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2


if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

embedding_size = dataset[0]['x'].shape[1]
model = Net(max_nodes=dataset.max_nodes_in_dataset,pooling_factor=args.pooling_factor,embedding_size=embedding_size, pooling_layers = args.pooling_layers).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, capturable = True)

print("Model parameters:",f'{sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

def train(epoch,weight):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        ## removed data.mask input
        try:
            output, _, _ = model(data.x, data.adj)
        except:
            print("broken slide with data",data, "label", data.y)
            assert 1==2
        loss = F.nll_loss(output, data.y.view(-1), weight=weight)
        loss.backward()
        loss_all += data.y.size(0) * float(loss)
        optimizer.step()
    return loss_all / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        ## removed data.mask from the input
        pred = model(data.x, data.adj)[0].max(dim=1)[1]
        correct += int(pred.eq(data.y.view(-1)).sum())
    return correct / len(loader.dataset)

def test_all(loader):
    model.eval
    correct = 0
    preds=[]
    labels=[]
    for data in loader:
        data = data.to(device)
        ## removed data.mask from the input
        pred = model(data.x, data.adj)[0].max(dim=1)[1]
        preds.append(pred.cpu())
        labels.append(data.y.cpu()[0])
    return preds, labels

best_val_acc = test_acc = 0
times = []
y = pd.DataFrame([data.y.item() for data in train_dataset])
weight=torch.tensor(sklearn.utils.class_weight.compute_class_weight('balanced',classes=np.unique(y),y=y.values.reshape(-1))).to(device).float()
print("loss weight",weight)
for epoch in range(args.epochs):
    start = time.time()
    train_loss = train(epoch,weight)
    train_acc = test(train_loader)
    val_acc = test(val_loader)
    if val_acc > best_val_acc:
        test_acc = test(test_loader)
        best_val_acc = val_acc
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} '
          f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s \n")

preds, labels = test_all(train_loader)
preds_int = [int(pred) for pred in preds]
print("train set confusion matrix (predicted x axis, true y axis): ")
print(confusion_matrix(labels,preds_int))
try:
    print("train set balanced accuracy: ",balanced_accuracy_score(labels,preds_int)
            )
    print("AUC: ",roc_auc_score(labels,preds)
            )
    print( "F1:",f1_score(labels,preds_int),"\n")
except:
    print("train scores didn't work \n")

preds, labels = test_all(val_loader)
preds_int = [int(pred) for pred in preds]
print("val set confusion matrix (predicted x axis, true y axis): ")
print(confusion_matrix(labels,preds_int))
try:
    print("val set balanced accuracy: ",balanced_accuracy_score(labels,preds_int), "  AUC: ",roc_auc_score(labels,preds), "  F1:",f1_score(labels,preds),"\n")
except:
    print("val scores didn't work \n")

preds, labels = test_all(test_loader)
preds_int = [int(pred) for pred in preds]
print("test set confusion matrix (predicted x axis, true y axis): ")
print(confusion_matrix(labels,preds_int))
try:
    print("test set balanced accuracy: ",balanced_accuracy_score(labels,preds_int), "  AUC: ",roc_auc_score(labels,preds), "  F1:",f1_score(labels,preds))
except:
    print("test scores didn't work")
print(balanced_accuracy_score(labels,preds_int))
print(roc_auc_score(labels,preds))
print(f1_score(labels,preds))
