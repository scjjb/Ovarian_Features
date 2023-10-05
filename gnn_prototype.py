import os.path as osp
import time
from math import ceil
import os 
import numpy as np
import pandas as pd
import sklearn

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
#from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch_geometric.data import Batch, Dataset, Data, DataLoader
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

import h5py

from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, roc_auc_score

epochs = 10
#total_slides = 400
max_nodes = 250
#max_nodes = 30000
#num_classes = 2
learning_rate = 0.0001

class GraphDataset(Dataset):
    def __init__(self, root, node_features_dir, coordinates_dir, labels_file, max_nodes = 250, transform=None, pre_transform=None):
        #super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.node_features_dir = node_features_dir
        self.coordinates_dir = coordinates_dir
        self.labels_file = labels_file
        self.max_nodes = max_nodes

    @property
    def dir_names(self):
        return [self.node_features_dir, self.coordinates_dir]
    
    def process(self):
        #node_features_files = os.listdir(os.path.join(self.root, self.node_features_dir))
        #coordinates_files = os.listdir(os.path.join(self.root, self.coordinates_dir))
        labels_df = pd.read_csv(self.labels_file)
        slides = labels_df['slide_id']
        # Create a list of Data objects, each representing a graph
        data_list = []
        label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1}

        #for i in range(len(node_features_files)):
        ## just doing 10 slides to check its working. Will want to change functionality to be more efficient and choose slides from dataset_csv
        #slide_names = []
        print("processing dataset")
        total_slides = len(slides)
        #total_slides=500
        for i in tqdm(range(total_slides)):
            #print("processing slide {}".format(i))
            ## Need to edit this to not be doing a read.csv but instead loading from pt file
            #node_features = pd.read_csv(os.path.join(self.dir, self.node_features_dir, node_features_files[i])).values
            #coordinates = pd.read_csv(os.path.join(self.raw_dir, self.coordinates_dir, coordinates_files[i])).values
            slide_name = slides[i]
            #print(slide_name)
            #print(slide_name)
            #print("load path",os.path.join(self.root, self.node_features_dir, slide_name+".pt"))
            node_features = torch.load(os.path.join(self.root, self.node_features_dir, str(slide_name)+".pt"))
            with h5py.File(os.path.join(self.root, self.coordinates_dir, str(slide_name)+".h5"),'r') as hdf5_file:
                coordinates = hdf5_file['coords'][:]
            if len(node_features)>self.max_nodes:
                node_features=node_features[:max_nodes]
                coordinates=coordinates[:max_nodes]
            #print("len coords",len(coordinates))
            adjacency_matrix = to_dense_adj(torch.tensor(coordinates), max_num_nodes=min(len(coordinates),max_nodes), edge_attr=None)
            #print("node_features_file",node_features_files[i])
            #print("coordinates_file",coordinates_files[i])
            x = torch.tensor(node_features, dtype=torch.float)
            adj = torch.tensor(adjacency_matrix, dtype=torch.float).squeeze(0)
            #pos = torch.tensor(coordinates, dtype=torch.int)
            label_name = labels_df[labels_df['slide_id']==slide_name]['label'].values[0]
            #print("label_name",label_name)
            #print("label",int(label_dict[label_name]))
            label = torch.tensor(int(label_dict[label_name]))
            #print("label",label)
            data = Data(x=x, adj=adj, y=label)
            data_list.append(data)
        #self.data, self.slices = self.collate(data_list)
        self.data = data_list
        self.y = [data['y'] for data in data_list]
        #self.num_classes = len(np.unique(self.y))
        #print("num classes",self.num_classes)
        #print("self.y",self.y)
        #self.num_classes = len(np.unique(self.y))

    def num_classes(self):
        return len(np.unique(self.y))

    def collate(self, batch):
        #print("batch:", batch)
        #img = torch.cat([item['x'] for item in batch], dim = 0)
        #coords = np.vstack([item['pos'] for item in batch])
        #return [img, coords]
        return Batch.from_data_list(batch)

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    #def __repr__(self):
    #    return f'{self.__class__.__name__}({len(self)})'

dataset = GraphDataset(root='../mount_outputs', node_features_dir='features/ovarian_leeds_hipt4096_features_normalised/pt_files', coordinates_dir='patches/ovarian_leeds_mag20x_patch8192_fp/patches', labels_file = 'dataset_csv/ESGO_train_all.csv')
#dataset = GraphDataset(root='../', node_features_dir='mount_i/features/ovarian_dataset_features_256_patches_20x/pt_files', coordinates_dir='mount_outputs/patches/512_patches_40x/patches', labels_file = 'dataset_csv/ESGO_train_all.csv')

dataset.process()
print(dataset)
#loader = DataLoader(dataset, batch_size=1, shuffle=True)
#for batch in loader:
#    print(batch)
#    print(batch['x'])

#path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                #'PROTEINS_dense')
#dataset = TUDataset(
#    path,
#    name='PROTEINS',
#    transform=T.ToDense(max_nodes),
#    pre_filter=lambda data: data.num_nodes <= max_nodes,
#)


#dataset = dataset.shuffle()
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
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, self.conv1(x0, adj, mask).relu())
        x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        x3 = self.bn(3, self.conv3(x2, adj, mask).relu())

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = self.lin(x).relu()

        return x


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        pooling_factor = 0.6
        embedding_size = 256

        num_nodes = ceil(pooling_factor * max_nodes)
        self.gnn1_pool = GNN(dataset.num_features, embedding_size, num_nodes)
        self.gnn1_embed = GNN(dataset.num_features, embedding_size, embedding_size, lin=False)

        num_nodes = ceil(pooling_factor * num_nodes)
        self.gnn2_pool = GNN(3 * embedding_size, embedding_size, num_nodes)
        self.gnn2_embed = GNN(3 * embedding_size, embedding_size, embedding_size, lin=False)

        num_nodes = ceil(pooling_factor * num_nodes)
        self.gnn3_pool = GNN(3 * embedding_size, embedding_size, num_nodes)
        self.gnn3_embed = GNN(3 * embedding_size, embedding_size, embedding_size, lin=False)

        self.lin1 = torch.nn.Linear(3 * embedding_size, embedding_size)
        self.lin2 = torch.nn.Linear(embedding_size, dataset.num_classes())

    def forward(self, x, adj, mask=None):
        ## removed mask from these
        #print("adj size",adj.size())
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        s = self.gnn3_pool(x, adj)
        x = self.gnn3_embed(x, adj)
        #x = self.gnn3_embed(x, adj)

        x, adj, l3, e3 = dense_diff_pool(x, adj, s)

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

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, capturable = True)


def train(epoch,weight):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        ## removed data.mask input
        #print("x shape",data.x.shape)
        #print("adj shape",data.adj.squeeze(0).shape)
        #print("training data label",data.y)
        try:
            output, _, _ = model(data.x, data.adj)
        except:
            print("broken slide with data",data, "label", data.y)
            assert 1==2
        #loss = F.cross_entropy(output, data.y.view(-1), weight=weight)
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
#print("train_dataset",train_dataset)
#print("y",y)
#print("np.unique(y)",np.unique(y))
#print("torch.tensor(y)",torch.tensor(y))
weight=torch.tensor(sklearn.utils.class_weight.compute_class_weight('balanced',classes=np.unique(y),y=y.values.reshape(-1))).to(device).float()
print("loss weight",weight)
for epoch in range(epochs):
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
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

preds, labels = test_all(train_loader)
preds_int = [int(pred) for pred in preds]
print("train set confusion matrix (predicted x axis, true y axis)")
print(confusion_matrix(labels,preds_int))
try:
    print("train set balanced accuracy: ",balanced_accuracy_score(labels,preds_int)
            )
    print("  AUC: ",roc_auc_score(labels,preds)
            )
    print( "  F1:",f1_score(labels,preds_int))
except:
    print("train scores didn't work")

preds, labels = test_all(val_loader)
preds_int = [int(pred) for pred in preds]
print("val set confusion matrix (predicted x axis, true y axis): \n")
print(confusion_matrix(labels,preds_int),"\n")
try:
    print("val set balanced accuracy: ",balanced_accuracy_score(labels,preds_int), "  AUC: ",roc_auc_score(labels,preds), "  F1:",f1_score(labels,preds))
except:
    print("val scores didn't work")

preds, labels = test_all(test_loader)
preds_int = [int(pred) for pred in preds]
print("test set confusion matrix (predicted x axis, true y axis): \n")
print(confusion_matrix(labels,preds_int),"\n")
try:
    print("test set balanced accuracy: ",balanced_accuracy_score(labels,preds_int), "  AUC: ",roc_auc_score(labels,preds), "  F1:",f1_score(labels,preds))
except:
    print("test scores didn't work")
