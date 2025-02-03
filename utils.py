import time
import numpy as np
import torch
import os
import random
from dgl.data import FraudYelpDataset, FraudAmazonDataset, TolokersDataset, QuestionsDataset
import dgl
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from torch_geometric.utils import degree
from dgl.data.utils import load_graphs
from scipy.sparse import csgraph
import pygod.utils as pygodutils

from name import *

def set_seed(seed):
    if seed == 0:
        seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed

def load_data(data):
    datadir = os.path.join(DATADIR, data)
    datapath = os.path.join(datadir, data)
    if data == TFINANCE:
        graph, label_dict = load_graphs(datapath)
        graph = graph[0]
        graph.ndata['label'] = graph.ndata['label'].argmax(1)
    elif data in [TSOCIAL, DGRAPHFIN, ELLIPTIC]:
        graph, label_dict = load_graphs(datapath)
        graph = graph[0]

    elif data == YELP:
        dataset = FraudYelpDataset(raw_dir=datadir)
        graph = dataset[0]
        graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        graph = dgl.add_self_loop(graph)
    elif data == AMAZON:
        dataset = FraudAmazonDataset(raw_dir=datadir)
        graph = dataset[0]
        graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        graph = dgl.add_self_loop(graph)
    elif data == TOLOKERS:
        dataset = TolokersDataset(raw_dir=datadir)
        graph = dataset[0]
        graph = dgl.to_homogeneous(dataset[0], ndata=['feat', 'label', 'train_mask', 'val_mask', 'test_mask'])
        graph = dgl.add_self_loop(graph)
        graph.ndata['feature'] = graph.ndata['feat']
    elif data == QUESTIONS:
        dataset = QuestionsDataset(raw_dir=datadir)
        graph = dataset[0]
        graph = dgl.to_homogeneous(dataset[0], ndata=['feat', 'label', 'train_mask', 'val_mask', 'test_mask'])
        graph = dgl.add_self_loop(graph)
        graph.ndata['feature'] = graph.ndata['feat']
    elif data == DGRAPHFIN:
        graph, label_dict = load_graphs(datapath)
        graph = graph[0]
    elif data == REDDIT:
        dataset = pygodutils.load_data(data)
        edge_index = (dataset.edge_index[0], dataset.edge_index[1])
        features = dataset.x
        labels = dataset.y
        graph = dgl.graph(edge_index)
        graph.ndata['label'] = labels
        graph.ndata['feature'] = features
    else:
        print('no such dataset')
        exit(1)

    graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
    graph.ndata['feature'] = graph.ndata['feature'].float()

    features = graph.ndata['feature']
    labels = graph.ndata['label']
    edge_index = graph.edges()
    edge_index = torch.vstack((edge_index[0], edge_index[1]))
    
    index_path = os.path.join(datadir, data + INDEX)
    index = np.loadtxt(index_path, dtype=np.int64)

    return graph, features, labels, edge_index, index

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_lap(edge_index, n):
    edge_indext = edge_index.T
    adjacency = csr_matrix((np.ones(edge_indext.shape[0]), (edge_indext[:, 0], edge_indext[:, 1])), shape=(n, n))

    nadj = csgraph.laplacian(adjacency, normed=True)

    return sparse_mx_to_torch_sparse_tensor(nadj)

def get_infmatrix(edge_index, n, m, eps=0):
    deg = degree(edge_index[0], n) + 1

    deg = torch.sqrt(deg / (2 * m + n))
    deg = torch.where(deg < eps, 0, deg)
    deg = deg.unsqueeze(dim=-1)
    degt = deg.T
    deg = deg.to_sparse()
    degt = degt.to_sparse()

    matrix = torch.spmm(deg, degt)

    return matrix
