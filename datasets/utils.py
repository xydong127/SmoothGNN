import os
import dgl
import torch
from dgl.data import FraudYelpDataset, FraudAmazonDataset, TolokersDataset, QuestionsDataset
import numpy as np
from dgl.data.utils import load_graphs
import time
from sklearn.model_selection import train_test_split
import random
import scipy.io as sio
import scipy.sparse as sp
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import dgl 
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
    datapath = os.path.join(os.path.join(DATADIR, data), data)
    if data == TFINANCE:
        graph, label_dict = load_graphs(datapath)
        graph = graph[0]
        graph.ndata['label'] = graph.ndata['label'].argmax(1)
    elif data in [TSOCIAL, DGRAPHFIN, ELLIPTIC]:
        graph, label_dict = load_graphs(datapath)
        graph = graph[0]

    elif data == YELP:
        dataset = FraudYelpDataset(raw_dir=data)
        graph = dataset[0]
        graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        graph = dgl.add_self_loop(graph)
    elif data == AMAZON:
        dataset = FraudAmazonDataset(raw_dir=data)
        graph = dataset[0]
        graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        graph = dgl.add_self_loop(graph)
    elif data == TOLOKERS:
        dataset = TolokersDataset(raw_dir=data)
        graph = dataset[0]
        graph = dgl.to_homogeneous(dataset[0], ndata=['feat', 'label', 'train_mask', 'val_mask', 'test_mask'])
        graph = dgl.add_self_loop(graph)
        graph.ndata['feature'] = graph.ndata['feat']
    elif data == QUESTIONS:
        dataset = QuestionsDataset(raw_dir=data)
        graph = dataset[0]
        graph = dgl.to_homogeneous(dataset[0], ndata=['feat', 'label', 'train_mask', 'val_mask', 'test_mask'])
        graph = dgl.add_self_loop(graph)
        graph.ndata['feature'] = graph.ndata['feat']
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

    y = graph.ndata['label']
    
    return y


def split_train_val_test(data, y):
    normalinds = []
    abnormalinds = []
    wronglabels = []
    for i, label in enumerate(y):
        if int(label) == 0:
            normalinds.append(i)
        elif int(label) == 1:
            abnormalinds.append(i)
        else:
            wronglabels.append(label)
    
    if wronglabels:
        print("Exist wrong label: {}".format(torch.unique(torch.LongTensor(wronglabels), return_counts=True)))

    normal = np.array(normalinds)
    abnormal = np.array(abnormalinds)

    index = np.concatenate((normal, abnormal))

    print("Train/Test size: {}, normal size: {}, abnormal size: {}".format(len(index), len(normal), len(abnormal)))

    print("Total size: {}, generate size: {}".format(len(y), len(index)))

    index_path = os.path.join(data, data + INDEX)

    np.savetxt(index_path, index, fmt='%d')
