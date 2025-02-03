import argparse
import json
import time 
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import psutil
import os

import utils
import name
from name import *
from graphdata import Graph
import model

parser = argparse.ArgumentParser()
parser.add_argument('--data', default=AMAZON, help='Dataset used')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--nepoch', type=int, default=100, help='Num of epoches')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dim')
parser.add_argument('--hop', type=int, default=6, help='K hop')
parser.add_argument('--eps', type=float, default=4e-3, help='Epsilon')
parser.add_argument('--decay', type=float, default=1e-6, help='Weight decay')
parser.add_argument('--init', type=float, default=0.001, help='Weight init')
args = parser.parse_args()

print("Model info:")
print(json.dumps(args.__dict__, indent='\t'))

data = args.data
seed = args.seed
lr = args.lr
nepoch = args.nepoch
hidden_dim = args.hidden_dim
hop = args.hop
eps = args.eps
decay = args.decay
init = args.init

if data in DATASETS:
   lr, hop, init, seed, eps  = name.set_paras(data)    

utils.set_seed(seed)
graph, features, labels, edge_index, index = utils.load_data(data)
n = len(features)
m = edge_index.shape[1]
num_class = 2


print("Train/Test node num: {}, edge num: {}".format(n, m))

s = time.time()
print("Generating laplacian matrix...")
lap = utils.get_lap(edge_index, n)
e = time.time()
print("Generating laplacian matrix successfully, time cost: {}".format(e - s))

s = time.time()
print("Generating infinite matrix...")
infmatrix = utils.get_infmatrix(edge_index, n, m, eps)
e = time.time()
print("Generate infinite matrix successfully, time cost: {}".format(e - s))

s = time.time()
print("Generating graph data...")
graphdata = Graph(graph, features, labels, edge_index, infmatrix, lap, hop) 
e = time.time()
print("Generate graph data successfully, time cost: {}".format(e - s))

model = model.NAD(features.shape[1], hidden_dim, num_class, graphdata, init)
optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=decay)

bestepoch = 0
bestauc = 0
bestprec = 0

results = []

ts = time.time()
for epoch in range(nepoch):
    s = time.time()
    model.train()
    reconembed, anomalyembed = model()
    loss = torch.mean(reconembed[index]) +  torch.mean(anomalyembed[index])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    e = time.time()

    model.eval()
    probs = anomalyembed
    auc = roc_auc_score(labels[index], probs[index].detach().numpy())
    prec = average_precision_score(labels[index], probs[index].detach().numpy())

    if bestauc < auc:
        bestauc = auc
        bestprec = prec
        bestepoch = epoch

    print('Epoch {}, train loss: {}'.format(epoch, loss))
    e = time.time()
    print('Time cost: {}'.format(e - s))

    if epoch == nepoch - 1:
        print('Result for data: {}, AUC: {}, Precision: {}'.format(data, auc, prec))

te = time.time()

print("Total time cost: {}".format(te - ts))

