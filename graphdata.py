import torch
import copy
import dgl.function as fn

class Graph:
    def __init__(self, graph, features, labels, edge_index, infmatrix, lap, hop):
        self.graph = graph
        self.features = features
        self.labels = labels
        self.edge_index = edge_index
        self.infmatrix = infmatrix
        self.lap = lap
        self.hop = hop

        self.embeddings = []
        self.distances = []
        D_invsqrt = torch.pow(self.graph.in_degrees().float().clamp(min=1), -0.5).unsqueeze(-1)
        self.xlx = torch.sigmoid(torch.diag(torch.spmm(torch.spmm(self.features.T, self.lap), self.features)))

        tempgraph = copy.deepcopy(self.graph)
        tempinfmatrix = torch.spmm(self.infmatrix, self.features)
        tempembedding = self.features
        
        self.distances.append(torch.abs(tempembedding - tempinfmatrix))
        self.embeddings.append(tempembedding)

        with tempgraph.local_scope():
            for i in range(1, hop):
                tempgraph.ndata['h'] = tempembedding * D_invsqrt
                tempgraph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                tempembedding = tempgraph.ndata.pop('h') * D_invsqrt
                self.distances.append(torch.abs(tempembedding - tempinfmatrix))
                self.embeddings.append(tempembedding)
        
