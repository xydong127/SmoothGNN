import torch
import torch.nn as nn
import dgl.function as fn


class NAD(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, graphdata, init):
        super(NAD, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.graphdata = graphdata

        self.featuretrans1 = []
        self.featuretrans2 = []
        self.convs = []
        self.coefs = []
        for i in range(self.graphdata.hop):
            self.featuretrans1.append(nn.Linear(in_dim, hidden_dim))
            self.featuretrans2.append(nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(nn.Linear(hidden_dim, hidden_dim))

    
        self.coef = nn.Linear(in_dim, hidden_dim)

        self.linear1 = nn.Linear(hidden_dim * len(self.convs), hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, in_dim)
        
        self.act = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.init_weight(init)

    def init_weight(self, init):
       
        m1 = 0
        s1 = init
        
        for i in range(self.graphdata.hop):
            nn.init.normal_(self.featuretrans1[i].weight, m1, s1)
            nn.init.normal_(self.featuretrans2[i].weight, m1, s1)
            nn.init.normal_(self.convs[i].weight, m1, s1)

        nn.init.normal_(self.coef.weight, m1, s1)
        nn.init.normal_(self.linear1.weight, m1, s1)
        nn.init.normal_(self.linear2.weight, m1, s1)

    def forward(self):
        h1_final = torch.zeros([len(self.graphdata.features), 0])
        h2_final = torch.zeros([len(self.graphdata.features), 0])
        for i, conv in enumerate(self.convs):
            h1 = self.featuretrans1[i](self.graphdata.embeddings[i])
            h1 = self.act(h1)
            h1 = self.featuretrans2[i](h1) 
            h1 = self.act(h1)
            h1 = conv(h1) 
            h1 = self.act(h1)
            h1_final = torch.cat([h1_final, h1], -1)
        
            h2 = self.featuretrans1[i](self.graphdata.distances[i])
            h2 = self.act(h2)
            h2 = self.featuretrans2[i](h2) 
            h2 = self.act(h2)
            h2 = conv(h2) 
            h2 = self.act(h2)
            h2_final = torch.cat([h2_final, h2], -1)
            

        h1 = self.linear1(h1_final)
        h1 = self.act(h1)
        reconembed = torch.pow(self.linear2(h1 * self.coef(self.graphdata.xlx)) - self.graphdata.features, 1)

        h2 = self.linear1(h2_final)
        h2 = self.act(h2)
        h2 = self.bn2(h2)
        anomalyembed = torch.mean(torch.sigmoid(torch.mean(h2 * self.coef(self.graphdata.xlx), dim=1, keepdim=True)), dim=1)
        return reconembed, anomalyembed
