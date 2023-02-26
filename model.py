import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    def __init__(self,in_features = 1433,hidden_features = 16,
                num_class = 7,dropout_rate = .5):
        super().__init__()
        self.layer_1 = nn.Parameter(torch.FloatTensor(in_features,hidden_features))
        self.layer_2 = nn.Parameter(torch.FloatTensor(hidden_features,num_class))
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout_rate = dropout_rate

    def forward(self,adj,features):
        adj = torch.t(adj)
        x = torch.mm(adj,features)
        x = torch.spmm(x,self.layer_1)
        x = self.relu(x)
        x = F.dropout(x, self.dropout_rate)
        x = torch.mm(torch.t(adj),x)
        x = torch.spmm(x,self.layer_2)
        return x
