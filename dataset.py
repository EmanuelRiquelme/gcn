import torch
import numpy as np
import scipy.sparse as sp
import os
from numpy.linalg import inv
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F

class Cites(Dataset):
    def __init__(self, root_dir,transform = False):
        self.root_dir = root_dir
        self.features = self.__get_features__()
        self.labels = self.__get_labels__()
        self.adj_matrix = self.__get_aj__()

    def __row_norm__(self,matrix):
        mean = np.mean(matrix,1)
        std = np.std(matrix,1)
        matrix = (np.transpose(matrix)-mean)/std
        return np.nan_to_num(np.transpose(matrix))

    def __get_labels__(self):
        labels = np.genfromtxt(f'{self.root_dir}/cora.content',dtype=np.dtype(str))[...,-1]
        index = np.unique(labels)
        numeric_labels = []
        for label in labels:
            numeric_labels.append(np.where(label==index))
        return np.array(numeric_labels).flatten()

    def __get_features__(self):
        features = np.genfromtxt(f'{self.root_dir}/cora.content',dtype=np.float32)[...,1:-1]
        return self.__row_norm__(features)

    def __get_aj__(self):
        papers_id = np.genfromtxt(f'{self.root_dir}/cora.content',dtype=np.float32)[...,0] 
        papers_id = np.unique(papers_id)
        adj = np.zeros([papers_id.shape[0],papers_id.shape[0]])
        edges = np.genfromtxt(f'{self.root_dir}/cora.cites',dtype=np.int32).reshape(2,-1)
        cited_papers,citing_papers = edges[0],edges[1]
        for cited_paper,citing_paper in zip(cited_papers,citing_papers):
            x_index = np.where(cited_paper == papers_id)[0][0]
            y_index = np.where(citing_paper == papers_id)[0][0]
            adj[x_index][y_index] = 1
        adj_tilda = adj+np.eye(adj.shape[0])
        d = np.eye(adj_tilda.shape[0])*np.sum(adj_tilda,1)
        d = np.linalg.inv(d)
        return d@adj_tilda@d

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.adj_matrix)[idx].float(),torch.tensor(self.features)[idx],torch.tensor(self.labels)[idx]
        #return self.features[idx]

if __name__ == '__main__':
    from model import GNN
    data = Cites(root_dir = 'cora')
    adj,features,labels = data[:2]
    print(features.size())
    model = GNN()
    print(model(adj,features))
