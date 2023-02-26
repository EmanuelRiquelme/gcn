import torch
import os
import numpy as np
from dataset import Cites

def reshape_format(data,idx,bs):
    adj_matrix,features,target  = data[idx]
    lenght =  idx.shape[0]//bs
    features = features.reshape(lenght,bs,-1) 
    adj_matrix = adj_matrix.reshape(lenght,bs,-1) 
    target = target.reshape(lenght,bs,-1)
    return adj_matrix,features,target 

def get_BS(data,BS = 500,size_train = 1000,size_val = 500):
    train_idx = np.random.choice(len(data), size=size_train, replace=False)
    p_val_bs = np.ones(len(data))
    p_val_bs[train_idx] = 0
    p_val_bs = p_val_bs*1/(len(data)-size_train)
    val_idx = np.random.choice(len(data), size=size_val, replace=False, p = p_val_bs)
    return reshape_format(data,train_idx,BS),reshape_format(data,val_idx,BS)

def validation(val_data,model,device = 'cuda'):
    acc = []
    BS = val_data[0].size(0)
    adj_matrix,features,target = val_data
    adj_matrix,features,target = adj_matrix.to(device),features.to(device),target.to(device)
    with torch.no_grad():
        for idx in range(BS):
            pred = model(adj_matrix[idx],features[idx])
            pred = torch.argmax(pred,-1)
            target_bs = target[idx].squeeze(1)
            acc.append(((pred == target_bs).nonzero()).size(0)/target_bs.size(0))
        return sum(acc)/len(acc)
def load_model(model):
    model.load_state_dict(torch.load(f'{os.getcwd()}/gcn.pt'))

def save_model(model):
    torch.save(model.state_dict(), f'{os.getcwd()}/gcn.pt')
