import torch
from tqdm import trange
from model import GNN
from dataset import Cites
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import validation,save_model,get_BS

data = Cites('cora')
train_data, val_data = get_BS(data)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = GNN().to(device)
opt = optim.Adam(model.parameters(),lr=.01,weight_decay = 5e-4)
loss_fn = nn.CrossEntropyLoss()
epochs = 5000*10

def train(data = data,model = model,opt = opt,loss_fn = loss_fn,epochs = epochs,device = device):
    for epoch in (t := trange(epochs)):
        train_data, val_data = get_BS(data)
        BS = val_data[0].size(0)
        adj_matrix,features,target = train_data 
        adj_matrix,features,target = adj_matrix.to(device),features.to(device),target.to(device)
        for idx in range(BS):
            opt.zero_grad()
            output = model(adj_matrix[idx],features[idx])
            loss = loss_fn(output,target[idx].squeeze(1))
            loss.backward()
            opt.step()
        model.eval()
        t.set_description("validation: %.2f" % (validation2(val_data,model,device)))
        model.train()

if __name__ == '__main__':
    print(validation2(val_data,model,device))
    train()
    save_model(model)
