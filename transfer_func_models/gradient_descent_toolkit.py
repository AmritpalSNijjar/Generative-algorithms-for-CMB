import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

def load_camb_data(camb_data_fname):

    data = np.loadtxt(camb_data_fname)

    return data

def get_data_sets(filename, percent_train):
    data = torch.from_numpy(load_camb_data(filename)) # "camb_transfer_data.txt"
    
    rng = np.random.default_rng()
    data = rng.permutation(data, axis = 0)
    
    len_dat = data.shape[0]
    
    upto = int(percent_train*len_dat)
    
    train_data = torch.from_numpy(data[:upto, :])
    val_data   = torch.from_numpy(data[upto:, :])
    
    
    train_dataset = cambDataSet(train_data)
    val_dataset = cambDataSet(val_data)
    
    return train_dataset, val_dataset

class cambDataSet(Dataset):
    
    def __init__(self, data_ndarray):
        self.data = data_ndarray #torch.from_numpy(load_camb_data("camb_transfer_data.txt"))
        self.len = self.data.shape[0]
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.data[idx, :]

class transferFunctionModel(nn.Module):
    
    def __init__(self, n_params, init_pars_ranges):
        super(transferFunctionModel, self).__init__()
        
        
        # parameters
        self.n_params = n_params
        
        _params = torch.rand(self.n_params)
        _params = _params * (init_pars_ranges[1] - init_pars_ranges[0]) + init_pars_ranges[0]
        
        self.params = nn.Parameter(_params)
        
    def compute(self, omh2, obh2, ks):
        
        # ks is 1 dimensional NUMPY ARRAY of k values
        # ADD CHECKS TO SEE IF ks INPUT IS ARRAY OR TENSOR! or maybe that doesn't matter...
        
        
        len_ks = ks.shape[-1]
        
        data = torch.zeros((len_ks, 3))
        data[:, 0] = omh2
        data[:, 1] = obh2
        data[:, 2] = ks
        
        model.eval()
        
        with torch.no_grad():
            out = self.train_compute(data)
        
        model.train()
        
        return out
        
    def train_compute(self, data):
        # Equation Numbers refer to https://arxiv.org/abs/2407.16640
        
        data_shape = data.shape
        n_Tks = data_shape[0]
        
        num_a_genes = 19
        a_s = torch.narrow(self.params, 0, 0, num_a_genes)
        c_s = torch.narrow(self.params, 0, num_a_genes, self.params.size()[0] - num_a_genes)
        
        out = torch.zeros(n_Tks)
        
        # CAN I DO THIS WITHOUT ANY LOOP ? FOR PARALLELIZABILITY!!!
        # like ommh2 = data[:, 0] etc etc
        for i in range(n_Tks):
            
            # HOW TO ENSURE THAT NO COPIES ARE MADE IN THIS FOR LOOP ???
            # CAN I MAKE THESE VARIABLES VIEWS OF THE TENSOR!?
            
            ommh2 = data[i, 0]
            ombh2 = data[i, 1]
            k     = data[i, 2]
            
            # Eqs. (3) & (4)
            x       = k/(ommh2-ombh2)
            T_nw    = (1+59.0998*x**(1.49177) + 4658.01*x**(4.02755) + 3170.79*x**(6.06) + 150.089*x**(7.28478))**(-0.25)

            # Eqs. (17) - (19) w/ trainable a8 - a19
            f_alpha = a_s[7] - a_s[8]*(ombh2**a_s[9]) + a_s[10]*(ommh2**a_s[11])
            f_beta  = a_s[12] - a_s[13]*(ombh2**a_s[14]) + a_s[15]*(ommh2**a_s[16])
            f_node  = a_s[17]*(ommh2**a_s[18])

            # Eq. (11)
            s_GA    = (c_s[0]*(ombh2**c_s[1]) + c_s[2]*(ommh2**c_s[3]) + c_s[4]*(ombh2**c_s[5])*(ommh2**c_s[6]))**(-1)

            # Eq. (12)
            k_Silk  = 1.6*(ombh2**0.52)*(ommh2**0.73)*(1 + (10.4*ommh2)**(-0.95))

            # Eq. (13)
                                         # what to do if this term is negative ?
                
            f_amp   = f_alpha/(a_s[0] + (f_beta/(k*s_GA))**a_s[1])

            # Eq. (14)
            f_Silk  = (k/k_Silk)**a_s[2]

            # Eq. (15)
            f_osc   = a_s[3]*k*s_GA/(a_s[4] + f_node/(k*s_GA)**a_s[5])**a_s[6]

            # Eq. (5)
            T_w     = 1 + f_amp*torch.exp(-f_Silk)*torch.sin(f_osc)
            
            
            Tk      = T_nw * T_w
            
            out[i]  = Tk
            
            
        return out
        
    def forward(self, in_data):
        
        # CHECK: is in_data torch tensor? ... it must be!
        
        out = self.train_compute(in_data)
        
        return out
        
        

def loss_fn(train, pred):
    
    n = train.shape[0]
    
    val = torch.sum(torch.abs(train - pred)/train)
    val *= 100/n
    
    return val


def training_loop(n_epochs, model, train_loader, loss_fn, optimizer):
    
    for epoch in range(n_epochs):
        
        for train_data in train_loader: # ADD LOADER
            
            train_pred = model(train_data)
            loss_train = loss_fn(train_data[:, 3], train_pred)

            # val_pred = model(val_data)
            # loss_val = loss_fn(val_data[3, :], val_pred)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Training Loss: {round(loss_train.item(), 3)}")# Validation Loss: {round(loss_val.item(), 3)}")
    
        