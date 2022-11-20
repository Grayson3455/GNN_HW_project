import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

# max-min scaler in the training stage
def max_min_scaling(X):
    scale_max = X.max() # locate max 
    scale_min = X.min() # locate min

    # scale to [-1,1]
    X_scaled = 2.*(X-scale_min)/(scale_max-scale_min) - 1
    return scale_min, scale_max, X_scaled


# auto-batching tools for regression 
class MyDatasetXY(Dataset):
    def __init__(self, X, Y):
        super(MyDatasetXY, self).__init__()
        
        # sample size checker, the first dimension is always the batch size
        assert X.shape[0] == Y.shape[0]
        
        self.X = X
        self.Y = Y

    # number of samples to be batched
    def __len__(self):
        return self.X.shape[0] 
       
    # get samples
    def __getitem__(self, index):
        return self.X[index], self.Y[index]


# nonlinear mlp functions, can be generalized to various purposes
        # NI: input size
        # NO: ouput size
        # NN: hidden size
        # NL: num of hidden layers
        # act: type of nonlinear activations, default: relu

def MLP_nonlinear(NI,NO,NN,NL,act='relu'):

    # select act functions
    if act == "relu":
        actF = nn.ReLU()
    elif act == "tanh":
        actF = nn.Tanh()
    elif act == "sigmoid":
        actF = nn.Sigmoid()
    elif act == 'leaky':
        actF = nn.LeakyReLU()

    #----------------construct layers----------------#
    MLP_layer = []

    # Input layer
    MLP_layer.append( nn.Linear(NI, NN) )
    MLP_layer.append(actF)
    
    # Hidden layer
    for ly in range(NL-2):
        MLP_layer.append(nn.Linear(NN, NN))
        MLP_layer.append(actF)
   
    # Output layer
    MLP_layer.append(nn.Linear(NN, NO))
    
    # seq
    return nn.Sequential(*MLP_layer)



def simple_loss_plot(PATH, train_save, train_acc_save):

    fs = 24
    plt.rc('font',  family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.rc('text',  usetex=True)

    fig1 = plt.figure(figsize=(10,8))
    plt.subplot(1,2,1)
    plt.semilogy(train_save, '-b', linewidth=2, label = 'training loss')
    plt.xlabel(r'$\textrm{Epoch}$',fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.legend(loc='upper right',fontsize=fs-3)

    plt.subplot(1,2,2)
    plt.plot(train_acc_save, '-b', linewidth=2, label = 'training accuracy')
    plt.xlabel(r'$\textrm{Epoch}$',fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.legend(loc='upper right',fontsize=fs-3)
    plt.tight_layout()
    
    fig_name = PATH + '/train-test-loss-acc.png'
    plt.savefig(fig_name)
        
    train_loss_name  = PATH + '/train_loss.csv'
    train_acc_name  = PATH + '/train_acc.csv'


    np.savetxt(train_loss_name, train_save,   delimiter = ',')
    np.savetxt(train_acc_name, train_acc_save,   delimiter = ',')


    return 0
