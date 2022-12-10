#------loading QM9 dataset------#
import torch
from torch import nn
import warnings
import numpy as np
from HW4_p1 import *
from matplotlib import pyplot as plt
from rdkit import Chem
#--------------------------------#

# QM9 dataset loading from 
# https://github.com/nshervt/ACMS-80770-Deep-Learning-with-Graphs/blob/trunk/Homework%203/Problem%201.py
#------------------------------------------------------------------------------------------------------#
warnings.simplefilter(action='ignore', category=UserWarning)
from chainer_chemistry import datasets
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor

"""
    load data
"""
num_train = 5000
num_test  = 1000
dataset, _ = datasets.get_qm9(GGNNPreprocessor(kekulize=True), return_smiles=True,
                                           target_index=np.random.choice(range(133000), num_train+num_test, False))
# testing
Tdataset = dataset[num_train:]

# training
dataset  = dataset[:num_train]

V = 9
atom_types = [6, 8, 7, 9, 1]

def adj(x):
    x = x[1]
    adjacency = np.zeros((V, V)).astype(float)
    adjacency[:len(x[0]), :len(x[0])] = x[0] + 2 * x[1] + 3 * x[2]
    return torch.tensor(adjacency)


def sig(x):
    x = x[0]
    atoms = np.ones((V)).astype(float)
    atoms[:len(x)] = x
    out = np.array([int(atom == atom_type) for atom_type in atom_types for atom in atoms]).astype(float)
    return torch.tensor(out).reshape(5, len(atoms)).T


def target(x):
    x = x[2]
    return torch.tensor(x)

# code referred from: 
# https://github.com/nshervt/ACMS-80770-Deep-Learning-with-Graphs/blob/trunk/Homework%204/Problem%202.py
def Hbond(x):
    mol = Chem.MolFromSmiles(x)
    return torch.tensor(Chem.rdMolDescriptors.CalcNumHBA(mol))


adjs   = torch.stack(list(map(adj, dataset)))
sigs   = torch.stack(list(map(sig, dataset)))
prop   = torch.stack(list(map(target, dataset)))[:, 5]
#prop_2 = torch.stack(list(map(Hbond, dataset_smiles)))
#------------------------------------------------------------------------------------------------------#


# eta function as the average graph pooling operator 
# input:
#      f: feature
def eta(f):
    return torch.mean(f, dim=1)#.unsqueeze(1) # mean over vertices

# normalized laplacian matrix from the adjacency 
# input:
#       A: adjacency matrix
# ouput:
#       X: batched eigen vector
#       LMD: batched eigen values
def Normalized_Laplacian(A):

    #----------------------------------------------#
    # add self component
    dim = A.shape[1]
    I   = torch.eye(dim).reshape((1,dim,dim)) # prepare identity matrix for self-loop
    I   = I.repeat(A.shape[0],1,1)            # create batched identity matrix

    A_tilde = A + I                           # add diagonals to adj matrices

    # find degree matrix (batched version via dia_embed)
    d = torch.sum(A_tilde, dim=1) 
    D = torch.diag_embed(d)

    D_inv_sqrt = torch.sqrt(torch.linalg.inv(D))

    # calculate laplacian matrix
    Laplacian  = D-A_tilde

    # calculate normalized laplacian matrix
    L_sym      = D_inv_sqrt @ Laplacian @ D_inv_sqrt
    
    # batched eigen decomposition for symm mat
    LMD, X = torch.linalg.eigh(L_sym)

    return LMD + 1e-15, X # machine precision 

# apply fourier transform to the singnal
# inputs:
#       X: batched eigen vector
#       LMD: batched eigen values
#       f: signal
#       j: scale label
# outputs:
#       f_hat: transformed signal

def Fourier_transform(LMD,X,f,j):
    # init
    gj = torch.zeros(LMD.shape[0],LMD.shape[1]).double()

    # batch loop
    for b in range(LMD.shape[0]):
        # eig loop
        for i in range(LMD.shape[1]):
            gj[b,i] = spec_filter(j, LMD[b,i])

    # make it as a diagonal mat
    Gj = torch.diag_embed(gj)

    t1 = torch.bmm(X, Gj)

    t2 = torch.bmm(t1, torch.transpose(X, 1, 2))

    return torch.bmm(t2, f)

# generate zG
# Inputs:
#       L: depth
#       J: number of scales
#       f: original signal
#       LMD: eigenvalues
#       X: eigenvectors
# Output:
#       zG: generated feature

def zG_generation(L,J, f, LMD, X):

    zG = eta(f) # init as z^0

    for l in range(1,L+1):

        # loop over j
        for j in range(1,J+1):

            result = torch.abs(Fourier_transform(LMD,X,f,j))

            if l == 1: 
                zG = torch.cat((zG, eta(result)), dim=1)
            if l == 2:
                for j2 in range(1,J+1):
                    result2 = torch.abs(Fourier_transform(LMD,X,result,j2))
                    zG = torch.cat((zG, eta(result2)), dim=1)
    return zG


# define the 2-layer model
class MyModel(nn.Module):
    """
        model
    """
    def __init__(self, in_feature, out_feature):
        super(MyModel, self).__init__()

        self.nn  = 10

        self.fc1 = nn.Linear(in_feature, self.nn)
        self.fc2 = nn.Linear(self.nn, out_feature)
        self.act = nn.ReLU()

    def forward(self, x):
        return  self.fc2(self.act( self.fc1(x.float()) ))


# define the model
model = MyModel(2,1)
lr    = 0.01 # learning

MyLoss = nn.MSELoss() # it is a regression problem
MyOptimizer = torch.optim.Adam(model.parameters(), lr = lr) 

# start to construct zG
L = 2
J = 8

# prepare scattering features, dimension reduced via PCA
LMD, X = Normalized_Laplacian(adjs)
zG = zG_generation(L, J, sigs, LMD, X)
# start to do pca, use the default function
(_,_,V_train) = torch.pca_lowrank(zG) # pca decomposition
twoDzG  = torch.matmul(zG, V_train[:, :2])  # feature reduction

# start to train the model
Loss_epochs = []
mini_B = 50
B_size = int(num_train/mini_B)

for epoch in range(500):

    loss_per_epoch = 0

    for i in range(mini_B):

        pred = model(twoDzG[i*B_size:(i+1)*B_size])  

        # indexing the ground truth
        truth = (prop[i*B_size:(i+1)*B_size]).reshape((-1,1))

        loss = MyLoss(pred, truth) # compute batched mse loss
        
        loss_per_epoch += loss
        #   Zero-out the gradient
        MyOptimizer.zero_grad()
        
        # back-prop
        loss.backward()

        # gradient update
        MyOptimizer.step()

    Loss_epochs.append(loss.item()/mini_B)

# -- plot loss
plt.figure(figsize=(10, 8))
plt.semilogy(Loss_epochs,'b', linewidth=2)
plt.xlabel('Epoch',fontsize=16)
plt.ylabel('log MSE loss',fontsize=16)
plt.tick_params(labelsize=16)
fig_name = 'P2_C.png'
plt.savefig(fig_name)
#------------------------------Testing----------------------------------------------#

# prepare testing variables
Tadjs   = torch.stack(list(map(adj, Tdataset)))
Tsigs   = torch.stack(list(map(sig, Tdataset)))
Tprop   = torch.stack(list(map(target, Tdataset)))[:, 5]

# prepare testing scattering representations
T_LMD, T_X = Normalized_Laplacian(Tadjs)
T_zG = zG_generation(L, J, Tsigs, T_LMD, T_X)
(_,_,T_V) = torch.pca_lowrank(T_zG) # pca decomposition
T_twoDzG  = torch.matmul(T_zG, T_V[:, :2])  # feature reduction

model.eval()  # eval mode

pred_save = []
truth_save = []

with torch.no_grad():

    for i in range(num_test):

        pred = model(T_twoDzG[i])  # batch size of 1

        truth = (Tprop[i]).reshape((-1,1))

        pred_save.append(pred[0].item())
        truth_save.append(truth[0].item())


# -- plot scatter plot
y = lambda x: x

fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111)
plt.scatter(np.array(pred_save), np.array(truth_save), marker='*', c='r')
plt.plot(np.array(pred_save), y(np.array(pred_save)), 'k--')
plt.xlabel('Predicted',fontsize=16)
plt.ylabel('Truth',fontsize=16)
plt.tick_params(labelsize=16)
#ax.set_aspect('equal', adjustable='box')
fig_name = 'P2_D.png'
plt.savefig(fig_name)
