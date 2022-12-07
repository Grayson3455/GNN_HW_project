#------loading QM9 dataset------#
import torch
from torch import nn
import warnings
import numpy as np
from HW4_p1 import *
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
dataset, dataset_smiles = datasets.get_qm9(GGNNPreprocessor(kekulize=True), return_smiles=True,
                                           target_index=np.random.choice(range(133000), 100, False))

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


adjs = torch.stack(list(map(adj, dataset)))
sigs = torch.stack(list(map(sig, dataset)))
prop = torch.stack(list(map(target, dataset)))[:, 5]
#------------------------------------------------------------------------------------------------------#


# eta function as the average graph pooling operator 
# input:
#      f: feature
def eta(f):
    return 0

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
    
    print(L_sym.shape)
    # batched eigen decomposition for symm mat
    LMD, X = torch.linalg.eigh(L_sym)

    return LMD, X

# apply fourier transform to the singnal
# inputs:
#       X: batched eigen vector
#       LMD: batched eigen values
#       f: signal
#       j: scale label
# outputs:
#       f_hat: transformed signal

def Fourier_transform(LMD,X,f,j):
    
    gj = torch.zeros(len(LMD))

    for i in range(1,len(LMD)+1):
        gj[i-1] = spec_filter(i, LMD[i-1])

    return X @ gj @ torch.transpose(X, 1, 2) @ f


LMD, X = Normalized_Laplacian(adjs[35])
print(LMD.shape)
f_hat  = Fourier_transform(LMD, X, sigs[35], 2)

# start to construct zG
L = 2
J = 8

LMD, X = Normalized_Laplacian(adjs)


