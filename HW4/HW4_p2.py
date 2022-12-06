#------loading QM9 dataset------#
import torch
from torch import nn
import warnings
import numpy as np
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

print(sigs[5,:,:])
print(prop.shape)