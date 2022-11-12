"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame
Homework 2: Programming assignment
Problem 1
"""
import torch
from torch import nn
import warnings
import numpy as np

warnings.simplefilter(action='ignore', category=UserWarning)
from chainer_chemistry import datasets
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor


"""
    load data
"""
#--------------------------------------------------------------------------------------------#
dataset, dataset_smiles = datasets.get_qm9(GGNNPreprocessor(kekulize=True), return_smiles=True,
                                           target_index=np.random.choice(range(133000), 500 , False))
# change it from 100 --> 500 to get enough training samples
#--------------------------------------------------------------------------------------------#

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

# batched adjacency matrices
adjs = torch.stack(list(map(adj, dataset)))

# batched node embeddings as one-hot vec, the last one is alawys H
sigs = torch.stack(list(map(sig, dataset)))


prop = torch.stack(list(map(target, dataset)))[:, 5]



class GCN:
    """
        Graph convolutional layer
    """
    def __init__(self, in_features, out_features):
        # -- initialize weight
        

        #---------------------------------------------------------------------------#
        self.in_features  = in_features
        self.out_features = out_features 
        self.W      = nn.Parameter(data=torch.randn(size=(self.in_features,self.out_features)), requires_grad=True)
        self.b      = nn.Parameter(data=torch.randn(size=(self.out_features,)), requires_grad=True)
        #---------------------------------------------------------------------------#
        
        pass


        # -- non-linearity
        #------------------#
        self.act = nn.ReLU()
        #------------------#

    def __call__(self, A, H):
        # -- GCN propagation rule

        #----------------------------------------------#
        

        # find degree matrix (batched version)
        D = torch.zeros_like(A)
        d = torch.sum(A, dim=1) 
        


        # define identity matrix
        I = torch.eye(len(D))
        pass

        #print(D.shape, I.shape, A.shape, H.shape, self.W.shape, self.b.shape)

        return self.act( D.pow(-0.5) @  (I + A) @ D.pow(-0.5) @ H @ self.W + self.b )
        #----------------------------------------------#


class GraphPooling:
    """
        Graph pooling layer
    """
    def __init__(self):
        pass

    def __call__(self, H):
        # -- multi-set pooling operator
        pass
        #-------------------------------#
        return torch.sum(H, dim = 0)
        #-------------------------------#

class MyModel(nn.Module):
    """
        Regression  model
    """
    #----------------------------------------------#
    def __init__(self, in_features, out_features ):
    #----------------------------------------------#
        super(MyModel, self).__init__()
        # -- initialize layers

        #------------------------------------------#
        self.GCN = GCN(in_features, out_features)
        self.GraphPooling = GraphPooling()
        self.fc = nn.Linear(out_features, out_features)

        #-------------------------------------------#
        pass

    def forward(self, A, h0):
        pass

        #-------------------------------------------#
        return self.fc( self.GraphPooling( self.GCN(A,h0) ) )
        #--------------------------------------------#
"""
    Train
"""
# -- Initialize the model, loss function, and the optimizer
#--------------------------------------------#

in_feature = 5  # length of one-hot vec
out_feature = 1 # HOMO energy  

model = MyModel(in_feature,out_feature)
lr    = 0.01 # learning

MyLoss = nn.MSELoss() # it is a regression problem
MyOptimizer = torch.optim.SGD(model.parameters(), lr = lr) # SGD optimizer

#--------------------------------------------#


# -- update parameters
for epoch in range(200):
    for i in range(10):

        # -- predict
        pred = model(adjs[i*10:(i+1)*10], sigs[i*10:(i+1)*10])

        # -- loss
        # loss = ?

        # -- optimize

# -- plot loss