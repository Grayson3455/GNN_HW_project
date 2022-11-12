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

#------------------------------------------------------------------------
# set tensor type to double
torch.set_default_tensor_type(torch.DoubleTensor)
from matplotlib import pyplot as plt
#------------------------------------------------------------------------

"""
    load data
"""
#--------------------------------------------------------------------------------------------#
dataset, dataset_smiles = datasets.get_qm9(GGNNPreprocessor(kekulize=True), return_smiles=True,
                                           target_index=np.random.choice(range(133000), 5000 , False))
#--------------------------------------------------------------------------------------------#


# Gather testing dataset
#-----------------------------------------------------------------------------------------------#
Tdataset, Tdataset_smiles = datasets.get_qm9(GGNNPreprocessor(kekulize=True), return_smiles=True,
                                           target_index=np.random.choice(range(133000), 1000 , False))
#-----------------------------------------------------------------------------------------------#


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


#--------------Gather testing data---------------------#
# batched adjacency matrices
Tadjs = torch.stack(list(map(adj, Tdataset)))

# batched node embeddings as one-hot vec, the last one is alawys H
Tsigs = torch.stack(list(map(sig, Tdataset)))


Tprop = torch.stack(list(map(target, Tdataset)))[:, 5]
#------------------------------------------------------#



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
        dim = A.shape[1]
        I   = torch.eye(dim).reshape((1,dim,dim)) # prepare identity matrix for self-loop
        I   = I.repeat(A.shape[0],1,1)            # create batched identity matrix

        A_tilde = A + I                           # add diagonals to adj matrices

        # find degree matrix (batched version via dia_embed)
        d = torch.sum(A_tilde, dim=1) 
        D = torch.diag_embed(d)

        D_inv_sqrt = torch.sqrt(torch.linalg.inv(D))

        pass

        return self.act(  D_inv_sqrt @  A_tilde @  D_inv_sqrt @ H @ self.W + self.b )
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
        return torch.sum(H, dim = 1) # dim 1 is the node idx dimension
        #-------------------------------#

class MyModel(nn.Module):
    """
        Regression  model
    """
    #----------------------------------------------#
    def __init__(self, in_features, out_features, in_linear, out_linear ):
    #----------------------------------------------#
        super(MyModel, self).__init__()
        # -- initialize layers

        #------------------------------------------#
        self.GCN = GCN(in_features, out_features)
        self.GraphPooling = GraphPooling()
        self.fc = nn.Linear(in_linear, out_linear)

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

GCN_in_feature = 5  # length of one-hot vec
GCN_out_feature = 3 # number of GCN output features 
Linear_in_feature = GCN_out_feature  # in-feature of the final linear layer
Linear_out_feature = 1        # energy 

model = MyModel(GCN_in_feature, GCN_out_feature, Linear_in_feature, Linear_out_feature)
lr    = 0.001 # learning

MyLoss = nn.MSELoss() # it is a regression problem
MyOptimizer = torch.optim.SGD(model.parameters(), lr = lr) # SGD optimizer

#--------------------------------------------#


# -- update parameters
Loss_epochs = []
for epoch in range(200):
    for i in range(100):

        # -- predict
        pred = model(adjs[i*50:(i+1)*50], sigs[i*50:(i+1)*50])  # batch size of 10

        #--------------------------------------------------------------------------
        # -- loss

        # indexing the ground truth
        truth = (prop[i*50:(i+1)*50]).reshape((-1,1)).double()

        loss = MyLoss(pred, truth) # compute batched mse loss
        # -- optimize

        # Zero-out the gradient
        MyOptimizer.zero_grad()
        
        # back-prop
        loss.backward()

        # gradient update
        MyOptimizer.step()

    Loss_epochs.append(loss.item())
        #--------------------------------------------------------------------------

# -- plot loss
plt.figure(figsize=(10, 8))
plt.semilogy(Loss_epochs,'b', linewidth=2)
plt.xlabel('Epoch',fontsize=16)
plt.ylabel('log MSE loss',fontsize=16)
plt.tick_params(labelsize=16)
fig_name = 'P1-QE.png'
plt.savefig(fig_name)
plt.show()


#------------------------------Testing----------------------------------------------#
model.eval()  # eval mode

pred_save = []
truth_save = []
with torch.no_grad():

    for i in range(1000):

        Tadj = Tadjs[i].unsqueeze(0) # add batch dim
        Tsig = Tsigs[i].unsqueeze(0) # add batch dim

        pred = model(Tadj, Tsig)  # batch size of 1

        truth = (Tprop[i]).reshape((-1,1)).double()

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
fig_name = 'P1-QF.png'
plt.savefig(fig_name)
plt.show()

