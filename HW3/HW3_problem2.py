"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame
Homework 2: Programming assignment
Problem 2
"""
import torch

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch import nn
from torch.autograd.functional import jacobian


torch.manual_seed(0)


class GCN(nn.Module):
    """
        Graph convolutional layer
    """
    def __init__(self, in_features, out_features):
        super(GCN, self).__init__()

        # -- initialize weight
        
        #---------------------------------------------------------------------------#
        self.in_features  = in_features
        self.out_features = out_features 
        self.W      = nn.Parameter(data=torch.randn(size=(self.in_features,self.out_features)), requires_grad=True)
        self.b      = nn.Parameter(data=torch.randn(size=(self.out_features,)), requires_grad=True)
        #---------------------------------------------------------------------------#

        pass

        # -- non-linearity
        #--------------------------------------------------------------------#
        self.act = nn.ReLU()
        #--------------------------------------------------------------------#

    def __call__(self, A, H):
        # -- GCN propagation rule

        #----------------------------------------------#
        dim = A.shape[1]
        I   = torch.eye(dim)                      # prepare identity matrix for self-loop

        A_tilde = A + I                           # add diagonals to adj matrices

        # find degree matrix (batched version via dia_embed)
        d = torch.sum(A_tilde, dim=1) 
        D = torch.diag_embed(d)

        D_inv_sqrt = torch.sqrt(torch.linalg.inv(D))

        pass

        return self.act(  D_inv_sqrt @  A_tilde @  D_inv_sqrt @ H @ self.W + self.b )
        #----------------------------------------------#

# define the first, single msg passing model
class MyModel(nn.Module):
    """
        model
    """
    def __init__(self, A, in_feature, out_feature):
        super(MyModel, self).__init__()
        # -- initialize layers
        pass

        self.A = A

        self.GCN = GCN(in_feature, out_feature)


    # tag == None: get all features
    # tag ~= None, get selected features
    def forward(self, h0, tag):
        pass

        if tag ==  None:
            return self.GCN(self.A, h0)
        else:
            return self.GCN(self.A, h0)[tag,:]

# define the second model, multiple msg passing
class MyModel_2(nn.Module):
    """
        model
    """
    def __init__(self, A):
        super(MyModel_2, self).__init__()
        # -- initialize layers
        pass

        self.A = A


        self.GCN1 = GCN(200,100) 
        self.GCN2 = GCN(100,50)
        self.GCN3 = GCN(50,20)


    # tag == None: get all features
    # tag ~= None, get selected features
    def forward(self, h0, tag):
        pass

        if tag ==  None:
            return self.GCN3(A,self.GCN2(A, self.GCN1(self.A, h0)))
        else:
            return self.GCN3(A,self.GCN2(A,self.GCN1(self.A, h0)))[tag,:]




# define the third model, multiple msg passing
class MyModel_3(nn.Module):
    """
        model
    """
    def __init__(self, A):
        super(MyModel_3, self).__init__()
        # -- initialize layers
        pass

        self.A = A


        self.GCN1 = GCN(200,100) 
        self.GCN2 = GCN(100,50)
        self.GCN3 = GCN(50,20)
        self.GCN4 = GCN(20,20)
        self.GCN5 = GCN(20,20)


    # tag == None: get all features
    # tag ~= None, get selected features
    def forward(self, h0, tag):
        pass

        if tag ==  None:
            return self.GCN5(A, self.GCN4(A, self.GCN3(A,self.GCN2(A, self.GCN1(self.A, h0)))))
        else:
            return self.GCN5(A, self.GCN4(A,self.GCN3(A,self.GCN2(A,self.GCN1(self.A, h0)))))[tag,:]




"""
    Effective range
"""
# -- Initialize graph
seed = 32
n_V = 200   # total number of nodes
G = nx.barabasi_albert_graph(n_V, 2, seed=seed)

# -- plot neighborhood
for i in [17,27]:
    for k in [2,4,6]:
        fig = plt.figure(figsize=(6, 6))
        # -- plot graph
        layout = nx.spring_layout(G, seed=seed, iterations=400)
        nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=100)


        nodes = nx.single_source_shortest_path_length(G, i, cutoff=k)
        im2 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color='red', node_size=100)

        # -- visualize
        
        #plt.colorbar(im2,fig.add_axes([0.92, 0.5, 0.03, 0.38]))
        fig_name = 'P2-QB-k='+str(k)+ '-i=' + str(i) +'.png'
        plt.savefig(fig_name)


#-------------------------------------------------#
H = torch.eye(n_V)
A = torch.from_numpy(nx.adjacency_matrix(G).todense()) # adjacency matrix
#-------------------------------------------------#



# define influence score function
# InPUTS:
#     V: NUMBER OF NODES
#     J: Jacobian matrix 
#     outfeature: outfeature of GNN
def IK( N_v, J, outfeature):

    # initialize inflence score vector
    IK    = torch.zeros(n_V)

    for j in range(N_v):

        J_ij = J[:,j,:]

        IK[j] = torch.transpose( torch.ones(outfeature,1) , 0, 1 ) @ J_ij @ torch.ones(N_v,1)

    return IK

# define picture save
def IK_save(IK, model_idx, tag):

    G = nx.barabasi_albert_graph(n_V, 2, seed=seed)
    nodes = G.nodes()
    layout = nx.spring_layout(G, seed=seed, iterations=400)
    fig = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color=IK, node_size=100)
    plt.colorbar(fig, shrink = 0.8)
    fig_name = 'P2-QD-model='+str(model_idx)+ '-i=' + str(tag) +'.png'
    plt.savefig(fig_name)
    plt.close()
    return 0


#-----------------MODEL-1-------------------------#
model_idx  =  1
outfeature = 100
model = MyModel(A, H.shape[0], outfeature)

# node 17
tag        = 17
H_K   = lambda H:  model(H, tag) # forward the model, but defined as lambda function
J = jacobian(H_K,H)
ik_model = IK(n_V, J, outfeature)
IK_save(ik_model, model_idx, tag)

# node 27
tag        = 27
H_K   = lambda H:  model(H, tag) # forward the model, but defined as lambda function
J = jacobian(H_K,H)
ik_model = IK(n_V, J, outfeature)
IK_save(ik_model, model_idx, tag)
#-------------------------------------------------#



#-----------------MODEL-2-------------------------#
model_idx  =  2
outfeature = 20
model = MyModel_2(A)

# node 17
tag        = 17
H_K   = lambda H:  model(H, tag) # forward the model, but defined as lambda function

J = jacobian(H_K,H)
ik_model = IK(n_V, J, outfeature)
IK_save(ik_model, model_idx, tag)

# node 27
tag        = 27
H_K   = lambda H:  model(H, tag) # forward the model, but defined as lambda function

J = jacobian(H_K,H)
ik_model = IK(n_V, J, outfeature)
IK_save(ik_model, model_idx, tag)
#-------------------------------------------------#



#-----------------MODEL-3-------------------------#
model_idx  =  3
outfeature = 20
model = MyModel_3(A)

# node 17
tag        = 17
H_K   = lambda H:  model(H, tag) # forward the model, but defined as lambda function

J = jacobian(H_K,H)
ik_model = IK(n_V, J, outfeature)
IK_save(ik_model, model_idx, tag)

# node 27
tag        = 27
H_K   = lambda H:  model(H, tag) # forward the model, but defined as lambda function

J = jacobian(H_K,H)
ik_model = IK(n_V, J, outfeature)
IK_save(ik_model, model_idx, tag)
