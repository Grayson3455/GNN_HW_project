import torch
import torch.nn as nn
import numpy as np
from GNN_tools.DNN_tools import *

class MsgPassingNN(nn.Module):
	def __init__(self, device, input_size_Edge, output_size_node, msg_passing):
		
		super().__init__()               # allow inherience from nn.Module
		
		self.device = device 
		self.input_edge     = input_size_Edge       # number of input neurons
		self.output_node    = output_size_node      # number of output neurons
		self.msg            = msg_passing           # number of massage passing steps


		#-----------------Define edge mlp parameters-----------------#
		self.output_edge    = 9   # output of the edge function
		self.hidden_edge    = 9   # hidden units of the edge function
		self.layer_edge     = 3   # hidden layers of the edge function

		# define edge function as normal mlp
		self.fe             = MLP_nonlinear(self.input_edge,self.output_edge, \
											self.hidden_edge,self.layer_edge,act='relu')
		
		#-----------------Define node mlp parameters-----------------#
		self.input_node     = self.output_edge + self.output_node      # number of inputs of the node function
		self.hidden_node    = 9   # hidden units of the node function
		self.layer_node     = 3   # hidden layers of the node function
		
		# define node function as normal mlp
		self.fx             = MLP_nonlinear(self.input_node,self.output_node, \
											self.hidden_node,self.layer_node,act='relu')
		
	# the main forward function
	# input:
	#           X: feature mat |V| x 3
	#           AN: neighbor information dic
	#           l: number of nodes
	# output:   Y_hat: updated node feature

	def forward(self,X,AN,l):

		for k in range(self.msg): # number of message passing rounds

			X_new = torch.zeros_like(X)

			# loop tho nodes
			for i in range(l):

				neighbors = AN[str(i)]

				# ini agged msg
				m_i = 0

				# loop tho neighbors
				
				for j in neighbors:

					m_i += self.fe(torch.cat((X[i,:],X[j,:])))

				X_new[i,:] = self.fx(torch.cat((X[i,:],m_i)))

			# replace node features
			X = X_new

		return X


# training alg
# inputs:
#		device: cpu or gpu
#       model : dnn model
#       criterion: loss function
#       optimizer: gradient alg
#       x        : LOW fid solution
#       y        : high fid solution
#       cells    : cell labels
#       points   : coordinates
#       AN       : neighbor info dic
# outputs:
#       model:    updated model
#       train_loss: training loss of the current epoch
#       train_acc : training accuracy of the current epoch

def GNN_train(device, model, criterion, optimizer, X, Y, Cells, Points, AN):
	# init losses
	train_loss, train_acc  = 0.0, 0.0
	

	# train status
	model.train()

	# number of nodes
	num_nodes = len(Points)

	# Zero-out the gradient
	optimizer.zero_grad()

	Y_hat = model(X,AN,num_nodes)

	# mse loss
	loss_mse = criterion(Y_hat, Y)

	Rel_acc  = 1.0 - loss_mse/criterion(Y, torch.zeros_like(Y, device=device))

	# record the loss and acc
	train_loss += loss_mse.item()
	train_acc  += Rel_acc.item()
	
	loss_mse.backward()

	# gradient update
	optimizer.step()
	
	return 	model, train_loss, train_acc 
