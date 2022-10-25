import torch
import torch.nn as nn
import numpy as np


class MsgPassingNN(nn.Module):
	def __init__(self, device, input_size_Edge, output_size_node, msg_passing):
		
		super().__init__()               # allow inherience from nn.Module
		
		self.device = device 
		self.input_edge     = input_size_Edge       # number of input neurons
		self.output_node    = output_size_node      # number of output neurons
		self.msg            = msg_passing           # number of massage passing steps

		
	
	def forward(self,x):
		return 0