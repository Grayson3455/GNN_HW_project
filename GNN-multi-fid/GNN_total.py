# Low fid solution to high fid mesh via Graph neural network
import meshio
import torch
from Mesh_info.Mesh_Tools.Mesh_connectivity import *
import os
from GNN_tools.GNN import *

# determine if to use gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

#------------------------prepare dataset----------------------#
# low fid solution
Low_name  = 'Sol/annular/lv1-to-lv3000000.vtu'
Low_data  = meshio.read(Low_name)
Low_Dis   = (Low_data.point_data)['f_32'] # displacement solution

# low fid solution
High_name  = 'Sol/annular/Mesh_lv=3-p=1000000.vtu'
High_data  = meshio.read(High_name)
High_Dis   = (High_data.point_data)['f_72'] # displacement solution


#----------load mesh for accessing connectivity information----------#
mesh_name =  "Mesh_info/Annular/annular_lv3.msh"  
Cells, Points, Adj = Mesh_info_extraction2D(mesh_name)

#---------------------Get the maximum neighbor numbers-----------------#
# Note: we need to pad zeros 
# loop thorough nodes
# max_n = 0
# for i in range(len(Points)):

# 	# find neighbors of node i
# 	neighbors = np.where(Adj[i,:] == 1)[0]

# 	if len(neighbors) >= max_n:

# 		max_n = len(neighbors)

input_size_Edge  = 6 
output_size_node = 3

# start to build msg passing neural network
for learning_rate in [1e-3]:        # search learning rate
	for msg_passing in [2]:         # search for msg passing rounds

		# create folder to save the trained model
		Ori_PATH = 'Model_save/' 
		
		# save name
		Save_name = 'Lr-' + str(learning_rate) + '-msg-' + str(msg_passing)
		PATH = Ori_PATH + Save_name

		os.makedirs(PATH,exist_ok = True)

		# other hyper-para spec
		lr_min      = 1e-6              # keep the minimal learning rate the same, avoiding updates that is too small
		decay       = 0.998            	# learning rate decay rate
		
		# call model
		model = MsgPassingNN(device, input_size_Edge, output_size_node, msg_passing)

		# convert to GPU
		model = model.to(device) 
		print(model)
		pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print( 'Number of trainable para is:'  + str(pytorch_total_params) )
		
		# standard loss function
		criterion = nn.MSELoss()

		# calculate corresponding epoch number
		num_epochs = int(math.log(lr_min/learning_rate, decay))
		print('Number of epoch  is: ' + str(num_epochs))

		# define optimizer
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
		
		# build lr scheduler
		lambda1 = lambda epoch: decay ** epoch # lr scheduler by lambda function
		scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1) # define lr scheduler with the optim



