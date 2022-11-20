# Low fid solution to high fid mesh via Graph neural network
import meshio
import torch
from Mesh_info.Mesh_Tools.Mesh_connectivity import *
import os
from GNN_tools.GNN import *
from GNN_tools.DNN_tools import *
import math

# Note: no idea how to build the training and testing dataset, tbd 
# this code is only build for a ``big regression"" problem

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
Cells, Points, AN = Mesh_info_extraction2D(mesh_name)

# tensorize
X = torch.from_numpy(Low_Dis).float().to(device)
Y = torch.from_numpy(High_Dis).float().to(device)

# scale the data, using max-min scaler
scale_minX, scale_maxX, X = max_min_scaling(X)
scale_minY, scale_maxY, Y = max_min_scaling(Y)

input_size_Edge  = 6 # vp cat with vq
output_size_node = 3

# start to build msg passing neural network
for learning_rate in [1e-3]:        # search learning rate
	for msg_passing in [3]:         # search for msg passing rounds

		# create folder to save the trained model
		Ori_PATH = 'Model_save/' 
		
		# save name
		Save_name = 'Lr-' + str(learning_rate) + '-msg-' + str(msg_passing)
		PATH = Ori_PATH + Save_name

		os.makedirs(PATH,exist_ok = True)

		# other hyper-para spec
		lr_min      = 1e-6              # keep the minimal learning rate the same, avoiding updates that is too small
		decay       = 0.995            	# learning rate decay rate
		
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
		#num_epochs = 51
		print('Number of epoch  is: ' + str(num_epochs))

		# define optimizer
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
		
		# build lr scheduler
		lambda1 = lambda epoch: decay ** epoch # lr scheduler by lambda function
		scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1) # define lr scheduler with the optim


		#-------------training--prepare--------------------#
		# loss and accuracy saves placeholders
		train_save   = [] # epoch-wise training loss save 
		train_acc_save   = [] # epoch-wise training accuracy save 

		#--------------------------Start-training----------------------------------#
		for epoch in range(num_epochs):

			#-----------------Train steps--------------#
			model, train_loss_per_epoch, train_acc_per_epoch = GNN_train(device, model, criterion, optimizer, \
																X, Y, Cells, Points, AN)

			# save training results per epoch
			train_save.append(train_loss_per_epoch)
			train_acc_save.append(train_acc_per_epoch)

			# print out values for training stats
			if epoch%50 == 0:
				print("Training: Epoch: %d, mse loss: %1.5e" % (epoch, train_save[epoch]) ,\
					", mse acc : %1.5f" % (train_acc_save[epoch]), \
					', lr=' + str(optimizer.param_groups[0]['lr']))

			# update learning rate
			scheduler.step()		


		# save the model
		model_save_name   = PATH + '/model.pth'
		torch.save(model.state_dict(), model_save_name)

		# plot loss curves
		train_save = (torch.FloatTensor(train_save)).cpu()
		train_acc_save = (torch.FloatTensor(train_acc_save)).cpu()
		simple_loss_plot(PATH, train_save, train_acc_save)
