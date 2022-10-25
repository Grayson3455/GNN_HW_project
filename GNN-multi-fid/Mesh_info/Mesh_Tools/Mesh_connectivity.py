# Mesh --> graph tools
import numpy as np
import meshio

# mesh information extraction for 2D meshes
# Inputs:
#        mesh_name: path of the math file
# Outputs:
#        cells: list of finite element d labels
#        points: nodal coordinates
#        A     : the adj matrix of this mesh graph

def Mesh_info_extraction2D(mesh_name):
	
	Mesh      = meshio.read(mesh_name)	
	Cells     = (Mesh.cells_dict)['triangle']   # tri3 cells
	Points    =  Mesh.points                    # nodal points
	nELE      =  len(Cells[:,0])    			# number of element

	# construct adj matrix, assuming un-directed graph
	V = len(Points)     # number of nodes
	A = np.zeros((V,V), dtype=int) # initialize adj matrix

	# loop tho elements
	for i in range(nELE):
		# locate an element
		ele = Cells[i]
		# loop vertices of the current element
		for p in ele:
			neighbors = np.setdiff1d(ele,p)
			# loop neighbors of the current vertex
			for q in neighbors:
				A[p,q] = 1
				A[q,p] = 1

	return Cells, Points, A

