# Convert a 3D unstructured mesh to a graph

import numpy as np
import meshio

# Mesh info
mesh_name =  "Mesh_info/beam.vtk"  
Mesh      =  meshio.read(mesh_name)  		# import mesh
Cells     =  (Mesh.cells_dict)['tetra']     # tet4 elements
Facets    = (Mesh.cells_dict)['triangle']   # tri3 boundary facets
Points    =  Mesh.points                    # nodal points
nELE      =  len(Cells[:,0])    			# number of element

# construct adj matrix, assuming un-directional
V = len(Points)     # number of nodes
A = np.zeros((V,V)) # initialize adj matrix

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

# mesh partition using parmetis
