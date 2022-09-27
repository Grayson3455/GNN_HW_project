import numpy as np
import meshio

# Mesh info
mesh_name =  "beam.vtk"  
Mesh      =  meshio.read(mesh_name)  		# import mesh
Cells     =  (Mesh.cells_dict)['tetra']  
Facets    = (Mesh.cells_dict)['triangle']
Points    =  Mesh.points                    # nodal points
nELE      =  len(Cells[:,0])    			# number of element

print(Cells)