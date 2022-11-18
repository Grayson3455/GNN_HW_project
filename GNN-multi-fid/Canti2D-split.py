# we apply the idea shown in Black, Najifi's paper 
# the refinement from the low-fid solution to high-fid solution is in a hierarchical way


from dolfin import *
from fenics import *

P1 = Point(0,0)
P2 = Point(10,1)

nx = 100
ny = 10

mesh = RectangleMesh.create([P1,P2],[nx,ny],CellType.Type.quadrilateral)


mesh_file = File("mesh_quad_test.pvd")
mesh_file << mesh

# import meshio
# from dolfinx.io import XDMFFile
# from mpi4py import MPI
# from dolfinx import fem
# import h5py

# # load low-fid mesh
# path = 'Sol/Canti2D/canti2D-lv=0.xdmf'
# with XDMFFile(MPI.COMM_WORLD, path, "r") as xdmf:
#     mesh = xdmf.read_mesh(name = "mesh")


# # create function associated with the current mesh
# U   = fem.VectorFunctionSpace(mesh, ("CG", 1))
# u_h = fem.Function(U) # create intepolator


# # extract geo information
# geo       = mesh.geometry # geo obj
# points    = geo.x         # all points
# dim       = mesh.topology.dim # geo dimension

# # read h5 file 
# h5_path = 'Sol/Canti2D/canti2D-lv=0.h5'

# with h5py.File(h5_path, "r") as f:
# 	print(f)
