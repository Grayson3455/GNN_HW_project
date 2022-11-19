# # fenics solver for 2D linear elasticity 
# # geo: 2D cantilever
# # status: plain stress
# # ref: https://fenicsproject.org/pub/tutorial/html/._ftut1008.html
# # ref: https://fenics-solid-tutorial.readthedocs.io/en/latest/2DPlaneStrain/2D_Elasticity.html
# # ref: https://github.com/CMGLab/IGA-Notes
# # ref: https://comet-fenics.readthedocs.io/en/latest/demo/elasticity/2D_elasticity.py.html
# # ref: https://jorgensd.github.io/dolfinx-tutorial/

# # Boundary condition:
# #                   Left (x=0): d = 0

# from dolfin import *
# from fenics import *


# # sym-ed gradient tensor (strain)
# def epsilon(u):
#     return sym(grad(u)) 

# # isotropic stress
# def sigma(u, lmd, mu):
#     return 2*mu*epsilon(u) + lmd * div(u) * Identity(2) 

# f = Constant((0,-0.1)) # external loading

# #----------------------Meshing--------------------------#
# L = 10. # length of the cantilever
# H = 1.  # height of the cantilever
# h = 0.5 # mesh size

# # lv0 is the low-fid, lv1 is the high fid
# if h == 0.1:
#     lv = 1
# if h == 0.5:
#     lv = 0

# P1 = Point(0,0)     # left corner
# P2 = Point(L,H)     # right corner

# nx = int(L/h)             # num of points in x direction
# ny = int(H/h)             # num of points in y direction

# # create cantilever 2D mesh with quad element
# mesh = RectangleMesh.create([P1,P2],[nx,ny],CellType.Type.quadrilateral) 

# k   = 1 # polynomial order

# # Define function space 
# U   = VectorFunctionSpace(mesh, "CG", k) 
# #------------------------------------------------------#


# #-------------------material properties-----------------#
# E  = 1e6     # young's modulus 
# nu = 0.3     # possion's ratio

# # lame parameters for plain strain
# lmd = E*nu/ ( (1+nu)*(1-2*nu) )
# mu  = E/ ( 2*(1+nu) )

# lmd = 2*lmd*mu /(lmd + 2*mu) # if plan-stress
# #--------------------------------------------------------#


# #-------------------boundary conditions------------------#
# def uD(x, on_boundary):   # x = 0, u = 0
#     return near(x[0], 0)

# bc_x0 = DirichletBC(U, Constant((0., 0.)), uD)
# #--------------------------------------------------------#

# #-------------------Variational forms--------------------#
# # trial and test function are form the same functional space
# u = TrialFunction(U)
# v = TestFunction(U)

# # set bilinear form
# A = inner(sigma(u, lmd, mu) , epsilon(v))*dx

# # set linear form
# L = dot(f, v) * dx
# #--------------------------------------------------------#


# #------------------solve and save------------------------#
# # solve the problem
# u_h = Function(U)
# solve(A==L, u_h, bc_x0)

# sol_save = File("Sol/canti2D/Mesh_lv="+str(lv)+"-p="+str(k)+".pvd")
# sol_save << u_h
# #--------------------------------------------------------#




#------------------------------FENICSX SOLUTION-------------------------------------------------#
#TODO: figure out how to post-process xdmf solutions

# fenics solver for 2D linear elasticity 
# geo: 2D cantilever
# status: plain stress
# ref: https://fenicsproject.org/pub/tutorial/html/._ftut1008.html
# ref: https://fenics-solid-tutorial.readthedocs.io/en/latest/2DPlaneStrain/2D_Elasticity.html
# ref: https://github.com/CMGLab/IGA-Notes
# ref: https://comet-fenics.readthedocs.io/en/latest/demo/elasticity/2D_elasticity.py.html
# ref: https://jorgensd.github.io/dolfinx-tutorial/

# Boundary condition:
#                   Left (x=0): d = 0

# Note: quad element is not well supported in classical fenics
#		we'd have to use the fenicsX

import dolfinx
from mpi4py import MPI
from dolfinx import mesh
import numpy as np
from dolfinx import fem
from petsc4py.PETSc import ScalarType
import ufl

# sym-ed gradient tensor (strain)
def epsilon(u):
    return ufl.sym(ufl.grad(u)) 

# isotropic stress
def sigma(u, lmd, mu):
    return 2*mu*epsilon(u) + lmd * ufl.div(u) * ufl.Identity(2) 


#------------------problem-set-up---------------------#
fy = 0.1 # y-component 

# material properties
E  = 1e6     # young's modulus 
nu = 0.3     # possion's ratio

# lame parameters for plain strain
lmd = E*nu/ ( (1+nu)*(1-2*nu) )
mu  = E/ ( 2*(1+nu) )

#lmd = 2*lmd*mu /(lmd + 2*mu) # if plan-stress
#-----------------------------------------------------#


#-----------------create mesh-------------------------#
L = 10. # length of the cantilever
H = 1.  # height of the cantilever
h = 0.5 # mesh size

corners = [ np.array([0,0]), np.array([L,H])]
num_points = [int(L/h), int(H/h)]

# use built-in mesh 
domain = mesh.create_rectangle(MPI.COMM_WORLD, \
								 corners , num_points, \
								 mesh.CellType.quadrilateral)
#------------------------------------------------------#

#------------------discretization----------------------#
p = 1 # linear element
U = fem.VectorFunctionSpace(domain, ("CG", p))
#------------------------------------------------------#


#---------------------boundary condition---------------#
def clamped(x):
    return np.isclose(x[0], 0.0) # the side x=0 is clamped

facet_dim = domain.topology.dim - 1 # line--> dim = 1
boundary_facets = mesh.locate_entities_boundary(domain, facet_dim, clamped) # located the x = 0 line
u_D        = np.array([0,0], dtype=ScalarType) # dirichlet boundary condition
bc_clamped = fem.dirichletbc(u_D, fem.locate_dofs_topological(U, facet_dim, boundary_facets), U) # bc constructor
#------------------------------------------------------#

#--------------------variational form------------------#
# trial function and test function
u = ufl.TrialFunction(U)
v = ufl.TestFunction(U)
f = fem.Constant(domain, ScalarType((0, -fy))) # external loading


# bilinear form
A = ufl.inner(sigma(u, lmd, mu) , epsilon(v))*ufl.dx

# linear form
l = ufl.dot(f, v) * ufl.dx
#------------------------------------------------------#

#------------------------------------solve and save------------------------------------------#
problem = fem.petsc.LinearProblem(A, l, bcs=[bc_clamped], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# create high fid mesh
h = 0.05
num_points = [int(L/h), int(H/h)]
domain = mesh.create_rectangle(MPI.COMM_WORLD, \
                                 corners , num_points, \
                                 mesh.CellType.quadrilateral)

VH = fem.VectorFunctionSpace(domain, ("CG", p))
#uH = fem.Function(VH, dtype=ScalarType)
#uH.interpolate(uh)

uH = fem.project(uh,VH)

# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "Sol/Canti2D/canti2D" + "-lv=" + str(lv) + ".xdmf", "w") as f:
#     f.write_mesh(domain)
#     f.write_function(uh)

#-------------------------------------------------------------------------------------------#
