# map low fidelity solution to high fidelity mesh
# case: 2D cantilever

from dolfin import *
from fenics import *
import numpy as np

parameters['allow_extrapolation'] = True

# sym-ed gradient tensor (strain)
def epsilon(u):
    return sym(grad(u)) 

# isotropic stress
def sigma(u, lmd, mu):
    return 2*mu*epsilon(u) + lmd * div(u) * Identity(2) 

f = Constant((0,-0.1)) # external loading

#----------------------Meshing--------------------------#
Length = 10. # length of the cantilever
Height = 1.  # height of the cantilever
h = 0.5 # mesh size

P1 = Point(0,0)     # left corner
P2 = Point(Length,Height)     # right corner

nx = int(Length/h)             # num of points in x direction
ny = int(Height/h)             # num of points in y direction

# create cantilever 2D mesh with quad element
mesh = RectangleMesh.create([P1,P2],[nx,ny],CellType.Type.quadrilateral) 

k   = 1 # polynomial order

# Define function space 
U   = VectorFunctionSpace(mesh, "CG", k) 
#------------------------------------------------------#


#-------------------material properties-----------------#
E  = 1e6     # young's modulus 
nu = 0.3     # possion's ratio

# lame parameters for plain strain
lmd = E*nu/ ( (1+nu)*(1-2*nu) )
mu  = E/ ( 2*(1+nu) )

lmd = 2*lmd*mu /(lmd + 2*mu) # if plan-stress
#--------------------------------------------------------#


#-------------------boundary conditions------------------#
def uD(x, on_boundary):   # x = 0, u = 0
    return near(x[0], 0)

bc_x0 = DirichletBC(U, Constant((0., 0.)), uD)
#--------------------------------------------------------#

#-------------------Variational forms--------------------#
# trial and test function are form the same functional space
u = TrialFunction(U)
v = TestFunction(U)

# set bilinear form
A = inner(sigma(u, lmd, mu) , epsilon(v))*dx

# set linear form
L = dot(f, v) * dx
#--------------------------------------------------------#


#------------------solve and save------------------------#
# solve the problem
u_h = Function(U)
solve(A==L, u_h, bc_x0)
#--------------------------------------------------------#


#-----------------build-high-fid-mesh--------------------#
h  = 0.1
nx = int(Length/h)             # num of points in x direction
ny = int(Height/h)             # num of points in y direction

# create cantilever 2D mesh with quad element
mesh = RectangleMesh.create([P1,P2],[nx,ny],CellType.Type.quadrilateral) 

# define new function space 
U_H   = VectorFunctionSpace(mesh, "CG", k) 

# interp is not supported for quad element!
u_H_interp = project(u_h,U_H)


sol_H_save = File("Sol/canti2D/low-to-high.pvd")
sol_H_save << u_H_interp