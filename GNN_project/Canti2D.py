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


# we project the low fid solution to high fid mesh via interpolation

from dolfin import *
from fenics import *

parameters['allow_extrapolation'] = True


# sym-ed gradient tensor (strain)
def epsilon(u):
    return sym(grad(u)) 

# isotropic stress
def sigma(u, lmd, mu):
    return 2*mu*epsilon(u) + lmd * div(u) * Identity(2) 

f = Constant((0,-0.1)) # external loading

#----------------------Meshing--------------------------#
L = 10. # length of the cantilever
H = 1.  # height of the cantilever

# define low-fid mesh size
h_low = 0.5 # mesh size

P1 = Point(0,0)     # left corner
P2 = Point(L,H)     # right corner

nx = int(L/h_low)             # num of points in x direction
ny = int(H/h_low)             # num of points in y direction

# create low-fid cantilever 2D mesh
mesh_low = RectangleMesh(P1, P2, nx, ny, diagonal='right')

k   = 1 # polynomial order

# Define function space 
U   = VectorFunctionSpace(mesh_low, "CG", k) 
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
l = dot(f, v) * dx
#--------------------------------------------------------#

#------------------solve and save------------------------#
# solve the problem
u_h_low = Function(U)
solve(A==l, u_h_low, bc_x0)

sol_save = File("Sol/canti2D/low-fid.pvd")
sol_save << u_h_low
#--------------------------------------------------------#


#---------------projection from low to high--------------#
# define high-fid mesh size
h_high = 0.05 

nx = int(L/h_high)    # num of points in x direction
ny = int(H/h_high)    # num of points in y direction

# create low-fid cantilever 2D mesh
mesh_high  = RectangleMesh(P1, P2, nx, ny, diagonal='right')
U_High     = VectorFunctionSpace(mesh_high, "CG", k) 
u_H_interp = Function(U_High)
u_H_interp.interpolate(u_h_low)

sol_H_save = File("Sol/canti2D/low-to-high.pvd")
sol_H_save << u_H_interp
#--------------------------------------------------------#