# map low fidelity solution to high fidelity mesh
# case: 2D annular

# Solve low fid problem
import numpy as np
from fenics import *

parameters['allow_extrapolation'] = True

# sym-ed gradient tensor (strain)
def epsilon(u):
    return sym(grad(u)) 

# isotropic stress
def sigma(u, lmd, mu):
    return 2*mu*epsilon(u) + lmd * div(u) * Identity(2) 

# geo info
r_0 = 0.09  # outer circle radius
r_i = 0.075 # inner circle radius


# material properties
E  = 200e9   # young's modulus 
nu = 0.3     # possion's ratio

# lame parameters for plain strain
lmd = E*nu/ ( (1+nu)*(1-2*nu) )
mu  = E/ ( 2*(1+nu) )

p = 1 # polynomial degree

# radial traction
P_r = 30e6
lv  = 1   # low fid level
mesh       = Mesh('Mesh_info/Annular/annular_lv'+str(lv)+'.xml')
bdies      = MeshFunction("size_t", mesh, 'Mesh_info/Annular/annular_lv'+str(lv)+'_facet_region.xml')

# mesh info
x = SpatialCoordinate(mesh)
ds = ds(subdomain_data=bdies) 

# define function space 
U   = VectorFunctionSpace(mesh, "CG", p) 

# assign bc to only one component, the other one is shear traction-free
bc_bot = DirichletBC(U.sub(1), 0.0 , bdies, 8) # 8 means bot, check .geo file
bc_top = DirichletBC(U.sub(0), 0.0 , bdies, 7) # 7 means top, check .geo file

bcs = [bc_bot, bc_top] # group bc

# traction decomposition back to carti
t_r = as_vector((P_r*x[0]/r_i,P_r*x[1]/r_i))

# print(assemble( dot(t_r, FacetNormal(mesh))*ds(5)) ) # 5 means inner circ bdy, check .geo file

# trial and test function are form the same functional space
u = TrialFunction(U)
v = TestFunction(U)

# set bilinear form
A = inner(sigma(u, lmd, mu) , epsilon(v))*dx

# set linear form
L = dot(v, t_r) * ds(5)  # 5 means inner circ bdy

# solve the low-fidelity problem
u_h = Function(U)
solve(A==L, u_h, bcs)

# intepolate low-fid solution to high fid mesh
#--------------------------------------------#
lv_H         = 3 
mesh_H       = Mesh('Mesh_info/Annular/annular_lv'+str(lv_H)+'.xml')
# define function space 
U_H   = VectorFunctionSpace(mesh_H, "CG", p) 
u_H_interp = Function(U_H)
u_H_interp.interpolate(u_h)


sol_H_save = File("Sol/annular/lv"+str(lv)+"-to-lv"+str(lv_H)+".pvd")
sol_H_save << u_H_interp
#--------------------------------------------#