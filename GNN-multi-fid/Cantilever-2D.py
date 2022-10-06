# fenics solver for linear elasticity
# ref: https://jorgensd.github.io/dolfinx-tutorial/chapter2/linearelasticity_code.html

import numpy as np
from dolfinx import *
import ufl
import meshio

def C2D(lmd,mu,case='plane-strain'):
    if case == 'plane-strain':
        lmd =  lmd
    elif case == 'plane-stress':
        lmd = 2*lmd*mu/(lmd+2*mu)
    return np.array([[lmd + 2*mu, lmd, 0],[lmd,lmd + 2*mu,0],[0,0,mu]])

# define external loading components
fx,fy = 0,-0.1

# define material properties
E  = 1e6   # young's modulus 
nu = 0.3   # possion's ratio

# lame parameters
lmd = E*nu/ ( (1+nu)*(1-2*nu) )
mu  = E/ ( 2*(1+nu) )


# sym-ed gradient tensor
def epsilon(u):
    return ufl.sym(ufl.grad(u)) 

# discretization details
p         =  1        # polynomial order

# construct the mesh obj
Mesh = meshio.read("Mesh_info/beam.msh")

# define function space
U    = fem.FunctionSpace(Mesh, ("CG", p))

# define boundary condtion
# x[0] ~ 0 is clamped
def X_L(x):
    return np.isclose(x[0], 0)

# locate boundary facets
BC_dim = Mesh.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(Mesh, BC_dim, X_L)

# enforce boundary condition
u_Xl = np.array([0,0], dtype=ScalarType)
bc   = fem.dirichletbc(u_Xl, fem.locate_dofs_topological(U, BC_dim, boundary_facets), U)

#------------------start to build to weak form----------------#

# trial and test function are form the same functional space
u = ufl.TrialFunction(U)
v = ufl.TestFunction(U)

# external loading
f = fem.Constant(domain, ScalarType((0, fy)))

A = inner(dot(C2D(lmd,mu,'plane-stress'),epsilon(u)),epsilon(v))*dx

