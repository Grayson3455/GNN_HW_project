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
h = 0.1 # mesh size


if h == 0.1:
    lv = 1
if h == 0.5:
    lv = 0


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
L = ufl.dot(f, v) * ufl.dx
#------------------------------------------------------#

#------------------------------------solve and save------------------------------------------#
problem = fem.petsc.LinearProblem(A, L, bcs=[bc_clamped], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "Sol/Canti2D/canti2D" + "-lv=" + str(lv) + ".xdmf", "w") as f:
    f.write_mesh(domain)
    f.write_function(uh)
#-------------------------------------------------------------------------------------------#
