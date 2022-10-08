# fenics solver for linear elasticity
import numpy as np
from fenics import *
import ufl

# overkill for convergence
QUAD_DEG = 10

# sym-ed gradient tensor (strain)
def epsilon(u):
    return sym(grad(u)) 

# isotropic stress
def sigma(u, lmd, mu):
    return 2*mu*epsilon(u) + lmd * tr(epsilon(u)) * Identity(u.geometric_dimension()) 

# load mesh
mesh = Mesh("Mesh_info/annular.xml")

# define material properties
E  = 200e9   # young's modulus 
nu = 0.3     # possion's ratio

# lame parameters for plain strain
lmd = E*nu/ ( (1+nu)*(1-2*nu) )
mu  = E/ ( 2*(1+nu) )
r_0 = 0.09
r_i = 0.075
# radial traction
P_r = 30e6
x = SpatialCoordinate(mesh) 
# decomposition back to carti
t_r = as_vector((-P_r*x[0]/r_i,P_r*x[1]/r_i))

# discretization details
p         = 3    # polynomial order

# define function space 
U   = VectorFunctionSpace(mesh, "P", p) 

# Dirichlet BC
def Bottom_BC(x, on_boundary):
    return on_boundary and near(x[1], 0)
def Top_BC(x, on_boundary):
    return on_boundary and near(x[0], 0)

# assign bc to only one component, the other one is free of shear traction
bc_bot = DirichletBC(U.sub(1), 0.0 , Bottom_BC)
bc_top = DirichletBC(U.sub(0), 0.0 , Top_BC)

bcs = [bc_bot, bc_top]

# Neumann bc
class arc(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0]**2 + x[1]**2, r_i**2)

Arc_neu = arc()
Neumann_arc = MeshFunction("size_t", mesh,  mesh.topology().dim()-1)
Neumann_arc.set_all(0) # initialization
Arc_neu.mark(Neumann_arc,0)
dx = dx(metadata={"quadrature_degree":QUAD_DEG})
ds = Measure('ds', domain=mesh, subdomain_data=Neumann_arc,metadata={"quadrature_degree":QUAD_DEG})

# trial and test function are form the same functional space
u = TrialFunction(U)
v = TestFunction(U)

# set bilinear form
A = inner(sigma(u, lmd, mu) , epsilon(v))*dx

# set linear form
L = dot(v, t_r) * ds(0) 

# solve the problem
u_h = Function(U)
solve(A==L, u_h, bcs)
sol_save = File("p="+str(p)+"-Annular.pvd")
sol_save << u_h

# calculate L2 norm
r = (x[0]*x[0] + x[1]*x[1])**0.5 # force float type
u_r = P_r * (1+nu) / E * (r / ( (r_0/r_i)**2 -1 ) ) * ( 1-2*nu + r_0/r * r_0/r )

u_exact = as_vector((-u_r * x[0]/r, u_r * x[1]/r))

L2_error = (assemble( (u_h-u_exact)**2 *dx)  )**0.5
print(L2_error)