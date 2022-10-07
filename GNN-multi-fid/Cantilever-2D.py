# fenics solver for linear elasticity
import numpy as np
from fenics import *
import math

# sym-ed gradient tensor (strain)
def epsilon(u):
    return sym(grad(u)) 

# isotropic stress
def sigma(u, lmd, mu):
    return 2*mu*epsilon(u) + lmd * div(u) * Identity(u.geometric_dimension()) 

# define external loading components
fx,fy = 0,-10

# define material properties
E  = 1e6   # young's modulus 
nu = 0.3   # possion's ratio

# load mesh
#mesh = Mesh("Mesh_info/canti2D.xml")
mesh = RectangleMesh( Point(0, 0), Point(5, 1), 100, 20, diagonal="right")
# lame parameters
lmd = E*nu/ ( (1+nu)*(1-2*nu) )
mu  = E/ ( 2*(1+nu) )

lmd = 2*lmd*mu/(lmd+2*mu) # changed due to plane stress

l = 5 # length
H = 1 # width

# discretization details
p         = 3       # polynomial order

# define function space 
U   = VectorFunctionSpace(mesh, "CG", p) 

# define boundary condtion
# x[0] ~ 0 is clamped
def X_L(x,on_boundary):
    return on_boundary and np.isclose(x[0], 0)

bc   = DirichletBC(U, Constant((0.,0.)), X_L)

#------------------start to build to weak form----------------#

# trial and test function are form the same functional space
u = TrialFunction(U)
v = TestFunction(U)

# external loading
f = Constant((0,fy))

# set bilinear form
A = inner(sigma(u, lmd, mu) , epsilon(v))*dx

# set linear form
L = dot(f, v) * dx 



# get the problem
u_h = Function(U)
solve(A==L, u_h, bc)

sol_save = File("p="+str(p)+"-Displacement.pvd")
sol_save << u_h

# calculate L2 error 
W = 0.001
I = H**3*W/12
x = SpatialCoordinate(mesh)

ux_exact = 0.0
uy_exact = fy*x[0]*x[0]/24/E/I*(6*l*l-4*l*x[0] + x[0]*x[0])
u_exact = as_vector((ux_exact,uy_exact))
L2_error = math.sqrt(assemble(inner(u_h-u_exact,u_h-u_exact)*dx)  )
print(L2_error)