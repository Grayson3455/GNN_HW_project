# fenics solver for linear elasticity for the full circle
import numpy as np
from fenics import *
import ufl
from mshr import *

# sym-ed gradient tensor (strain)
def epsilon(u):
    return sym(grad(u)) 

# isotropic stress
def sigma(u, lmd, mu):
    return 2*mu*epsilon(u) + lmd * div(u) * Identity(2) 

# geo info
r_0 = 0.09
r_i = 0.075

# domain = Circle(Point(0,0), r_0)-Circle(Point(0,0), r_i)
# mesh = generate_mesh(domain, 20)

mesh = Mesh('Mesh_info/Meshes/Full-circle_lv1.xml')

# define material properties
E  = 200e9   # young's modulus 
nu = 0.3     # possion's ratio

# lame parameters for plain strain
lmd = E*nu/ ( (1+nu)*(1-2*nu) )
mu  = E/ ( 2*(1+nu) )

#lmd = 2*lmd*mu /(lmd + 2*mu) # if plan-stress

# radial traction
P_r = 30e6
x = SpatialCoordinate(mesh) 

# decomposition back to carti
t_r = as_vector((P_r*x[0]/r_i,P_r*x[1]/r_i))

# discretization details
p         = 2    # polynomial order

# define function space 
U   = VectorFunctionSpace(mesh, "CG", p) 

# Neumann bc only
class arc(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near( (x[0]**2 + x[1]**2)**0.5, r_i)

Arc_neu = arc()
Neumann_arc = MeshFunction("size_t", mesh, 1) # 1 means line
Neumann_arc.set_all(0) # initialization
Arc_neu.mark(Neumann_arc,0)

ds = Measure('ds', domain=mesh, subdomain_data=Neumann_arc)

# print( assemble(inner(t_r,FacetNormal(mesh))*ds(0)) )

# trial and test function are form the same functional space
u = TrialFunction(U)
v = TestFunction(U)

# set bilinear form
A = inner(sigma(u, lmd, mu) , epsilon(v))*dx
# the stiffness matrix
K = assemble(A)
# set linear form
L = dot(v, t_r) * ds(0) 

# solve the problem
u_h = Function(U)
solve(A==L, u_h)

f_ext_known = assemble(L)
f_ext_unknown = assemble( action(A,u_h)  ) - f_ext_known

x_dofs = U.sub(0).dofmap().dofs()
y_dofs = U.sub(1).dofmap().dofs()

Fx = 0
for i in x_dofs:
    print(f_ext_unknown[i])
    Fx += f_ext_unknown[i]

Fy = 0
for i in y_dofs:
    Fy += f_ext_unknown[i]

# The following must be equal to -1 * f
print("Horizontal reaction force: %.10f"%(Fx))
print("Vertical reaction force: %.10f"%(Fy))




# exact solution
r = (x[0]*x[0] + x[1]*x[1])**0.5 # force float type
u_r = P_r * (1+nu) / E * (r / ( (r_0/r_i)**2 -1 ) ) * ( 1-2*nu + r_0/r * r_0/r )

u_exact = as_vector((u_r * x[0]/r, u_r * x[1]/r))


sol_save = File("p="+str(p)+"-Full-circle.pvd")
sol_save << u_h

sol_save = File("p="+str(p)+"-Full-circle-diff.pvd")
sol_save << project(u_h - u_exact,U)


# K = K.array()
# u_h = u_h.vector().get_local()
# print(K @ u_h)

# calculate L2 norm

L2_error = (assemble( (u_h-u_exact)**2 *dx)  )**0.5
semi_H1_error = (assemble( (grad(u_h-u_exact))**2 *dx)  )**0.5
print(L2_error)
print(semi_H1_error)