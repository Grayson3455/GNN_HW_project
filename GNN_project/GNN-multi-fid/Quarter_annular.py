# fenics solver for 2D linear elasticity 
# geo: quarter annular
# status: plain strain
# ref: https://fenicsproject.org/pub/tutorial/html/._ftut1008.html
# ref: https://fenics-solid-tutorial.readthedocs.io/en/latest/2DPlaneStrain/2D_Elasticity.html
# ref: https://github.com/CMGLab/IGA-Notes
import numpy as np
from fenics import *
import ufl
from mshr import *
from matplotlib import pyplot as plt

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

#lmd = 2*lmd*mu /(lmd + 2*mu) # if plan-stress

# radial traction
P_r = 30e6

# discretization details
plt.figure()
for p in [1]:
    
    L2_error  = []   # l2 error placeholder
    semi_H1_error = [] # semi-H1 error placeholder
    MeshSize = []      # meshsize placeholder
    
    for lv in range(1,7): # 6 levels of mesh refinement

        mesh       = Mesh('Mesh_info/Annular/annular_lv'+str(lv)+'.xml')
        bdies      = MeshFunction("size_t", mesh, 'Mesh_info/Annular/annular_lv'+str(lv)+'_facet_region.xml')

        # mesh info
        x = SpatialCoordinate(mesh)
        ds = ds(subdomain_data=bdies) 
        MeshSize.append(mesh.hmax())

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

        # solve the problem
        u_h = Function(U)
        solve(A==L, u_h, bcs)

        # exact solution
        r = (x[0]*x[0] + x[1]*x[1])**0.5 # force float type
        u_r = P_r * (1+nu) / E * (r / ( (r_0/r_i)**2 -1 ) ) * ( 1-2*nu + r_0/r * r_0/r )

        u_exact = as_vector((u_r * x[0]/r, u_r * x[1]/r))

        sol_save = File("Sol/annular/Mesh_lv="+str(lv)+"-p="+str(p)+".pvd")
        sol_save << u_h

        L2_error.append((assemble( (u_h-u_exact)**2 *dx)  )**0.5)
        semi_H1_error.append((assemble( (grad(u_h-u_exact))**2 *dx)  )**0.5)

    plt.subplot(1,2,1)
    plt.loglog(MeshSize, L2_error, '-*')
    print( (np.log(L2_error[5]) - np.log(L2_error[4])) /(np.log(MeshSize[5]) - np.log(MeshSize[4]))  )
    #plt.loglog(MeshSize, np.array(MeshSize)**2, '-*')
    plt.subplot(1,2,2)
    plt.loglog(MeshSize, semi_H1_error, '-*')
    print( (np.log(semi_H1_error[5]) - np.log(semi_H1_error[4])) /(np.log(MeshSize[5]) - np.log(MeshSize[4]))  )
    #plt.loglog(MeshSize, np.array(MeshSize)**1, '-*')
plt.show()
