# The main difference here compared to `solve_emi_lagrange_iface` is that
# we use one DG space
from fvm_utils import CellCenterDistance
import dolfin as df

df.set_log_level(50)


def get_system_fvm(u_prev, ju, subdomains, boundaries, dt, data):
    '''Implicit discretization'''
    V = u_prev.function_space()
    assert V.ufl_element().degree() == 0
    
    u, v = df.TrialFunction(V), df.TestFunction(V)

    dt = df.Constant(dt)
    gamma = df.Constant(1/dt)

    u_bdry, f_vol, g_robin = data['solution'], data['f'], data['robin']    

    mesh = V.mesh()
    dx = df.Measure('dx', domain=mesh, subdomain_data=subdomains)
    dS = df.Measure('dS', domain=mesh, subdomain_data=boundaries)
    ds = df.Measure('ds', domain=mesh, subdomain_data=boundaries)    
    
    iface_tags = tuple(g_robin.keys())
    dtags = (1, 2, 3, 4)

    pdegree = V.ufl_element().degree()
    gdim = mesh.geometry().dim()
    gamma_DG = df.Constant(1)

    nGamma = interface_normal(subdomains)
    
    a = (sum(gamma*df.inner(jump(u, nGamma), jump(v, nGamma))*dS(tag) for tag in iface_tags))
    # Add the DG bits
    n, hF = df.FacetNormal(mesh), CellCenterDistance(mesh)
    a += gamma_DG/df.avg(hF)*df.inner(df.jump(u), df.jump(v))*dS(0)
    # It has Dirichlet bits
    a += sum(gamma_DG/hF*df.inner(u, v)*ds(tag) for tag in dtags)

    ju_ = df.avg(jU)
    # rhs
    L = (df.inner(f_vol[0], v)*dx(1) + df.inner(f_vol[1], v)*dx(2)
         + sum(gamma*df.inner(ju_, jump(v, nGamma))*dS(tag) for tag in iface_tags)
         + sum(df.inner(g_robin[tag], jump(v, nGamma))*dS(tag) for tag in iface_tags))

    # Dirichlet contrib
    L += sum(gamma_DG/hF*df.inner(u_bdry[0], v)*ds(tag) for tag in dtags)
    
    return a, L, None

    
def get_system(u_prev, ju, subdomains, boundaries, dt, data):
    '''Implicit discretization'''
    V = u_prev.function_space()

    if V.ufl_element().degree() == 0:
        return get_system_fvm(u_prev, ju, subdomains, boundaries, dt, data)
    
    u, v = df.TrialFunction(V), df.TestFunction(V)

    dt = df.Constant(dt)
    gamma = df.Constant(1/dt)

    u_bdry, f_vol, g_robin = data['solution'], data['f'], data['robin']    

    mesh = V.mesh()
    dx = df.Measure('dx', domain=mesh, subdomain_data=subdomains)
    dS = df.Measure('dS', domain=mesh, subdomain_data=boundaries)
    ds = df.Measure('ds', domain=mesh, subdomain_data=boundaries)    
    
    iface_tags = tuple(g_robin.keys())
    dtags = (1, 2, 3, 4)

    pdegree = V.ufl_element().degree()
    gdim = mesh.geometry().dim()
    gamma_DG = df.Constant(40*pdegree*gdim)

    nGamma = interface_normal(subdomains)
    
    a = (df.inner(df.grad(u), df.grad(v))*dx +
         sum(gamma*df.inner(jump(u, nGamma), jump(v, nGamma))*dS(tag) for tag in iface_tags))
    # Add the DG bits
    n, hF = df.FacetNormal(mesh), df.CellDiameter(mesh)
    a += (
        -df.dot(df.avg(df.grad(u)), df.jump(v, n))*dS(0)
        -df.dot(df.avg(df.grad(v)), df.jump(u, n))*dS(0)
        + gamma_DG/df.avg(hF)*df.inner(df.jump(u), df.jump(v))*dS(0)
    )
    # It has Dirichlet bits
    a += sum(-df.inner(df.dot(df.grad(u), n), v)*ds(tag)
             -df.inner(df.dot(df.grad(v), n), u)*ds(tag)
             + gamma_DG/hF*df.inner(u, v)*ds(tag) for tag in dtags)

    ju_ = df.avg(ju)    
    # rhs
    L = (df.inner(f_vol[0], v)*dx(1) + df.inner(f_vol[1], v)*dx(2)
         + sum(gamma*df.inner(ju_, jump(v, nGamma))*dS(tag) for tag in iface_tags)
         + sum(df.inner(g_robin[tag], jump(v, nGamma))*dS(tag) for tag in iface_tags))

    # Dirichlet contrib
    L += sum(-df.inner(df.dot(df.grad(v), n), u_bdry[0])*ds(tag)
             + gamma_DG/hF*df.inner(u_bdry[0], v)*ds(tag)
             for tag in dtags)
    
    return a, L, None
             
# --------------------------------------------------------------------

if __name__ == '__main__':
    from emi_utils import setup_mms
    from dg_utils import setup_geometry
    from dg_utils import (interface_normal, patch_interpolate, pcws_constant_project,
                          plus, minus)
    from utils import update_time
    from xii import *

    pdegree = 1
    # --------------
    T_final = 1.0
    dt = 1E-1
    alpha = 1E0 # 1E-3

    mms_data = setup_mms(alpha_value=alpha)

    print_freq = 0.1*T_final/dt
    # -----------
    n = 32
    subdomains, boundaries = setup_geometry(n)
    nGamma = interface_normal(subdomains)

    update_time(mms_data, time=0.)
    
    mesh = subdomains.mesh()
    V = df.FunctionSpace(mesh, 'DG', pdegree)

    u_prev = df.Function(V)
    patch_interpolate(V, subdomains, {1: mms_data['solution'][0],
                                      2: mms_data['solution'][1]})

    # Represent the jump at the interface in the HDiv trace
    # It should match the degree of V but then projection starts
    # to cost money?
    Vi = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    dM = df.Measure('dS', domain=mesh, subdomain_data=boundaries)

    jump = lambda v, n: plus(v, n) - minus(v, n)
    
    ju = df.Function(Vi)
    # u2 - u1    
    ju = pcws_constant_project(jump(u_prev, nGamma),
                               V=Vi,
                               fV=ju,
                               dM=dM, tags=(5, 6, 7, 8))
    
    a, L, bcs = get_system(u_prev, ju, subdomains, boundaries, dt, data=mms_data)

    assembler = df.SystemAssembler(a, L, bcs)
    A, b = df.Matrix(), df.Vector()

    assembler.assemble(A)
    print(f'|A| {A.norm("linf")}')
    
    solver = df.LUSolver(A, 'umfpack')
    x = u_prev.vector()
    
    step = 0
    t = step*dt
    while t < T_final:
        step += 1
        t = step*dt
        update_time(mms_data, t)
        
        assembler.assemble(b)
        solver.solve(x, b)
        # Update jump
        # u2 - u1
        ju = pcws_constant_project(jump(u_prev, nGamma),
                                   V=Vi,
                                   fV=ju,
                                   dM=dM, tags=(5, 6, 7, 8))

        step % print_freq == 0 and print(f'\tt = {t:.2f} |b| = {b.norm("linf"):.4E}, |x| = {x.norm("linf"):.4E}')
    time = mms_data['solution'][0].time
    assert abs(time - mms_data['solution'][1].time) < 1E-13

    dx = df.Measure('dx', subdomain_data=subdomains, metadata={'quadrature_degree': 5})
    error0 = df.sqrt(
        df.assemble((u_prev - mms_data['solution'][0])**2*dx(1) + (u_prev - mms_data['solution'][1])**2*dx(2))
    )
    error1 = -1
                 
    ndofs = V.dim()
    mesh = boundaries.mesh()
    print(f'time = {time:.2E} h = {mesh.hmin():.2E} dt = {dt:.2E} => |u(T)-uh(T)|_1 = {error1:.4E} |u(T)-uh(T)|_0 = {error0:.4E} # = {ndofs}')
