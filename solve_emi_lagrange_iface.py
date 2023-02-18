# The only difference here compared to `solve_emi_lagrange_iface` is that
# jump(u) is not computed in the weak form from intra/extre-cellular potentials
# but there data at the interface
from utils import check_bdry_tags
import dolfin as df

df.set_log_level(50)

def get_system(u_prev, ju, boundaries, dt, data):
    '''Dispatch'''
    V1, V2 = (u_prev[0].function_space(), u_prev[1].function_space())
    assert V1.ufl_element() == V2.ufl_element()
    elm = V1.ufl_element()

    if elm.family() == 'Lagrange':
        return get_system_cg(u_prev, ju, boundaries, dt, data)

    assert elm.family() == 'Discontinuous Lagrange'
    if elm.degree() > 0:
        return get_system_dg(u_prev, ju, boundaries, dt, data)

    
def get_system_dg(u_prev, ju, boundaries, dt, data):
    '''Implicit discretization'''
    V1, V2 = (u_prev[0].function_space(), u_prev[1].function_space())
    W = [V1, V2]
    
    u1, u2 = map(df.TrialFunction, W)
    v1, v2 = map(df.TestFunction, W)

    dt = df.Constant(dt)
    gamma = df.Constant(1/dt)

    u_bdry, f_vol, g_robin = data['solution'], data['f'], data['robin']    

    boundaries1, boundaries2 = boundaries
    # 1 is EXTERIOR
    mesh1 = boundaries1.mesh()
    ds1 = df.Measure('ds', domain=mesh1, subdomain_data=boundaries1)

    mesh2 = boundaries2.mesh()
    ds2 = df.Measure('ds', domain=mesh2, subdomain_data=boundaries2)
    
    iface_tags = tuple(g_robin.keys())
    dtags = (1, 2, 3, 4)
    # FIXME: Just to make sure that ... and then 1 is exterior
    foo = df.assemble(sum(df.Constant(1)*ds1(tag) for tag in dtags+iface_tags))
    bar = df.assemble(df.Constant(1)*ds1)
    assert abs(foo - bar) < 1E-13

    # and that 2 is interior ...
    foo = df.assemble(sum(df.Constant(1)*ds2(tag) for tag in iface_tags))
    bar = df.assemble(df.Constant(1)*ds2)
    assert abs(foo - bar) < 1E-13

    iface = EmbeddedMesh(boundaries2, iface_tags)
    dx_ = df.Measure('dx', domain=iface, subdomain_data=iface.marking_function)
    
    Tu1, Tu2 = (Trace(u, iface) for u in (u1, u2))
    Tv1, Tv2 = (Trace(u, iface) for u in (v1, v2))

    pdegree = V1.ufl_element().degree()
    gdim = mesh1.geometry().dim()
    gamma_DG = df.Constant(20*pdegree*gdim)
    # lhs
    a = block_form(W, 2)
    
    a[0][0] = df.inner(df.grad(u1), df.grad(v1))*df.dx + gamma*df.inner(Tu1, Tv1)*dx_
    a[0][1] = -gamma*df.inner(Tu2, Tv1)*dx_
    a[1][0] = -gamma*df.inner(Tu1, Tv2)*dx_
    a[1][1] = df.inner(df.grad(u2), df.grad(v2))*df.dx + gamma*df.inner(Tu2, Tv2)*dx_
    # Add the DG bits
    # For exterior domain
    n1, hF1 = df.FacetNormal(mesh1), df.CellDiameter(mesh1)
    a[0][0] += (
        -df.inner(df.dot(df.avg(df.grad(u1)), n1('+')), df.jump(v1))*df.dS
        -df.inner(df.dot(df.avg(df.grad(v1)), n1('+')), df.jump(u1))*df.dS
        + gamma_DG/df.avg(hF1)*df.inner(df.jump(u1), df.jump(v1))*df.dS
    )
    # It has Dirichlet bits
    a[0][0] += sum(-df.inner(df.dot(df.grad(u1), n1), v1)*ds1(tag)
                   -df.inner(df.dot(df.grad(v1), n1), u1)*ds1(tag)
                   + gamma_DG/hF1*df.inner(u1, v1)*ds1(tag) for tag in dtags)
        
    # For interior domain
    n2, hF2 = df.FacetNormal(mesh2), df.CellDiameter(mesh2)
    a[1][1] += (
        -df.inner(df.dot(df.avg(df.grad(u2)), n2('+')), df.jump(v2))*df.dS
        -df.inner(df.dot(df.avg(df.grad(v2)), n2('+')), df.jump(u2))*df.dS
        + gamma_DG/df.avg(hF2)*df.inner(df.jump(u2), df.jump(v2))*df.dS
    )


    # rhs
    L = block_form(W, 1)
    L[0] = (df.inner(f_vol[0], v1)*df.dx - 
            gamma*df.inner(ju, Tv1)*dx_ -
            sum(df.inner(g_robin[tag], Tv1)*dx_(tag) for tag in iface_tags))

    # Dirichlet contrib
    L[0] += sum(-df.inner(df.dot(df.grad(v1), n1), u_bdry[0])*ds1(tag)
                + gamma_DG/hF1*df.inner(u_bdry[0], v1)*ds1(tag)
                for tag in dtags)

    L[1] = (df.inner(f_vol[1], v2)*df.dx +
            gamma*df.inner(ju, Tv2)*dx_ + 
            sum(df.inner(g_robin[tag], Tv2)*dx_(tag) for tag in iface_tags))
    
    return a, L, None
    
    
def get_system_cg(u_prev, ju, boundaries, dt, data):
    '''Implicit discretization'''
    V1, V2 = (u_prev[0].function_space(), u_prev[1].function_space())
    W = [V1, V2]
    
    u1, u2 = map(df.TrialFunction, W)
    v1, v2 = map(df.TestFunction, W)

    dt = df.Constant(dt)
    gamma = df.Constant(1/dt)

    u_bdry, f_vol, g_robin = data['solution'], data['f'], data['robin']    

    boundaries1, boundaries2 = boundaries
    # 1 is EXTERIOR
    mesh1 = boundaries1.mesh()
    ds1 = df.Measure('ds', domain=mesh1, subdomain_data=boundaries1)

    mesh2 = boundaries2.mesh()
    ds2 = df.Measure('ds', domain=mesh2, subdomain_data=boundaries2)
    
    iface_tags = tuple(g_robin.keys())
    dtags = (1, 2, 3, 4)
    # FIXME: Just to make sure that ... and then 1 is exterior
    foo = df.assemble(sum(df.Constant(1)*ds1(tag) for tag in dtags+iface_tags))
    bar = df.assemble(df.Constant(1)*ds1)
    assert abs(foo - bar) < 1E-13

    # and that 2 is interior ...
    foo = df.assemble(sum(df.Constant(1)*ds2(tag) for tag in iface_tags))
    bar = df.assemble(df.Constant(1)*ds2)
    assert abs(foo - bar) < 1E-13

    iface = EmbeddedMesh(boundaries2, iface_tags)
    dx_ = df.Measure('dx', domain=iface, subdomain_data=iface.marking_function)
    
    Tu1, Tu2 = (Trace(u, iface) for u in (u1, u2))
    Tv1, Tv2 = (Trace(u, iface) for u in (v1, v2))

    # lhs
    a = block_form(W, 2)
    
    a[0][0] = df.inner(df.grad(u1), df.grad(v1))*df.dx + gamma*df.inner(Tu1, Tv1)*dx_
    a[0][1] = -gamma*df.inner(Tu2, Tv1)*dx_
    a[1][0] = -gamma*df.inner(Tu1, Tv2)*dx_
    a[1][1] = df.inner(df.grad(u2), df.grad(v2))*df.dx + gamma*df.inner(Tu2, Tv2)*dx_

    # rhs
    L = block_form(W, 1)
    L[0] = (df.inner(f_vol[0], v1)*df.dx - 
            gamma*df.inner(ju, Tv1)*dx_ -
            sum(df.inner(g_robin[tag], Tv1)*dx_(tag) for tag in iface_tags))

    L[1] = (df.inner(f_vol[1], v2)*df.dx +
            gamma*df.inner(ju, Tv2)*dx_ + 
            sum(df.inner(g_robin[tag], Tv2)*dx_(tag) for tag in iface_tags))
    
    V1_bcs = [df.DirichletBC(V1, u_bdry[0], boundaries1, tag) for tag in dtags]
    V2_bcs = []
    bcs = [V1_bcs, V2_bcs]

    return a, L, bcs
             
# --------------------------------------------------------------------

if __name__ == '__main__':
    from emi_utils import setup_mms, setup_geometry
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
    n = 16
    boundaries = setup_geometry(n)

    update_time(mms_data, time=0.)
    
    mesh1, mesh2 = (bdry.mesh() for bdry in boundaries)
    V1, V2 = (df.FunctionSpace(mesh, 'CG', pdegree) for mesh in (mesh1, mesh2))
    W = [V1, V2]
    
    u_prev = ii_Function([V1, V2])
    u_prev[0].vector()[:] = df.interpolate(mms_data['solution'][0], V1).vector()
    u_prev[1].vector()[:] = df.interpolate(mms_data['solution'][1], V2).vector()

    # Represent the jump at the interface
    iface = EmbeddedMesh(boundaries[1], (5, 6, 7, 8))
    dV = df.FunctionSpace(iface, V1.ufl_element().family(), pdegree)
    # u2 - u1
    ju = df.interpolate(u_prev[1], dV)
    ju.vector().axpy(-1, df.interpolate(u_prev[0], dV).vector())
    
    a, L, bcs = get_system(u_prev, ju, boundaries, dt, data=mms_data)

    is_CG = V1.ufl_element().family() == 'Lagrange'
    
    A, b = map(ii_assemble, (a, L))
    if is_CG:
        A, b, apply_bcs_b = apply_bc(A, b, bcs=bcs, return_apply_b=True)
    else:
        A, b = apply_bc(A, b, bcs=bcs, return_apply_b=False)
        apply_bcs_b = None
        
    A = ii_convert(A)
    print(f'|A| {A.norm("linf")}')
    
    solver = df.LUSolver(A, 'umfpack')
    x = u_prev.vector()
    
    step = 0
    t = step*dt
    while t < T_final:
        step += 1
        t = step*dt
        update_time(mms_data, t)
        
        b = ii_assemble(L)
        if apply_bcs_b is not None:
            b = apply_bcs_b(b)
        b = ii_convert(b)
        solver.solve(x, b)
        # Update jump
        # u2 - u1
        ju.assign(df.interpolate(u_prev[1], dV))
        ju.vector().axpy(-1, df.interpolate(u_prev[0], dV).vector())

        step % print_freq == 0 and print(f'\tt = {t:.2f} |b| = {b.norm("linf"):.4E}, |x| = {x.norm("linf"):.4E}')
    time = mms_data['solution'][0].time
    assert abs(time - mms_data['solution'][1].time) < 1E-13
    # NOTE: H1 here is questionable for DG but anyways
    error = df.errornorm(mms_data['solution'][0], u_prev[0], 'H1')**2
    error += df.errornorm(mms_data['solution'][1], u_prev[1], 'H1')**2
    error1 = df.sqrt(error)

    error = df.errornorm(mms_data['solution'][0], u_prev[0], 'L2')**2
    error += df.errornorm(mms_data['solution'][1], u_prev[1], 'L2')**2
    error0 = df.sqrt(error)
    
    ndofs = sum(Vi.dim() for Vi in W)
    mesh = boundaries[0].mesh()
    print(f'time = {time:.2E} h = {mesh.hmin():.2E} dt = {dt:.2E} => |u(T)-uh(T)|_1 = {error1:.4E} |u(T)-uh(T)|_0 = {error0:.4E} # = {ndofs}')
