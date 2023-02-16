from utils import check_bdry_tags
import dolfin as df


def get_system(u_prev, boundaries, dt, data, bdry_tags):
    '''Dispatch'''
    V = u_prev.function_space()
    elm = V.ufl_element()

    if elm.family() == 'Lagrange':
        return get_system_cg(u_prev, boundaries, dt, data, bdry_tags)

    
def get_system_cg(u_prev, boundaries, dt, data, bdry_tags):
    '''Implicit discretization'''
    assert check_bdry_tags(bdry_tags)
    dtags, ntags, rtags = (bdry_tags[k] for k in ('dirichlet', 'neumann', 'robin'))

    V = u_prev.function_space()
    
    mesh = boundaries.mesh()
    u, v = df.TrialFunction(V), df.TestFunction(V)

    dt = df.Constant(dt)
    gamma = df.Constant(1/dt)
    ds = df.Measure('ds', domain=mesh, subdomain_data=boundaries)

    a = df.inner(df.grad(u), df.grad(v))*df.dx + sum(gamma*df.inner(u, v)*ds(tag) for tag in rtags)

    u_bdry, f_vol, sigma, g_robin = data['solution'], data['f'], data['flux'], data['robin']
    
    L = (df.inner(f_vol, v)*df.dx +
         sum(gamma*df.inner(u_prev, v)*ds(tag) for tag in rtags) +
         sum(df.inner(g_robin[tag], v)*ds(tag) for tag in rtags))
    # Handle Neumann contrib
    n = df.FacetNormal(mesh)
    L += sum(-df.inner(df.dot(sigma, n), v)*ds(tag) for tag in ntags)

    bcs = [df.DirichletBC(V, u_bdry, boundaries, tag) for tag in dtags]

    return a, L, bcs
             
# --------------------------------------------------------------------

if __name__ == '__main__':
    from utils import setup_geometry, setup_mms, update_time

    T_final = 0.5
    dt = 1E-2
    alpha = 1E-3

    mms_data = setup_mms(alpha_value=alpha)

    tags = {'dirichlet': (),
            'robin': (1, 2, 3, 4),
            'neumann': ()}

    print_freq = 0.1*T_final/dt
    
    # -----------
    n = 64
    boundaries = setup_geometry(n)

    update_time(mms_data, time=0.)
    
    mesh = boundaries.mesh()
    V = df.FunctionSpace(mesh, 'CG', 1)
    u_prev = df.interpolate(mms_data['solution'], V)
    
    a, L, bcs = get_system(u_prev, boundaries, dt, data=mms_data, bdry_tags=tags)
    
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

        step % print_freq == 0 and print(f'\tt = {t:.2f} |b| = {b.norm("linf"):.4E}, |x| = {x.norm("linf"):.4E}')
    time = mms_data['solution'].time
    error = df.errornorm(mms_data['solution'], u_prev, 'H1')
    print(f'time = {time:.2f} h = {mesh.hmin():.2f} dt = {dt:.2f} => |u(T)-uh(T)|_1 = {error:.4E}')
