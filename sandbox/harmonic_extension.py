# Can we do harmonic extension with Nitsche?
import dolfin as df

def get_system_dirichlet(V, boundaries, f):
    '''Implicit discretization'''
    u, v = df.TrialFunction(V), df.TestFunction(V)
    
    a = df.inner(df.grad(u), df.grad(v))*df.dx
    L = df.inner(df.Constant(0), v)*df.dx

    bcs = [df.DirichletBC(V, f, boundaries, tag) for tag in (5, 6, 7, 8)]

    return a, L, bcs


def get_system_nitsche(V, boundaries, f):
    '''Implicit discretization'''
    u, v = df.TrialFunction(V), df.TestFunction(V)
    
    a = df.inner(df.grad(u), df.grad(v))*df.dx
    L = df.inner(df.Constant(0), v)*df.dx

    # Nitsche part
    dS = df.Measure('dS', domain=mesh, subdomain_data=boundaries)
    n, hK = df.FacetNormal(mesh), df.CellDiameter(mesh)

    gamma = df.Constant(50)
    for tag in (5, 6, 7, 8):
        a += (-df.inner(df.avg(df.dot(df.grad(u), n)), df.avg(v))*dS(tag)
              -df.inner(df.avg(df.dot(df.grad(v), n)), df.avg(u))*dS(tag)
              +(gamma/df.avg(hK))*df.inner(df.avg(v), df.avg(u))*dS(tag))

        L += (-df.inner(df.avg(df.dot(df.grad(v), n)), f)*dS(tag)
              +(gamma/df.avg(hK))*df.inner(df.avg(v), f)*dS(tag))

    return a, L, None

# --------------------------------------------------------------------

if __name__ == '__main__':
    from emi_utils import setup_mms, setup_geometry
    from utils import update_time
    from xii import *

    f = df.Expression('x[0]*x[0] - 2*x[1]', degree=2)
    for k in range(2, 6):
        n = 2**k

        boundaries = setup_geometry(n, monolithic=True)

        mesh = boundaries.mesh()
        V = df.FunctionSpace(mesh, 'CG', 1)
    
        a, L, bcs = get_system_dirichlet(V, boundaries, f)
        uh = df.Function(V)
        
        A, b = df.assemble_system(a, L, bcs)
        df.solve(A, uh.vector(), b)
        # ----
        a, L, bcs = get_system_nitsche(V, boundaries, f)
        vh = df.Function(V)
        
        A, b = df.assemble_system(a, L, bcs)
        df.solve(A, vh.vector(), b)

        e = df.Function(V)
        e.vector().axpy(1, uh.vector())
        e.vector().axpy(-1, vh.vector())
        print(e.vector().norm('linf'))
        
    df.File('dir.pvd') << uh
    df.File('nit.pvd') << vh
