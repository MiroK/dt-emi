from subdomain_marking import subdomain_marking_foo
import dolfin as df
import numpy as np

df.parameters['ghost_mode'] = 'shared_facet'


def setup_geometry(n, monolithic=False):
    # Square in square
    mesh = df.UnitSquareMesh(n, n, 'crossed')

    subdomains = df.MeshFunction('size_t', mesh, mesh.topology().dim(), 1)
    subd2 = df.CompiledSubDomain('(std::fabs(x[0] - 0.5) < 0.25+TOL) && (std::fabs(x[1] - 0.5) < 0.25+TOL)', TOL=1E-10)
    subd2.mark(subdomains, 2)
    
    boundaries = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    # Outer
    df.CompiledSubDomain('near(x[0], 0)').mark(boundaries, 1)
    df.CompiledSubDomain('near(x[0], 1)').mark(boundaries, 2)
    df.CompiledSubDomain('near(x[1], 0)').mark(boundaries, 3)
    df.CompiledSubDomain('near(x[1], 1)').mark(boundaries, 4)
    # Inner
    df.CompiledSubDomain('near(x[0], 0.25) && ((0.25-TOL < x[1]) && (x[1] < 0.75+TOL))', TOL=1E-10).mark(boundaries, 5)
    df.CompiledSubDomain('near(x[0], 0.75) && ((0.25-TOL < x[1]) && (x[1] < 0.75+TOL))', TOL=1E-10).mark(boundaries, 6)
    df.CompiledSubDomain('near(x[1], 0.25) && ((0.25-TOL < x[0]) && (x[0] < 0.75+TOL))', TOL=1E-10).mark(boundaries, 7)
    df.CompiledSubDomain('near(x[1], 0.75) && ((0.25-TOL < x[0]) && (x[0] < 0.75+TOL))', TOL=1E-10).mark(boundaries, 8)    

    return subdomains, boundaries


def patch_interpolate(V, subdomains, f):
    # FIXME: make this work in parallel
    mesh = V.mesh()
    assert mesh.geometry().dim() == subdomains.dim()
    dm = V.dofmap()

    target = df.Function(V)
    target_ = target.vector().get_local()

    masked = np.zeros(len(target_), dtype=bool)
    
    subdomains = subdomains.array()
    for tag, foo in f.items():
        source_ = df.interpolate(foo, V).vector().get_local()
        dofs = np.concatenate([dm.cell_dofs(c) for c in np.where(subdomains == tag)[0]])
        target_[dofs] = source_[dofs]

        assert not np.any(masked[dofs])
        
        masked[dofs] = True
    target.vector().set_local(target_)

    return target


def pcws_constant_project(f, V, fV=None, dM=None, tags=None):
    '''Project f onto V where V is some piecewise-constant space'''
    v = df.TestFunction(V)
    assert v.ufl_shape == f.ufl_shape
    
    mesh = V.mesh()
    hV, hA = df.CellVolume(mesh), df.FacetArea(mesh)
    # Might be useful to reuse the function for projecting
    if fV is None:
        fV = df.Function(V)
    x = fV.vector()

    elm = V.ufl_element()
    
    if dM is None:
        dM = {'Discontinuous Lagrange': df.dx, 'HDiv Trace': df.dS}[elm.family()]

    # Normally we would assemble linear system (A, b) and solve for x being
    # the coefficient vector of fV. Here we want to directly assemble the action
    # of inv(A) onto b. This is possible since A is diagonal
    if elm.degree() == 0:
        if tags is None:
            projection_forms = {
                'Discontinuous Lagrange': lambda f: (1/hV)*df.inner(f, v)*dM,
                'HDiv Trace': lambda f: (1/df.avg(hA))*df.inner(f, v('+'))*dM
            }
        else:
            projection_forms = {
                'Discontinuous Lagrange': lambda f: sum((1/hV)*df.inner(f, v)*dM(tag) for tag in tags),
                'HDiv Trace': lambda f: sum((1/df.avg(hA))*df.inner(f, v('+'))*dM(tag) for tag in tags)
            }
        form = projection_forms[elm.family()](f)
        # Assemble action into x
        df.assemble(form, x)
    else:
        if tags is None:
            rhs_forms = {
                'Discontinuous Lagrange': lambda f: df.inner(f, v)*dM,
                'HDiv Trace': lambda f: df.inner(f, v('+'))*dM
            }
        else:
            rhs_forms = {
                'Discontinuous Lagrange': lambda f: sum(df.inner(f, v)*dM(tag) for tag in tags),
                'HDiv Trace': lambda f: sum(df.inner(f, v('+'))*dM(tag) for tag in tags)
            }
        rhs_form = rhs_forms[elm.family()](f)

        u = df.TrialFunction(V)
        lhs_form = {
            'Discontinuous Lagrange': lambda f: df.inner(f, v)*df.dx,
            'HDiv Trace': lambda f: df.inner(f('+'), v('+'))*df.dS + df.inner(f, v)*df.ds
        }[elm.family()](u)

        A, b = map(df.assemble, (lhs_form, rhs_form))
        # FIXME: this should be cached
        solver = df.PETScKrylovSolver('cg', 'hypre_amg')
        solver.parameters['relative_tolerance'] = 1E-30
        solver.parameters['absolute_tolerance'] = 1E-14
        # solver.parameters['monitor_convergence'] = True        
        solver.set_operators(A, A)
        
        solver.solve(x, b)
        
    return fV


def interface_normal(subdomains):
    '''
    Computes a [DLT]^d function representing the facet normal vector
    such that on the interface between the subdomains it points from the
    higher (tag) value to a lower value.
    '''
    # Represent cell tags as P0 function so that we can query
    chi = subdomain_marking_foo(subdomains)

    mesh = subdomains.mesh()
    # Normal computation
    V = df.VectorFunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    v = df.TestFunction(V)

    n, hA = df.FacetNormal(mesh), df.FacetArea(mesh)
    n_ = df.Function(V)  # Our oriented normal
    # FEniCS will orient the normal form '+' to '-'. This is what we want
    # if the plus side is greate than minus side
    switch = df.conditional(df.ge(chi('+'), chi('-')), n('+'), n('-'))
    # Here we do L^2 projection to get the normal
    df.assemble((1/df.avg(hA))*df.inner(switch, v('+'))*df.dS + (1/hA)*df.inner(n, v)*df.ds,
                n_.vector())

    return n_


def plus(phi, normal):
    '''Restriction of phi to the cell from which the normal originates'''
    n = df.FacetNormal(normal.function_space().mesh())    
    switch = df.conditional(df.ge(df.dot(normal('+'), n('+')), df.Constant(0)), phi('+'), phi('-'))
    return switch


def minus(phi, normal):
    '''Restriction of phi to the cell at which the normal ends'''
    n = df.FacetNormal(normal.function_space().mesh())
    switch = df.conditional(df.ge(df.dot(normal('+'), n('+')), df.Constant(0)), phi('-'), phi('+'))    
    return switch

