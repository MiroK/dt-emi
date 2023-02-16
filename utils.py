from functools import reduce
import operator
import sympy as sp
import dolfin as df
import ulfy


def update_time(data, time):
    '''Set time to time'''
    for thing in data.values():
        if isinstance(thing, dict):
            things = thing.values()
        else:
            things = (thing, )
            
        for t in things:
            hasattr(t, 'time') and setattr(t, 'time', time)
    return data


def check_bdry_tags(tags):
    keys = tuple(tags.keys())
    expected = ('neumann', 'dirichlet', 'robin')
    for k in expected:
        if k not in keys:
            tags[k] = ()
    assert tags.keys() == set(expected)
    
    for key in tuple(tags.keys()):
        value = tags[key]
        if isinstance(value, int):
            tags[key] = (value, )
        assert isinstance(tags[key], tuple)

    assert reduce(operator.or_, map(set, tags.values())) == set((1, 2, 3, 4)) 
        
    return True


def setup_geometry(n):
    # Square edge marking assumed is
    #    4
    #  1   2
    #    3
    mesh = df.UnitSquareMesh(n, n, 'crossed')
    boundaries = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    df.CompiledSubDomain('near(x[0], 0)').mark(boundaries, 1)
    df.CompiledSubDomain('near(x[0], 1)').mark(boundaries, 2)
    df.CompiledSubDomain('near(x[1], 0)').mark(boundaries, 3)
    df.CompiledSubDomain('near(x[1], 1)').mark(boundaries, 4)

    return boundaries


def setup_mms(alpha_value=0):
    assert alpha_value >= 0
    # -Delta u = f
    # -grad(u).n = du/dt - g
    #
    mesh = df.UnitSquareMesh(2, 2)
    x, y = df.SpatialCoordinate(mesh)

    alpha, time = df.Constant(1), df.Constant(1)

    u_ = df.sin(df.pi*(x-y)) # The thing that stays constant in time
    u = u_*df.exp(-alpha*time)
    du_dt = u_*df.exp(-alpha*time)*(-alpha)

    sigma = -df.grad(u)
    f = df.div(sigma)

    normals = {1: df.Constant((-1, 0)),
               2: df.Constant((1, 0)),
               3: df.Constant((0, -1)),
               4: df.Constant((0, 1))}

    g_u = lambda n: du_dt - df.dot(sigma, n)

    subs = {alpha: sp.Symbol('alpha'), time: sp.Symbol('time')}
    
    as_expr = lambda v: ulfy.Expression(v, subs=subs, degree=5, alpha=alpha_value, time=0)

    return {'solution': as_expr(u),
            'f': as_expr(f),
            'flux': as_expr(sigma),
            'robin': {key: as_expr(g_u(normals[key])) for key in normals}}
