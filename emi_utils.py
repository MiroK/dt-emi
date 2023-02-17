from functools import reduce
import operator
import sympy as sp
import dolfin as df
import ulfy

from xii import *


def setup_geometry(n):
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

    mesh1, mesh2 = EmbeddedMesh(subdomains, 1), EmbeddedMesh(subdomains, 2)
    boundaries1 = mesh1.translate_markers(boundaries, (1, 2, 3, 4, 5, 6, 7, 8))
    boundaries2 = mesh2.translate_markers(boundaries, (5, 6, 7, 8))    

    boundaries = (boundaries1, boundaries2)
    
    return boundaries


def setup_mms(alpha_value=0):
    assert alpha_value >= 0
    # -Delta u = f
    # -grad(u).n = du/dt - g
    #
    mesh = df.UnitSquareMesh(2, 2)
    x, y = df.SpatialCoordinate(mesh)

    alpha, time = df.Constant(1), df.Constant(1)

    u_ = df.sin(2*df.pi*x)*df.sin(2*df.pi*y) # The thing that stays constant in time

    u2 = u_*(df.Constant(1) + df.exp(-alpha*time))
    u1 = u_

    ju = u2 - u1   # u_(1 + exp - 1) 
    dju_dt = ju*df.exp(-alpha*time)*(-alpha)

    sigma1 = -df.grad(u1)
    f1 = df.div(sigma1)

    sigma2 = -df.grad(u2)
    f2 = df.div(sigma2)
    
    normals = {5: df.Constant((-1, 0)),     # From 2 to 1
               6: df.Constant((1, 0)),
               7: df.Constant((0, -1)),
               8: df.Constant((0, 1))}

    g_u = lambda n2: dju_dt - df.dot(sigma2, n2)

    subs = {alpha: sp.Symbol('alpha'), time: sp.Symbol('time')}
    
    as_expr = lambda v: ulfy.Expression(v, subs=subs, degree=5, alpha=alpha_value, time=0)

    return {'solution': (as_expr(u1), as_expr(u2)),
            'f': (as_expr(f1), as_expr(f2)),
            'robin': {key: as_expr(g_u(normals[key])) for key in normals}}

# --------------------------------------------------------------------

if __name__ == '__main__':
    
    bdry1, bdry2 = setup_geometry(n=8)
    df.File('domain1.pvd') << bdry1
    df.File('domain2.pvd') << bdry2    
