from dolfin import *


parameters['ghost_mode'] = 'shared_vertex'


def CellCenterDistance(mesh):
    '''Discontinuous Lagrange Trace function that holds the cell-to-cell distance'''
    # Cell-cell distance for the interior facet is defined as a distance 
    # of midpoints of the cells that share the facet. For exterior facet
    # we take the distance of cell midpoint and the facet midpoint
    Q = FunctionSpace(mesh, 'DG', 0)
    V = FunctionSpace(mesh, 'CG', 1)
    L = FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)

    gdim = mesh.geometry().dim()

    cK, fK = CellVolume(mesh), FacetArea(mesh)
    q, l = TestFunction(Q), TestFunction(L)
    # The idea here to first assemble component by component the cell 
    # and (exterior) facet midpoint
    cell_centers, facet_centers = [], []
    for xi in SpatialCoordinate(mesh):
        qi = Function(Q)
        # Pretty much use the definition that a midpoint is int_{cell} x_i/vol(cell)
        # It's thanks to eval in q that we take values :)
        assemble((1/cK)*inner(xi, q)*dx, tensor=qi.vector())
        cell_centers.append(qi)
        # Same here but now our mean is over an edge
        li = Function(L)
        assemble((1/fK)*inner(xi, l)*ds, tensor=li.vector())
        facet_centers.append(li)
    # We build components to vectors
    cell_centers, facet_centers = map(as_vector, (cell_centers, facet_centers))

    distances = Function(L)
    # FIXME: This might not be necessary but it's better to be certain
    dS_, ds_ = dS(metadata={'quadrature_degree': 0}), ds(metadata={'quadrature_degree': 0})
    # Finally we assemble magniture of the vector that is determined by the
    # two centers
    assemble(((1/fK('+'))*inner(sqrt(dot(jump(cell_centers), jump(cell_centers))), l('+'))*dS_+
              (1/fK)*inner(sqrt(dot(cell_centers-facet_centers, cell_centers-facet_centers)), l)*ds_),
             distances.vector())

    return distances
