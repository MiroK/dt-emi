import gmsh, sys

gmsh.initialize(sys.argv)

model = gmsh.model
fac = model.occ

# outer
points = [[0, 0], [1, 0], [1, 1], [0, 1]]
outer_points = [fac.addPoint(*p, z=0) for p in points]
n = len(outer_points)
outer_lines = [fac.addLine(outer_points[i], outer_points[(i+1)%n]) for i in range(n)]
outer_loop = fac.addCurveLoop(outer_lines)

# Inner
points = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]
inner_points = [fac.addPoint(*p, z=0) for p in points]
n = len(inner_points)
inner_lines = [fac.addLine(inner_points[i], inner_points[(i+1)%n]) for i in range(n)]
inner_loop = fac.addCurveLoop(inner_lines)

inner_surface = fac.addPlaneSurface([inner_loop])
outer_surface = fac.addPlaneSurface([inner_loop, outer_loop])

fac.synchronize()

model.addPhysicalGroup(2, [outer_surface], 2)
model.addPhysicalGroup(2, [inner_surface], 1)
model.addPhysicalGroup(1, inner_lines, 1, name="Graph")
fac.synchronize()

# gmsh.fltk.initialize()
# gmsh.fltk.run()

gmsh.option.setNumber('Mesh.MeshSizeFactor', 0.1)
model.mesh.generate(2)

gmsh.write('mesh.msh')

gmsh.finalize()
