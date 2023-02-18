using Gridap
using GridapGmsh

model = GmshDiscreteModel("mesh.msh")

order = 1
V = TestFESpace(model,
                ReferenceFE(lagrangian, Float64, order),
                conformity=:H1)

Ω = Triangulation(model)

Γ = BoundaryTriangulation(model, tags=["Graph"])
n_Γ = get_normal_vector(Γ)

dΩ = Measure(Ω, order+1)
dΓ = Measure(Γ, order+1)
f(x) = 0

#=
For the harmonic extension from interior curve Γ we want to solve 
-Δ u = 0 in Ω
-∇u⋅n = 0 on ∂Ω
   u = g on Γ
=#
a_Ω(u,v) = ∫( ∇(v)⊙∇(u) )dΩ 
l_Ω(v) = ∫( v*f )dΩ  

# Data; we suppose that there is function defined just on Γ
U = TestFESpace(Γ,
                ReferenceFE(lagrangian, Float64, order),
                conformity=:H1) 
g(x) = x[1]*x[1] + 2*x[2]
gh = interpolate_everywhere(g, U)

h_Γ = get_array(∫(1) * dΓ)
h = CellField(lazy_map(h -> h, h_Γ), Γ)
# Nitsche part that weakly set u = g on Γ
γ = 10*order*(order+1)
a_Γ(u,v) = ∫( - v*(∇(u)⋅n_Γ) - (∇(v)⋅n_Γ)*u + (γ/h)*v*u )dΓ
l_Γ(v)   = ∫(                - (∇(v)⋅n_Γ)*gh + (γ/h)*v*gh )dΓ

a(u,v) = a_Ω(u,v) + a_Γ(u,v)
l(v) = l_Ω(v) + l_Γ(v)

op = AffineFEOperator(a, l, V, V)
uh = solve(op)

writevtk(Ω, "results", cellfields=["uh" => uh])