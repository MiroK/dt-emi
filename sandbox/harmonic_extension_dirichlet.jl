using Gridap
using GridapGmsh
using Gridap.CellData 

model = GmshDiscreteModel("mesh.msh")

order = 1

Ω = Triangulation(model)

Γ = BoundaryTriangulation(model, tags=["Graph"])
n_Γ = get_normal_vector(Γ)

dΩ = Measure(Ω, order+1)
dΓ = Measure(Γ, order+1)
f(x) = 0

δV = TestFESpace(model,
                 ReferenceFE(lagrangian, Float64, order),
                 conformity=:H1, dirichlet_tags=["Graph"])
#=
For the harmonic extension from interior curve Γ we want to solve 
-Δ u = 0 in Ω
-∇u⋅n = 0 on ∂Ω
   u = g on Γ
=#
a(u,v) = ∫( ∇(v)⊙∇(u) )dΩ 
l(v) = ∫( v*f )dΩ  

# Data; we suppose that there is function defined just on Γ
U = TestFESpace(Γ,
                ReferenceFE(lagrangian, Float64, order),
                conformity=:H1) 
g(x) = x[1]*x[1] + 2*x[2]
gh = interpolate_everywhere(g, U)
gh_ = Interpolable(gh)

V = TrialFESpace(δV, [gh_])
op = AffineFEOperator(a, l, V, δV)
uh = solve(op)

writevtk(Ω, "results_dir", cellfields=["uh" => uh])