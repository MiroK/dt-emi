# Stability of EMI time discretization

## Dependencies
- `FEniCS >= 2019.1.0`
- [`xii`](https://github.com/MiroK/fenics_ii) and it's dependencies
- [`ulfy`](https://github.com/MiroK/ulfy)

## TODO
- [x] `solve_lagrange.py` is just simple Poisson with time dep. Robin bcs and Langrange elements
- [x] `solve_emi_lagrange*.py` solves the time-dependent EMI problem (CG or DG or DG0)
----------------------------------------------------------------------
- [ ] drop assumption of unit diffusion parameters
- [ ] stationary case (for error estimates in $L^2$)