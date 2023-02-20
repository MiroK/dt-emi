import dolfin as df

code = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Cell.h>

class PCWS : public dolfin::Expression
{
public:

  PCWS() : dolfin::Expression() {}

  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& c) const override
  {
    const uint cell_index = c.index;

    const std::size_t value = (*subdomains)[c.index];
    values[0] = value;
  }

  std::shared_ptr<dolfin::MeshFunction<std::size_t>> subdomains;

};

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<PCWS, std::shared_ptr<PCWS>, dolfin::Expression>
    (m, "PCWS")
    .def(py::init<>())
    .def_readwrite("subdomains", &PCWS::subdomains);
}
"""

aux = df.compile_cpp_code(code)


def subdomain_marking_foo(subdomains, V=None, aux=aux):
    '''Function in P0 space with cell values given by tags of the cell'''
    mesh = subdomains.mesh()
    assert mesh.topology().dim() == subdomains.dim()
    # As Expression
    f = df.CompiledExpression(aux.PCWS(), subdomains=subdomains, degree=0)

    if V is not None:
        assert V.ufl_element().value_size() == 1
        assert V.ufl_element().family() == 'Discontinuous Lagrange'

        return df.interpolate(f, V)

    V = df.FunctionSpace(mesh, 'DG', 0)
    return df.interpolate(f, V)
