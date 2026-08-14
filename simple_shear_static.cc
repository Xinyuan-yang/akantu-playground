// Velocity boundary condition code //

#include <map>
#include <cmath>
#include <iostream>
#include <sstream>
#include <ostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <iomanip>

#include "dumpable_iohelper.hh"
#include "dumpable_inline_impl.hh"
#include "dumper_paraview.hh"

#include "dumper_text.hh"
#include "dumper_variable.hh"
#include "solid_mechanics_model.hh"
//#include "ntn_contact_solver.hh"
#include "mesh_partition_mesh_data.hh"

#include "non_linear_solver.hh"

#include "aka_common.hh"
#include "mesh_utils.hh"
#include "ntn_base_contact.hh"
#include "ntn_base_friction.hh"
#include "ntn_contact.hh"
#include "ntn_contact_solvercallback.hh"
#include "ntn_initiation_function.hh"

// for k-d tree
#include <limits>
#include <vector>

#ifdef AKANTU_USE_QVIEW
#include <libqview.h>
#endif

using namespace akantu;

//Write the results file
void writeDisplacementXY(const Array<Real> & displacement,
                         const std::string & filename) {
  std::ofstream out(filename);
  if (!out) {
    AKANTU_EXCEPTION("Cannot write static displacement file: " << filename);
  }

  out << std::scientific << std::setprecision(17);

  for (Idx node = 0; node < displacement.size(); ++node) {
    out << displacement(node, _x) << ' '
        << displacement(node, _y) << '\n';
  }
}

/// Affine displacement field for uniaxial plane-stress loading in y.  The
/// reference is the bottom face, so u_y is exactly zero there.  Centering the
/// lateral Poisson displacement about x_mid prevents a rigid x translation.
class UniformCompressionPS : public BC::Dirichlet::DirichletFunctor {
public:
  UniformCompressionPS(Real normal_strain, Real lateral_strain, Real x_mid,
                       Real y_bottom, Real mu, Real e12)
      : normal_strain(normal_strain), lateral_strain(lateral_strain),
        x_mid(x_mid), y_bottom(y_bottom), mu(mu), e12(e12) {}

  void operator()(Idx, VectorProxy<bool> & flags,
                  VectorProxy<Real> & displacement,
                  const VectorProxy<const Real> & position) override {
    flags(_x) = true;
    flags(_y) = true;

    displacement(_x) = lateral_strain * (position(_x) - x_mid)  +
                      mu * e12 * position(_y) * 0.95 ;
    displacement(_y) = normal_strain * (position(_y) - y_bottom);
  }

private:
  Real normal_strain;
  Real lateral_strain;
  Real x_mid;
  Real y_bottom;
  Real mu;
  Real e12;
};

class UniformCompression : public BC::Dirichlet::DirichletFunctor {
public:
  UniformCompression(Real normal_strain, Real lateral_strain, Real x_mid,
                       Real y_bottom)
      : normal_strain(normal_strain), lateral_strain(lateral_strain),
        x_mid(x_mid), y_bottom(y_bottom) {}

  void operator()(Idx, VectorProxy<bool> & flags,
                  VectorProxy<Real> & displacement,
                  const VectorProxy<const Real> & position) override {
    flags(_x) = true;
    flags(_y) = true;

    displacement(_x) = lateral_strain * (position(_x) - x_mid);
    displacement(_y) = normal_strain * (position(_y) - y_bottom);
  }

private:
  Real normal_strain;
  Real lateral_strain;
  Real x_mid;
  Real y_bottom;
};

/* ------------------------------------------------------------------------ */
/* Main                                                                     */
/* ------------------------------------------------------------------------ */
int main(int argc, char *argv[]) {

  if (argc != 3)
  {
    std::cerr << "Usage: " << argv[0]
              << " <input_file> <mesh_size>" << std::endl;
    return EXIT_FAILURE;
  }

  const auto & comm = Communicator::getStaticCommunicator();
  auto prank = comm.whoAmI();
  auto psize = comm.getNbProc();
  AKANTU_DEBUG_INFO("Running on " << psize << " processors");
  AKANTU_DEBUG_INFO("This is processor " << prank);

  std::cout << "Input: " << argv[1] << std::endl;
  std::cout << "Mesh size: " << argv[2] << std::endl;

  getStaticParser().parse(argv[1]);

  const ParserSection & data = getUserParser();
  UInt spatial_dimension = data.getParameter("spatial_dimension");
  std::string simulation_name = data.getParameter("simulation_name");

  // Compression is negative in the y direction.
  Real normal_stress = data.getParameter("normal_stress");

  /// Static solution
  std::unique_ptr<Mesh> mesh_static;
  std::unique_ptr<SolidMechanicsModel> model_static;
  mesh_static = std::make_unique<Mesh>(spatial_dimension);
  std::string mesh_file = "ntn_static_" + std::string(argv[2]) + ".msh";
  std::string output_name = "ntn_static" + std::string(argv[2]);
  std::string txt_name  = "ntn_static" + std::string(argv[2]) + ".txt";
  mesh_static->read(mesh_file);

  model_static = std::make_unique<SolidMechanicsModel>(*mesh_static);

  model_static->initFull(_analysis_method = _static);
  
  model_static->setBaseName(output_name);
  model_static->addDumpFieldVector("displacement");
  model_static->addDumpFieldVector("external_force");
  model_static->addDumpField("strain");
  model_static->addDumpField("stress");
  model_static->addDumpField("blocked_dofs");
  // Analytical plane-stress response to sigma_yy = normal_stress and
  // sigma_xx = 0:
  //   epsilon_yy = sigma_yy / E, epsilon_xx = -nu sigma_yy / E.
  // Thus u_x varies linearly along each edge because of the Poisson effect;
  // u_y is zero on slider_bottom and has the correct shortening on slider_top.
  const auto & material = model_static->getMaterial("slider");
  const Real E = material.getParam("E");
  const Real nu = material.getParam("nu");
  const Real normal_strain = normal_stress / E;
  const Real lateral_strain = -nu * normal_strain;
  const Real shear_modulus = E / (2.0 * (1.0 + nu));
  const Real e12 =  -normal_stress / shear_modulus ;
  const Real mu = 0.1;


  const auto & lower = mesh_static->getLowerBounds();
  const auto & upper = mesh_static->getUpperBounds();
  const Real x_mid = 0.5 * (lower(_x) + upper(_x));

  UniformCompressionPS imposed_compression_ps(normal_strain, lateral_strain, x_mid,
                                           lower(_y), mu, e12);
  UniformCompression imposed_compression(normal_strain, lateral_strain, x_mid,
                                           lower(_y));
  model_static->applyBC(imposed_compression, "slider_bottom");
  model_static->applyBC(imposed_compression_ps, "slider_top");
  auto && solver_static = model_static->getNonLinearSolver();
  solver_static.set("max_iterations", 10);
  solver_static.set("threshold", 1e-10);
  solver_static.set("convergence_type", SolveConvergenceCriteria::_residual);

  model_static->dump();
  model_static->solveStep();
  model_static->dump();

  writeDisplacementXY(
    model_static->getDisplacement(),
    txt_name);

  std::cout << "Displacement-controlled static solve completed." << std::endl;

  return EXIT_SUCCESS;
}
