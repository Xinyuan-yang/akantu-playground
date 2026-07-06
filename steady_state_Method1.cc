// Velocity boundary condition code //

#include <cmath>
#include <filesystem>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

#include "dumpable_iohelper.hh"

#include "dumper_text.hh"
#include "dumper_variable.hh"
#include "solid_mechanics_model.hh"

#include "aka_common.hh"
#include "mesh_utils.hh"
#include "ntn_base_contact.hh"
#include "ntn_contact_solvercallback.hh"
#include "ntn_initiation_function.hh"

using namespace akantu;

/* ------------------------------------------------------------------------ */
/* Main                                                                     */
/* ------------------------------------------------------------------------ */
int main(int argc, char *argv[])
{

  if (argc != 4)
  {
    std::cerr << "Usage: " << argv[0]
              << " <coulomb-mu> <nb-it-nodes> <damping: n|s|l>" << std::endl;
    return EXIT_FAILURE;
  }

  const std::string input_file = "ras_ss_coulomb_ve.in";
  const std::string coulomb_mu_text = argv[1];
  const Real coulomb_mu = std::stod(coulomb_mu_text);
  const UInt nb_it_nodes = std::stoul(argv[2]);
  const std::string damping_mode = argv[3];
  getStaticParser().parse(input_file);
  const ParserSection &data = getUserParser();
  std::string output_folder =
      "steady_state_M1_" + coulomb_mu_text + "_" + std::to_string(nb_it_nodes) + "_" + damping_mode;
  UInt spatial_dimension = data.getParameter("spatial_dimension");
  std::unique_ptr<Mesh> mesh;
  std::unique_ptr<SolidMechanicsModel> model;
  std::unique_ptr<NTNContactSolverCallback> solver_ntn;
  mesh = std::make_unique<Mesh>(spatial_dimension);
  const std::string mesh_file =
      "ntn_test_" + std::to_string(nb_it_nodes) + ".msh";
  mesh->read(mesh_file);

  // Periodic BC switch here
  // mesh->makePeriodic(_x, "slider_left", "slider_right");
  // mesh->makePeriodic(_x, "base_left", "base_right");

  model = std::make_unique<SolidMechanicsModel>(*mesh);

  Real time_step_factor = data.getParameter("time_step_factor");

  Int normal_dir = 1;

  solver_ntn = std::make_unique<NTNContactSolverCallback>(
      *model, "slider_bottom", "base_top", normal_dir, time_step_factor);

  auto &mat = model->getMaterial("slider");
  auto &base_mat = model->getMaterial("base");

  if (damping_mode == "n")
  {
    mat.set("Eta", 0.);
    base_mat.set("Eta", 0.);
    std::cout << "Viscoelastic Eta multiplier = 0: slider Eta = 0, base Eta = 0"
              << std::endl;
  }
  else if (damping_mode == "l")
  {
    Real slider_eta = mat.getParam("Eta");
    Real base_eta = base_mat.getParam("Eta");
    mat.set("Eta", 50. * slider_eta);
    base_mat.set("Eta", 50. * base_eta);
    std::cout << "Viscoelastic Eta multiplier = 50: slider Eta = "
              << 50. * slider_eta << ", base Eta = " << 50. * base_eta
              << std::endl;
  }
  else
  {
    std::cout << "Viscoelastic Eta multiplier = 1" << std::endl;
  }

  Real cp = mat.getPushWaveSpeed(ElementNull);
  Real cs = mat.getShearWaveSpeed(ElementNull);

  std::cout << "P-wave speed = " << cp << std::endl;
  std::cout << "S-wave speed = " << cs << std::endl;

  Real shear_vel = data.getParameter("shear_velocity");
  Vector<Real> trac_top = data.getParameter("top_traction");
  Vector<Real> trac_bottom = data.getParameter("bot_traction");

  model->setBaseName(output_folder);
  model->addDumpField("blocked_dofs");
  model->addDumpField("mass");
  model->addDumpFieldVector("velocity");
  model->addDumpFieldVector("acceleration");
  model->addDumpFieldVector("displacement");
  model->addDumpFieldVector("internal_force");
  model->addDumpFieldVector("external_force");

  // Static analytical solution
  Real fss = coulomb_mu;
  Real E = mat.getParam("E");
  Real nu = mat.getParam("nu");
  Real shear_modulus = E / (2. * (1. + nu));
  Real normal_strain_applied = trac_top(1) / E - nu * nu * trac_top(1) / E;

  Array<Real> &displacement = model->getDisplacement();
  Array<Real> &position = mesh->getNodes();
  UInt nb_nodes = model->getFEEngine().getMesh().getNbNodes();

  Real t_fin = 0.5 / cs;

  // Steady state initialization
  for (UInt n = 0; n < nb_nodes; ++n)
  {
    displacement(n, 0) = fss * -trac_top(1) / (shear_modulus)*position(n, 1);
    displacement(n, 1) = normal_strain_applied * position(n, 1);
  }

  // Set boundary conditions for dynamic simulation
  model->applyBC(BC::Neumann::FromTraction(trac_top), "slider_top");
  model->applyBC(BC::Neumann::FromTraction(trac_bottom), "base_bottom");

  ///// Set to steady state
  const auto &slider_nodes =
      mesh->getElementGroup("slider").getNodeGroup().getNodes();
  const auto &base_nodes =
      mesh->getElementGroup("base").getNodeGroup().getNodes();

  // Specify initial nodal velocity
  auto &velo = model->getVelocity();
  auto &increment = model->getIncrement();
  auto friction = solver_ntn->getFriction();
  friction->set("mu", coulomb_mu);
  auto dt = model->getTimeStep();

  for (auto n : slider_nodes)
  {
    velo(n, _x) = 0.5 * shear_vel;
    increment(n, _x) = 0.5 * shear_vel * dt;
  }
  for (auto n : base_nodes)
  {
    velo(n, _x) = -0.5 * shear_vel;
    increment(n, _x) = -0.5 * shear_vel * dt;
  }

  auto contact = solver_ntn->getContact();

  auto &slip_velocity = friction->getSlipVelocity();
  auto &slip_velocity_norm = friction->getSlipVelocityNorm();
  auto &is_sticking = friction->getIsSticking();
  // Real phi = friction->get("friction_state");  // Turn on if rate and state

  for (auto &&[n, master, slave, slip_vel, slip_vel_n, is_sticking] :
       enumerate(contact->getMasters(), contact->getSlaves(),
                 make_view(slip_velocity, slip_velocity.getNbComponent()),
                 slip_velocity_norm, is_sticking))
  {
    is_sticking = false;
    // friction->updateFrictionState(n, phi);
  }
  //////

  // Time of the simulation
  Real stable_time_step = model->getStableTimeStep();
  Real time_step = stable_time_step * time_step_factor;
  model->setTimeStep(time_step);
  UInt nb_steps = t_fin / time_step;
  UInt dump_every = nb_steps / 200;

  std::cout << "Time step = " << time_step << std::endl;
  std::cout << "Number of steps = " << nb_steps << std::endl;
  std::cout << "Dump every = " << dump_every << std::endl;

  Real alpha = 0; // mass proportional damping

  if (damping_mode == "n")
  {
    alpha = 0;
  }
  else if (damping_mode == "s")
  {
    alpha = 40;
  }
  else if (damping_mode == "l")
  {
    alpha = 40;
  }
  else
  {
    std::cerr << "Unknown damping mode '" << damping_mode
              << "'. Use n, s, or l." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Damping mode " << damping_mode << ": alpha = " << alpha
            << std::endl;

  model->assembleMass();
  auto &M = model->getDOFManager().getMatrix("M");

  auto &C = model->getDOFManager().getNewMatrix("C", "M");
  C.zero();
  C.add(M, alpha);
  std::cout << "has C = " << model->getDOFManager().hasMatrix("C") << std::endl;

  model->dump();

  std::ofstream energies;
  auto file_name = std::filesystem::path("friction-energies-" + output_folder + ".csv");
  energies.open(file_name.c_str(), std::ofstream::out | std::ofstream::trunc);

  energies << "time,ekin,epot,work,econ,efri,tot" << std::endl;

  auto einit = 0.;

  std::cout << "Starting simulation..." << std::endl;

  for (UInt s = 0; s < nb_steps; ++s)
  {
    // Apply velocity
    UInt nb_nodes = model->getFEEngine().getMesh().getNbNodes();
    Array<Real> &position = mesh->getNodes();
    Array<Real> &velo = model->getVelocity();
    const Vector<Real> &upperBounds = mesh->getUpperBounds();
    const Vector<Real> &lowerBounds = mesh->getLowerBounds();
    Real top = upperBounds(1);
    Real bottom = lowerBounds(1);
    Real stable_time_step = model->getStableTimeStep();
    Real time_step = stable_time_step * time_step_factor;
    Real disp_incr = shear_vel * time_step;
    Array<Real> &displacement = model->getDisplacement();
    Array<bool> &blocked = model->getBlockedDOFs();
    for (UInt n = 0; n < nb_nodes; ++n)
    {
      if (std::abs(position(n, 1) - top) < 1e-6)
      {
        velo(n, _x) = 0.5 * shear_vel;
      }
      if (std::abs(position(n, 1) - bottom) < 1e-6)
      {
        velo(n, _x) = -0.5 * shear_vel;
      }
    }

    for (UInt n = 0; n < nb_nodes; ++n)
    {
      if (std::abs(position(n, 1) - top) < 1e-6)
      {
        displacement(n, 0) += 0.5 * disp_incr;
        blocked(n, 0) = true;
      }
      if (std::abs(position(n, 1) - bottom) < 1e-6)
      {
        displacement(n, 0) += -0.5 * disp_incr;
        blocked(n, 0) = true;
      }
    }

    model->solveStep(*solver_ntn, "explicit_lumped");

    auto ekin = model->getEnergy("kinetic");
    auto epot = model->getEnergy("potential");
    auto work = model->getEnergy("external work new");
    auto econ = solver_ntn->getExternalWork();
    if (s == 0)
    {
      einit = ekin + epot - (work + econ[0] + econ[1]);
    }
    energies << s * time_step << "," << ekin << "," << epot << "," << work
             << "," << econ[0] << "," << econ[1] << ","
             << ekin + epot - (work + econ[0] + econ[1]) - einit << std::endl;

    if (s % dump_every == 0)
    {
      model->dump();
      std::cout << "Step " << s << "\t\r" << std::flush;
    }
  }
  std::cout << "Simulation done." << std::endl;
  return EXIT_SUCCESS;
}
