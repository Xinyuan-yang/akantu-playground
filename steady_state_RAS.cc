// Periodic rate-and-state steady-sliding experiment with a centered precrack.
// Define AKANTU_TRACTION_DRIVEN when compiling this translation unit to use a
// smoothly ramped shear traction instead of the imposed boundary velocity.
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "dumpable_iohelper.hh"
#include "dumper_text.hh"
#include "dumper_variable.hh"
#include "solid_mechanics_model.hh"
#include "sparse_matrix.hh"

#include "aka_common.hh"
#include "mesh_utils.hh"
#include "ntn_base_contact.hh"
#include "ntn_contact_solvercallback.hh"

using namespace akantu;

int main(int argc, char *argv[]) {
  const std::string input_file = "ras_ss.in";
  initialize(input_file, argc, argv);
  const ParserSection &data = getUserParser();

  const UInt spatial_dimension = data.getParameter("spatial_dimension");
  UInt nb_it_nodes = data.getParameter("nb_it_nodes");
  const std::string mesh_file =
      "ntn_test_" + std::to_string(nb_it_nodes) + ".msh";
  const std::string output_folder =
      "RAS_" + std::to_string(nb_it_nodes)
#ifdef AKANTU_TRACTION_DRIVEN
      + "_trac"
#endif
      ;
  const Real time_step_factor = data.getParameter("time_step_factor");
#ifndef AKANTU_TRACTION_DRIVEN
  const Real shear_velocity = data.getParameter("shear_velocity");
#endif

  Mesh mesh(spatial_dimension);
  mesh.read(mesh_file);
  // Initialise mesh synchronizers before initialising the contact's parallel
  // data.  A null partition is the serial distribution and is also required
  // for a one-rank run.
  std::shared_ptr<MeshPartition> partition;
  mesh.distribute(partition);
  mesh.makePeriodic(_x, "slider_left", "slider_right");
  mesh.makePeriodic(_x, "base_left", "base_right");

  SolidMechanicsModel model(mesh);
  NTNContactSolverCallback solver_ntn(model, "slider_bottom", "base_top", 1,
                                      time_step_factor);
  auto contact = solver_ntn.getContact();
  contact->initParallel();
  auto friction = solver_ntn.getFriction();

  const auto &mat = model.getMaterial("slider");
  const Real cs = mat.getShearWaveSpeed(ElementNull);
  const Real E = mat.getParam("E");
  const Real nu = mat.getParam("nu");
#ifndef AKANTU_TRACTION_DRIVEN
  const Real shear_modulus = E / (2. * (1. + nu));
#endif
  const Real fss = data.getParameter("fss");
  Vector<Real> traction_top = data.getParameter("top_traction");
  Vector<Real> traction_bottom = data.getParameter("bot_traction");
#ifdef AKANTU_TRACTION_DRIVEN
  // The RAS steady-state coefficient from ras_ss.in defines the final shear
  // loading.  Only its tangential part is ramped; normal compression is
  // present from the first step.
  traction_top(_x) = fss * std::abs(traction_top(_y));
  traction_bottom(_x) = -traction_top(_x);
#endif
  const Real normal_strain = traction_top(_y) / E -
                             nu * nu * traction_top(_y) / E;

  model.setBaseName(output_folder);
  model.addDumpField("blocked_dofs");
  model.addDumpField("mass");
  model.addDumpFieldVector("velocity");
  model.addDumpFieldVector("acceleration");
  model.addDumpFieldVector("displacement");
  model.addDumpFieldVector("internal_force");
  model.addDumpFieldVector("external_force");
  model.addDumpField("stress");

  auto &position = mesh.getNodes();
  auto &displacement = model.getDisplacement();
  const UInt nb_nodes = mesh.getNbNodes();
  for (UInt n = 0; n < nb_nodes; ++n) {
#ifdef AKANTU_TRACTION_DRIVEN
    displacement(n, _x) = 0.;
#else
    displacement(n, _x) = fss * -traction_top(_y) / shear_modulus *
                           position(n, _y);
#endif
    displacement(n, _y) = normal_strain * position(n, _y);
  }

#ifdef AKANTU_TRACTION_DRIVEN
  auto initial_traction_top = traction_top;
  auto initial_traction_bottom = traction_bottom;
  initial_traction_top(_x) = 0.;
  initial_traction_bottom(_x) = 0.;
  model.applyBC(BC::Neumann::FromTraction(initial_traction_top), "slider_top");
  model.applyBC(BC::Neumann::FromTraction(initial_traction_bottom), "base_bottom");
#else
  model.applyBC(BC::Neumann::FromTraction(traction_top), "slider_top");
  model.applyBC(BC::Neumann::FromTraction(traction_bottom), "base_bottom");
#endif

  contact->setBaseName(output_folder + "_contact_interface");
  contact->addDumpField("contact_pressure");
  friction->setBaseName(output_folder + "_friction_interface");
  friction->addDumpField("friction_traction");
  friction->addDumpField("frictional_strength");

  // Set the rate-and-state variable to the value prescribed in ras_ss.in.
  const Real phi = friction->get("friction_state");
  auto &slip_velocity = friction->getSlipVelocity();
  auto &slip_velocity_norm = friction->getSlipVelocityNorm();
  auto &is_sticking = friction->getIsSticking();

  // A centered, zero-strength patch is the precrack.  The a and b terms are
  // also removed there so the revised Dieterich law remains zero-strength.
  const Real left = mesh.getLowerBounds()(_x);
  const Real right = mesh.getUpperBounds()(_x);
  const Real x_mid = 0.5 * (left + right);
  const Real precrack_half_length = (right - left) / 40.;
  UInt precrack_nodes = 0;

  for (auto &&[n, master, slave, slip_vel, slip_vel_n, sticking] :
       enumerate(contact->getMasters(), contact->getSlaves(),
                 make_view(slip_velocity, slip_velocity.getNbComponent()),
                 slip_velocity_norm, is_sticking)) {
    sticking = false;
    friction->updateFrictionState(n, phi);
    if (std::abs(position(slave, _x) - x_mid) <= precrack_half_length) {
      friction->setParam("mu_0", slave, 0.);
      friction->setParam("a", slave, 0.);
      friction->setParam("b", slave, 0.);
      // Explicit for the precrack, even if a contact update changes defaults.
      sticking = false;
      ++precrack_nodes;
    }
  }
  std::cout << "Centered RAS precrack: length = "
            << 2. * precrack_half_length << ", nodes = " << precrack_nodes
            << std::endl;

  // Loading begins from rest and is imposed smoothly in the time loop.
  model.getVelocity().zero();
  model.getIncrement().zero();
  const Real stable_time_step = model.getStableTimeStep();
  const Real time_step = stable_time_step * time_step_factor;
  model.setTimeStep(time_step);
  const Real total_time = 5. * 0.5 / cs;
  const UInt nb_steps = total_time / time_step;
  const UInt dump_every = std::max<UInt>(1, nb_steps / 500);
  const Real ramp_time = 2. * 0.5 / cs;
  const Real pi = std::acos(-1.);
  auto ramp_factor = [&](Real time) {
    if (time <= 0.) return 0.;
    if (time >= ramp_time) return 1.;
    return 0.5 * (1. - std::cos(pi * time / ramp_time));
  };
#ifdef AKANTU_TRACTION_DRIVEN
  Real previous_traction_ramp = 0.;
#endif

  model.assembleMass();
  auto &M = model.getDOFManager().getMatrix("M");
  model.assembleStiffnessMatrix(true);
  auto &K = model.getDOFManager().getMatrix("K");
  auto &C = model.getDOFManager().getNewMatrix("C", "K");
  C.zero();
  C.add(M, 0.);
  C.add(K, 0.);

  std::ofstream energies("friction-energies-" + output_folder + ".csv",
                         std::ofstream::out | std::ofstream::trunc);
  energies << "time,ekin,epot,work,econ,efri,tot" << std::endl;
  Real initial_energy = 0.;
  Real external_work = 0.;
#ifndef AKANTU_TRACTION_DRIVEN
  const Real top = mesh.getUpperBounds()(_y);
  const Real bottom = mesh.getLowerBounds()(_y);
#endif

  std::cout << "Time step = " << time_step << ", steps = " << nb_steps
            << ", total time = " << total_time << ", ramp time = "
            << ramp_time << ", dump every = " << dump_every << std::endl;
  for (UInt step = 0; step < nb_steps; ++step) {
#ifdef AKANTU_TRACTION_DRIVEN
    const Real traction_ramp = ramp_factor(step * time_step);
    const Real traction_increment = traction_ramp - previous_traction_ramp;
    Vector<Real> shear_top(spatial_dimension);
    Vector<Real> shear_bottom(spatial_dimension);
    shear_top.setZero();
    shear_bottom.setZero();
    shear_top(_x) = traction_increment * traction_top(_x);
    shear_bottom(_x) = traction_increment * traction_bottom(_x);
    // Neumann contributions accumulate, hence apply only the increment.
    model.applyBC(BC::Neumann::FromTraction(shear_top), "slider_top");
    model.applyBC(BC::Neumann::FromTraction(shear_bottom), "base_bottom");
    previous_traction_ramp = traction_ramp;
#else
    const Real current_velocity = ramp_factor(step * time_step) * shear_velocity;
    const Real displacement_increment = current_velocity * time_step;
    auto &velocity = model.getVelocity();
    auto &increment = model.getIncrement();
    auto &blocked = model.getBlockedDOFs();
    for (UInt n = 0; n < nb_nodes; ++n) {
      if (std::abs(position(n, _y) - top) < 1e-6) {
        velocity(n, _x) = 0.5 * current_velocity;
        displacement(n, _x) += 0.5 * displacement_increment;
        increment(n, _x) += 0.5 * displacement_increment;
        blocked(n, _x) = true;
      }
      if (std::abs(position(n, _y) - bottom) < 1e-6) {
        velocity(n, _x) = -0.5 * current_velocity;
        displacement(n, _x) -= 0.5 * displacement_increment;
        increment(n, _x) -= 0.5 * displacement_increment;
        blocked(n, _x) = true;
      }
    }
#endif

    model.solveStep(solver_ntn, "explicit_lumped");
    const Real ekin = model.getEnergy("kinetic");
    const Real epot = model.getEnergy("potential");
    external_work += model.getEnergy("external work");
    const auto econ = solver_ntn.getExternalWork();
    if (step == 0)
      initial_energy = ekin + epot - (external_work + econ[0] + econ[1]);
    energies << step * time_step << ',' << ekin << ',' << epot << ','
             << external_work << ',' << econ[0] << ',' << econ[1] << ','
             << ekin + epot - (external_work + econ[0] + econ[1]) - initial_energy
             << std::endl;
    if (step % dump_every == 0) {
      const Real dump_time = (step + 1) * time_step;
      model.dump(dump_time, step + 1);
      contact->dump(dump_time, step + 1);
      friction->dump(dump_time, step + 1);
    }
  }
  return EXIT_SUCCESS;
}
