// Periodic rate-and-state steady-sliding experiment.
// All run-time configuration is supplied by the Akantu input file.
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>

#include "dumpable_iohelper.hh"
#include "dumper_text.hh"
#include "dumper_variable.hh"
#include "solid_mechanics_model.hh"
#include "sparse_matrix.hh"

#include "aka_common.hh"
#include "mesh_partition_mesh_data.hh"
#include "mesh_utils.hh"
#include "ntn_base_contact.hh"
#include "ntn_contact_solvercallback.hh"

using namespace akantu;

int main(int argc, char *argv[]) {
  const std::string input_file = "ras_ss.in";
  initialize(input_file, argc, argv);
  const ParserSection &data = getUserParser();

  const UInt spatial_dimension = data.getParameter("spatial_dimension");
  const std::string mesh_file = data.getParameter("mesh");
  const UInt nb_it_nodes = data.getParameter("nb_it_nodes");
  const std::string damping_mode = data.getParameter("damping_mode");
  const std::string output_prefix = data.getParameter("output_prefix");
  const bool is_traction_driven = data.getParameter("is_traction_driven");
  const Real time_step_factor = data.getParameter("time_step_factor");
  const Real shear_velocity = data.getParameter("shear_velocity");

  const auto &comm = Communicator::getStaticCommunicator();
  auto prank = comm.whoAmI();
  const Int psize = comm.getNbProc();
  Mesh mesh(spatial_dimension);
  if (prank == 0)
  {
    mesh.read(mesh_file);
  }

  // Build the same x-strip partition used by steady_state_SW_nh_peri.  Only
  // rank zero owns the complete mesh before distribute() is called.
  std::shared_ptr<MeshPartition> partition;
  if (psize > 1 && prank == 0) {
    auto partition_mapping =
        std::make_shared<ElementTypeMapArray<Idx>>("x_strip_partition");

    Real xmin = std::numeric_limits<Real>::max();
    Real xmax = -std::numeric_limits<Real>::max();
    for (const auto &type :
         mesh.elementTypes(spatial_dimension, _not_ghost, _ek_not_defined)) {
      const auto nb_element = mesh.getNbElement(type);
      for (Idx e = 0; e < nb_element; ++e) {
        const Element element{type, e, _not_ghost};
        const auto barycenter = mesh.getBarycenter(element);
        xmin = std::min(xmin, barycenter(_x));
        xmax = std::max(xmax, barycenter(_x));
      }
    }

    const Real length = xmax - xmin;
    for (const auto &type :
         mesh.elementTypes(spatial_dimension, _not_ghost, _ek_not_defined)) {
      const auto nb_element = mesh.getNbElement(type);
      auto &type_partition =
          partition_mapping->alloc(nb_element, 1, type, _not_ghost);
      for (Idx e = 0; e < nb_element; ++e) {
        const Element element{type, e, _not_ghost};
        const auto barycenter = mesh.getBarycenter(element);
        Int proc = 0;
        if (length > 0.) {
          const Real x_rel = (barycenter(_x) - xmin) / length;
          proc = std::min<Int>(psize - 1, std::floor(x_rel * psize));
        }
        type_partition(e) = proc;
      }
    }

    auto mesh_data_partition =
        std::make_shared<MeshPartitionMeshData>(mesh, spatial_dimension);
    mesh_data_partition->setPartitionMapping(partition_mapping);
    mesh_data_partition->partitionate(psize);
    partition = mesh_data_partition;
  }
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
  const Real shear_modulus = E / (2. * (1. + nu));
  const Real fss = data.getParameter("fss");
  std::ostringstream output_name;
  output_name << "RAS_" << (is_traction_driven ? "trac_" : "peri_")
              << std::trunc(fss * 100.0) / 100.0 << "_" << nb_it_nodes << "_" << damping_mode << "_"
              << output_prefix;
  const std::string output_folder = output_name.str();
  Vector<Real> traction_top = data.getParameter("top_traction");
  Vector<Real> traction_bottom = data.getParameter("bot_traction");
  if (is_traction_driven) {
    // The RAS steady-state coefficient defines the final shear loading. Only
    // its tangential part is ramped; normal compression is present initially.
    traction_top(_x) = fss * std::abs(traction_top(_y));
    traction_bottom(_x) = -traction_top(_x);
  }
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
    displacement(n, _x) = is_traction_driven
                             ? 0.
                             : fss * -traction_top(_y) / shear_modulus *
                                   position(n, _y);
    displacement(n, _y) = normal_strain * position(n, _y);
  }

  Real t_fin = 0.5 / cs * 25;


  if (is_traction_driven) {
    auto initial_traction_top = traction_top;
    auto initial_traction_bottom = traction_bottom;
    initial_traction_top(_x) = 0.;
    initial_traction_bottom(_x) = 0.;
    model.applyBC(BC::Neumann::FromTraction(initial_traction_top),
                  "slider_top");
    model.applyBC(BC::Neumann::FromTraction(initial_traction_bottom),
                  "base_bottom");
  } else {
    model.applyBC(BC::Neumann::FromTraction(traction_top), "slider_top");
    model.applyBC(BC::Neumann::FromTraction(traction_bottom), "base_bottom");
  }

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

  // Start from a uniform rate-and-state interface: all contact nodes use the
  // same law parameters and the same prescribed initial state.
  for (auto &&[n, master, slave, slip_vel, slip_vel_n, sticking] :
       enumerate(contact->getMasters(), contact->getSlaves(),
                 make_view(slip_velocity, slip_velocity.getNbComponent()),
                 slip_velocity_norm, is_sticking)) {
    sticking = false;
    friction->updateFrictionState(n, phi);
  }

  // Loading begins from rest and is imposed smoothly in the time loop.
  model.getVelocity().zero();
  model.getIncrement().zero();
  const Real stable_time_step = model.getStableTimeStep();
  const Real time_step = stable_time_step * time_step_factor;
  model.setTimeStep(time_step);
  UInt nb_steps = t_fin / time_step;
  UInt dump_every = nb_steps / 500;

  const Real ramp_time = 20 * 0.5 / cs;
  const Real pi = std::acos(-1.);
  auto ramp_factor = [&](Real time) {
    if (time <= 0.) return 0.;
    if (time >= ramp_time) return 1.;
    return 0.5 * (1. - std::cos(pi * time / ramp_time));
  };
  Real previous_traction_ramp = 0.;

  model.assembleMass();
  auto &M = model.getDOFManager().getMatrix("M");
  model.assembleStiffnessMatrix(true);
  auto &K = model.getDOFManager().getMatrix("K");
  auto &C = model.getDOFManager().getNewMatrix("C", "K");
  Real alpha = 0.;
  Real beta = 0.;
  if (damping_mode == "n") {
    alpha = 0.;
    beta = 0.;
  } else if (damping_mode == "s") {
    alpha = 40.;
    beta = 1e-10;
  } else if (damping_mode == "l") {
    alpha = 40.;
    beta = 5e-9;
  } else {
    if (prank == 0) {
      std::cerr << "Unknown damping mode '" << damping_mode
                << "'. Use n, s, or l." << std::endl;
    }
    return EXIT_FAILURE;
  }
  C.zero();
  C.add(M, alpha);
  C.add(K, beta);

  std::ofstream energies;
  if (prank == 0) {
    energies.open("friction-energies-" + output_folder + ".csv",
                  std::ofstream::out | std::ofstream::trunc);
    energies << "time,ekin,epot,work,econ,efri,tot" << std::endl;
  }
  Real initial_energy = 0.;
  Real external_work = 0.;
  const Real top = mesh.getUpperBounds()(_y);
  const Real bottom = mesh.getLowerBounds()(_y);

  if (prank == 0) {
    std::cout << "Time step = " << time_step << ", steps = " << nb_steps
              << ", ramp time = " << ramp_time << ", dump every = "
              << dump_every << ", mesh = " << mesh_file
              << ", damping mode = " << damping_mode << " (alpha = " << alpha
              << ", beta = " << beta << ")" << std::endl;
  }
  for (UInt step = 0; step < nb_steps; ++step) {
    if (is_traction_driven) {
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
    } else {
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
    }

    model.solveStep(solver_ntn, "explicit_lumped");
    const Real ekin = model.getEnergy("kinetic");
    const Real epot = model.getEnergy("potential");
    external_work += model.getEnergy("external work");
    const auto econ = solver_ntn.getExternalWork();
    if (step == 0)
      initial_energy = ekin + epot - (external_work + econ[0] + econ[1]);
    if (prank == 0) {
      energies << step * time_step << ',' << ekin << ',' << epot << ','
               << external_work << ',' << econ[0] << ',' << econ[1] << ','
               << ekin + epot - (external_work + econ[0] + econ[1]) - initial_energy
               << std::endl;
    }
    if (step % dump_every == 0) {
      const Real dump_time = (step + 1) * time_step;
      model.dump(dump_time, step + 1);
      contact->dump(dump_time, step + 1);
      friction->dump(dump_time, step + 1);
    }
  }
  return EXIT_SUCCESS;
}
