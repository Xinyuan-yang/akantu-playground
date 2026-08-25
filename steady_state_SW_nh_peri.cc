// Velocity boundary condition code.
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <ostream>
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
#include "ntn_initiation_function.hh"

using namespace akantu;

/* ------------------------------------------------------------------------ */
/* Main                                                                     */
/* ------------------------------------------------------------------------ */
int main(int argc, char *argv[])
{

  // Parse Akantu options first, so --aka_input_file can replace the default.
  const std::string input_file = "ras_ss_swnh.in";
  initialize(input_file, argc, argv);
  const ParserSection &data = getUserParser();
  const UInt nb_it_nodes = data.getParameter("nb_it_nodes");
  const std::string damping_mode = data.getParameter("damping_mode");
  const std::string output_prefix = data.getParameter("output_prefix");
  const bool is_traction_driven = data.getParameter("is_traction_driven");

  const auto &comm = Communicator::getStaticCommunicator();
  auto prank = comm.whoAmI();
  UInt spatial_dimension = data.getParameter("spatial_dimension");
  std::unique_ptr<Mesh> mesh;
  std::unique_ptr<SolidMechanicsModel> model;
  std::unique_ptr<NTNContactSolverCallback> solver_ntn;
  mesh = std::make_unique<Mesh>(spatial_dimension);
  const std::string mesh_file =
      "ntn_test_" + std::to_string(nb_it_nodes) + ".msh";
  if (prank == 0)
  {
    mesh->read(mesh_file);
  }

  std::shared_ptr<MeshPartition> partition;
  const Int psize = comm.getNbProc();

  if (psize > 1 && prank == 0)
  {
    auto partition_mapping =
        std::make_shared<ElementTypeMapArray<Idx>>("x_strip_partition");

    Real xmin = std::numeric_limits<Real>::max();
    Real xmax = -std::numeric_limits<Real>::max();

    for (const auto &type :
         mesh->elementTypes(spatial_dimension, _not_ghost, _ek_not_defined))
    {
      const auto nb_element = mesh->getNbElement(type);
      for (Idx e = 0; e < nb_element; ++e)
      {
        const Element element{type, e, _not_ghost};
        const auto barycenter = mesh->getBarycenter(element);
        xmin = std::min(xmin, barycenter(_x));
        xmax = std::max(xmax, barycenter(_x));
      }
    }

    const Real length = xmax - xmin;

    for (const auto &type :
         mesh->elementTypes(spatial_dimension, _not_ghost, _ek_not_defined))
    {
      const auto nb_element = mesh->getNbElement(type);
      auto &type_partition =
          partition_mapping->alloc(nb_element, 1, type, _not_ghost);

      for (Idx e = 0; e < nb_element; ++e)
      {
        const Element element{type, e, _not_ghost};
        const auto barycenter = mesh->getBarycenter(element);
        Int proc = 0;

        if (length > 0.)
        {
          const Real x_rel = (barycenter(_x) - xmin) / length;
          proc = std::min<Int>(psize - 1, std::floor(x_rel * psize));
        }

        type_partition(e) = proc;
      }
    }

    auto mesh_data_partition =
        std::make_shared<MeshPartitionMeshData>(*mesh, spatial_dimension);
    mesh_data_partition->setPartitionMapping(partition_mapping);
    mesh_data_partition->partitionate(psize);
    partition = mesh_data_partition;
  }

  mesh->distribute(partition);

  // Periodic BC switch ON
  mesh->makePeriodic(_x, "slider_left", "slider_right");
  mesh->makePeriodic(_x, "base_left", "base_right");

  model = std::make_unique<SolidMechanicsModel>(*mesh);

  Real time_step_factor = data.getParameter("time_step_factor");

  Int normal_dir = 1;

  solver_ntn = std::make_unique<NTNContactSolverCallback>(
      *model, "slider_bottom", "base_top", normal_dir, time_step_factor);
  solver_ntn->getContact()->initParallel();

  auto friction = solver_ntn->getFriction();
  const Real mu_s = friction->get("mu_s");
  const Real mu_k = friction->get("mu_k");
  const Real d_c = friction->get("d_c");
  std::ostringstream output_name;
  output_name << "SW_nh_" << (is_traction_driven ? "trac_" : "peri_")
              << mu_s << "_" << nb_it_nodes << "_" << damping_mode << "_"
              << output_prefix;
  const std::string output_folder = output_name.str();

  const auto &mat = model->getMaterial("slider");
  const Real E = mat.getParam("E");
  const Real nu = mat.getParam("nu");
  const Real shear_modulus = E / (2. * (1. + nu));
  const Real effective_mode_ii_modulus = E;  // for plane stress
  const Real left = mesh->getLowerBounds()(_x);
  const Real right = mesh->getUpperBounds()(_x);
  const Real x_mid = 0.5 * (left + right);
  const Real precrack_length = (right - left) / 20.;
  const Real precrack_half_length = 0.5 * precrack_length;

  Real cp = mat.getPushWaveSpeed(ElementNull);
  Real cs = mat.getShearWaveSpeed(ElementNull);

  std::cout << "P-wave speed = " << cp << std::endl;
  std::cout << "S-wave speed = " << cs << std::endl;

  Real shear_vel = data.getParameter("shear_velocity");
  Vector<Real> trac_top = data.getParameter("top_traction");
  Vector<Real> trac_bottom = data.getParameter("bot_traction");

  if (is_traction_driven)
  {
    // Choose the shear traction so the centered weak zone is 1.1 G_l.
    const Real normal_pressure = std::abs(trac_top(_y));
    const Real strength_drop = (mu_s - mu_k) * normal_pressure;
    const Real target_G_l = precrack_length / 1.1;
    if (strength_drop <= 0. || d_c <= 0. || target_G_l <= 0.)
    {
      std::cerr << "Cannot set traction from G_l: require mu_s > mu_k, "
                   "d_c > 0, and a nonzero precrack length."
                << std::endl;
      return EXIT_FAILURE;
    }
    const Real driving_stress = std::sqrt(
        effective_mode_ii_modulus * strength_drop * d_c /
        (std::acos(-1.) * target_G_l));
    const Real steady_shear_traction =
        mu_k * normal_pressure + driving_stress;
    trac_top(_x) = steady_shear_traction;
    trac_bottom(_x) = -steady_shear_traction;
  }

  model->setBaseName(output_folder);
  model->addDumpField("blocked_dofs");
  model->addDumpField("mass");
  model->addDumpFieldVector("velocity");
  model->addDumpFieldVector("acceleration");
  model->addDumpFieldVector("displacement");
  model->addDumpFieldVector("internal_force");
  model->addDumpFieldVector("external_force");

  // Static analytical solution
  const Real fss = data.getParameter("fss");
  Real normal_strain_applied = trac_top(1) / E - nu * nu * trac_top(1) / E;

  if (is_traction_driven)
  {
    // Static mode-II estimates for a linear slip-weakening interface.
    // G_l is the Griffith crack length and l_pz the zero-speed process-zone
    // size. They use the effective mode-II modulus mu / (1 - nu).
    const Real normal_pressure = std::abs(trac_top(_y));
    const Real tau_peak = mu_s * normal_pressure;
    const Real tau_residual = mu_k * normal_pressure;
    const Real tau_initial = std::abs(trac_top(_x));
    const Real strength_drop = tau_peak - tau_residual;
    const Real driving_stress = tau_initial - tau_residual;
    if (strength_drop > 0. && driving_stress > 0. && d_c > 0.)
    {
      const Real G_l = effective_mode_ii_modulus * strength_drop * d_c /
                       (std::acos(-1.) * driving_stress * driving_stress);
      const Real l_pz = 9. * std::acos(-1.) / 32. *
                        effective_mode_ii_modulus * d_c / strength_drop;
      if (prank == 0)
      {
        std::cout << "Traction-driven fracture scales: G_l = " << G_l
                  << ", l_pz = " << l_pz
                  << ", driving stress = " << driving_stress
                  << std::endl;
      }
    }
    else if (prank == 0)
    {
      std::cerr << "Cannot calculate G_l and l_pz: require mu_s > mu_k, "
                   "d_c > 0, and tau_initial > tau_residual."
                << std::endl;
    }
  }

  Array<Real> &displacement = model->getDisplacement();
  Array<Real> &position = mesh->getNodes();
  UInt nb_nodes = model->getFEEngine().getMesh().getNbNodes();

  Real t_fin = 0.5 / cs * 5;

  // Steady state initialization
  for (UInt n = 0; n < nb_nodes; ++n)
  {
    // Shear loading starts from zero in traction-driven mode.
    displacement(n, _x) = is_traction_driven
                             ? 0.
                             : fss * -trac_top(_y) / shear_modulus *
                                   position(n, _y) * 0.95;
    displacement(n, _y) = normal_strain_applied * position(n, _y);
  }

  // Set boundary conditions for dynamic simulation
  if (is_traction_driven)
  {
    // Apply normal loading initially; ramp the shear traction in the loop.
    auto initial_trac_top = trac_top;
    auto initial_trac_bottom = trac_bottom;
    initial_trac_top(_x) = 0.;
    initial_trac_bottom(_x) = 0.;
    model->applyBC(BC::Neumann::FromTraction(initial_trac_top), "slider_top");
    model->applyBC(BC::Neumann::FromTraction(initial_trac_bottom), "base_bottom");
  }
  else
  {
    model->applyBC(BC::Neumann::FromTraction(trac_top), "slider_top");
    model->applyBC(BC::Neumann::FromTraction(trac_bottom), "base_bottom");
  }

  ///// Set to steady state
  const auto &slider_nodes =
      mesh->getElementGroup("slider").getNodeGroup().getNodes();
  const auto &base_nodes =
      mesh->getElementGroup("base").getNodeGroup().getNodes();

  // Specify initial nodal velocity

  auto &velo = model->getVelocity();
  auto &increment = model->getIncrement();
  auto dt = model->getTimeStep();

  // for (auto n : slider_nodes)
  // {
  //   velo(n, _x) = 0.5 * shear_vel;
  //   increment(n, _x) = 0.5 * shear_vel * dt;
  // }
  // for (auto n : base_nodes)
  // {
  //   velo(n, _x) = -0.5 * shear_vel;
  //   increment(n, _x) = -0.5 * shear_vel * dt;
  // }

  auto contact = solver_ntn->getContact();

  contact->setBaseName(output_folder + "_contact_interface");
  contact->addDumpField("contact_pressure");

  friction->setBaseName(output_folder + "_friction_interface");
  friction->addDumpField("friction_traction");
  friction->addDumpField("frictional_strength");


  // The centered weak zone is a geometric precrack in velocity-driven mode;
  // traction-driven loading selects its stress so it is 1.1 G_l.
  UInt weak_zone_nodes = 0;
  for (Int n = 0; n < contact->getNbContactNodes(); ++n)
  {
    const Idx slave = contact->getSlaves()(n);
    if (std::abs(position(slave, _x) - x_mid) <= precrack_half_length)
    {
      // setParam expects a mesh-node ID and maps it to its contact-array
      // index internally. Passing n selects an unrelated contact node.
      friction->setParam("mu_s", slave, mu_k);
      friction->setParam("mu_k", slave, mu_k);
      if (mesh->isLocalOrMasterNode(slave))
      {
        ++weak_zone_nodes;
      }
    }
  }

  comm.allReduce(weak_zone_nodes, SynchronizerOperation::_sum);
  if (prank == 0)
  {
    std::cout << "Centered weak zone: length = " << precrack_length
              << " (L / 20), nodes = " << weak_zone_nodes << std::endl;
  }
  velo.zero();
  increment.zero();

  std::cout << "rank " << prank
            << " contact nodes = "
            << solver_ntn->getContact()->getNbContactNodes()
            << std::endl;

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
  UInt dump_every = nb_steps / 500;

  // Smoothly introduce the prescribed loading from rest.
  const Real ramp_time = 2. * 0.5 / cs;
  const Real pi = std::acos(-1.);
  auto ramp_factor = [&](Real t)
  {
    if (t >= ramp_time)
    {
      return 1.;
    }
    if (t <= 0.)
    {
      return 0.;
    }
    return 0.5 * (1. - std::cos(pi * t / ramp_time));
  };

  Real previous_traction_ramp_factor = 0.;

  std::cout << "Time step = " << time_step << std::endl;
  std::cout << "Number of steps = " << nb_steps << std::endl;
  std::cout << "Dump every = " << dump_every << std::endl;

  Real alpha = 0; // mass proportional damping
  Real beta = 0;  // stiffness proportional damping

  if (damping_mode == "n")
  {
    alpha = 0;
    beta = 0;
  }
  else if (damping_mode == "s")
  {
    alpha = 40;
    beta = 1e-10;
  }
  else if (damping_mode == "l")
  {
    alpha = 40;
    beta = 5e-9;
  }
  else
  {
    std::cerr << "Unknown damping mode '" << damping_mode
              << "'. Use n, s, or l." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Damping mode " << damping_mode << ": alpha = " << alpha
            << ", beta = " << beta << std::endl;

  model->assembleMass();
  auto &M = model->getDOFManager().getMatrix("M");

  model->assembleStiffnessMatrix(true);
  auto &K = model->getDOFManager().getMatrix("K");

  auto &C = model->getDOFManager().getNewMatrix("C", "K");
  C.zero();
  C.add(M, alpha);
  C.add(K, beta);
  std::cout << "has C = " << model->getDOFManager().hasMatrix("C") << std::endl;

  std::ofstream energies;
  if (prank == 0)
  {
    auto file_name =
        std::filesystem::path("friction-energies-" + output_folder + ".csv");
    energies.open(file_name.c_str(), std::ofstream::out | std::ofstream::trunc);
    energies << "time,ekin,epot,work,econ,efri,tot" << std::endl;
  }

  //Add a pertubation field for steady analysis
  auto einit = 0.;
  Real external_work = 0.;
  std::cout << "Starting simulation..." << std::endl;

  for (UInt s = 0; s < nb_steps; ++s)
  {
    if (is_traction_driven)
    {
      const Real traction_ramp_factor = ramp_factor(s * time_step);
      const Real traction_ramp_increment =
          traction_ramp_factor - previous_traction_ramp_factor;

      // Neumann loads accumulate in external force, so add only the change.
      Vector<Real> shear_traction_top(spatial_dimension);
      Vector<Real> shear_traction_bottom(spatial_dimension);
      shear_traction_top.setZero();
      shear_traction_bottom.setZero();
      shear_traction_top(_x) = traction_ramp_increment * trac_top(_x);
      shear_traction_bottom(_x) = traction_ramp_increment * trac_bottom(_x);
      model->applyBC(BC::Neumann::FromTraction(shear_traction_top), "slider_top");
      model->applyBC(BC::Neumann::FromTraction(shear_traction_bottom), "base_bottom");
      previous_traction_ramp_factor = traction_ramp_factor;
    }
    else
    {
      // Apply velocity.
      UInt nb_nodes = model->getFEEngine().getMesh().getNbNodes();
      Array<Real> &position = mesh->getNodes();
      Array<Real> &velo = model->getVelocity();
      const Vector<Real> &upperBounds = mesh->getUpperBounds();
      const Vector<Real> &lowerBounds = mesh->getLowerBounds();
      Real top = upperBounds(1);
      Real bottom = lowerBounds(1);
      Array<Real> &displacement = model->getDisplacement();
      Array<bool> &blocked = model->getBlockedDOFs();

      Real t = s * time_step;
      Real rf = ramp_factor(t);
      Real current_shear_vel = rf * shear_vel;

      for (UInt n = 0; n < nb_nodes; ++n)
      {
        if (std::abs(position(n, 1) - top) < 1e-6)
        {
          for (UInt d = 0; d < spatial_dimension; ++d)
          {
            velo(n, _x) = 0.5 * current_shear_vel;
          }
        }
        if (std::abs(position(n, 1) - bottom) < 1e-6)
        {
          for (UInt d = 0; d < spatial_dimension; ++d)
          {
            velo(n, _x) = -0.5 * current_shear_vel;
          }
        }
      }

      Real disp_incr = current_shear_vel * time_step;

      for (UInt n = 0; n < nb_nodes; ++n)
      {
        if (std::abs(position(n, 1) - top) < 1e-6)
        {
          displacement(n, 0) += 0.5 * disp_incr;
          increment(n, _x) += 0.5 * disp_incr;
          blocked(n, 0) = true;
        }
        if (std::abs(position(n, 1) - bottom) < 1e-6)
        {
          displacement(n, 0) += -0.5 * disp_incr;
          increment(n, _x) += -0.5 * disp_incr;
          blocked(n, 0) = true;
        }
      }
    }

    model->solveStep(*solver_ntn, "explicit_lumped");

    auto ekin = model->getEnergy("kinetic");
    auto epot = model->getEnergy("potential");
    const auto external_work_increment = model->getEnergy("external work");
    external_work += external_work_increment;
    auto econ = solver_ntn->getExternalWork();
    if (s == 0)
    {
      einit = ekin + epot - (external_work + econ[0] + econ[1]);
    }
    energies << s * time_step << "," << ekin << "," << epot << "," << external_work
             << "," << econ[0] << "," << econ[1] << ","
             << ekin + epot - (external_work + econ[0] + econ[1]) - einit
             << std::endl;
    if (s % dump_every == 0)
    {
      const Real dump_time = (s + 1) * time_step;
      model->dump(dump_time, s + 1);
      contact->dump(dump_time, s + 1);
      friction->dump(dump_time, s + 1);
      std::cout << "Step " << s << "\t\r" << std::flush;
    } 
  }
  std::cout << "Simulation done." << std::endl;
  return EXIT_SUCCESS;
}
