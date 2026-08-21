// Velocity boundary condition code.  Define AKANTU_TRACTION_DRIVEN in a
// translation unit that includes this file to build the traction-driven form.
// Takes into parameter the friction coefficient and the number of elements along the contact surface.
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <ostream>
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

  if (argc != 5)
  {
    std::cerr << "Usage: " << argv[0]
              << " <coulomb-mu> <nb-it-nodes> <damping: n|s|l> <Coulomb-like>" << std::endl;
    return EXIT_FAILURE;
  }

  const std::string input_file = "ras_ss_swnh.in";
  const std::string coulomb_mu_text = argv[1];
  const Real coulomb_mus = std::stod(coulomb_mu_text);
  const UInt nb_it_nodes = std::stoul(argv[2]);
  const std::string damping_mode = argv[3];
  const std::string is_coulomb = argv[4];
  initialize(input_file, argc, argv);
  const ParserSection &data = getUserParser();

  const auto &comm = Communicator::getStaticCommunicator();
  auto prank = comm.whoAmI();
#ifdef AKANTU_TRACTION_DRIVEN
  std::string output_folder =
      "SW_nh_trac_" + coulomb_mu_text + "_" + std::to_string(nb_it_nodes) + "_" + damping_mode + "_" + is_coulomb;
#else
  std::string output_folder =
      "SW_nh_peri_" + coulomb_mu_text + "_" + std::to_string(nb_it_nodes) + "_" + damping_mode + "_" + is_coulomb;
#endif
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

  const auto &mat = model->getMaterial("slider");

  Real cp = mat.getPushWaveSpeed(ElementNull);
  Real cs = mat.getShearWaveSpeed(ElementNull);

  std::cout << "P-wave speed = " << cp << std::endl;
  std::cout << "S-wave speed = " << cs << std::endl;

  Real shear_vel = data.getParameter("shear_velocity");
  Vector<Real> trac_top = data.getParameter("top_traction");
  Vector<Real> trac_bottom = data.getParameter("bot_traction");

#ifdef AKANTU_TRACTION_DRIVEN
  // At steady sliding the interface supports the residual strength
  // tau = mu_k |sigma_n|.  The top and bottom shear tractions must be equal
  // and opposite because their outward normals have opposite directions.
  const Real residual_friction = 0.1;
  const Real normal_pressure = std::abs(trac_top(_y));
  const Real steady_shear_traction = residual_friction * normal_pressure + 1e6;
  trac_top(_x) = steady_shear_traction;
  trac_bottom(_x) = -steady_shear_traction;
#endif

  model->setBaseName(output_folder);
  model->addDumpField("blocked_dofs");
  model->addDumpField("mass");
  model->addDumpFieldVector("velocity");
  model->addDumpFieldVector("acceleration");
  model->addDumpFieldVector("displacement");
  model->addDumpFieldVector("internal_force");
  model->addDumpFieldVector("external_force");
  model->addDumpField("stress");

  // Static analytical solution
#ifndef AKANTU_TRACTION_DRIVEN
  Real fss = 0.10;
#endif
  Real E = mat.getParam("E");
  Real nu = mat.getParam("nu");
  Real shear_modulus = E / (2. * (1. + nu));
  Real normal_strain_applied = trac_top(1) / E - nu * nu * trac_top(1) / E;

  Array<Real> &displacement = model->getDisplacement();
  Array<Real> &position = mesh->getNodes();
  UInt nb_nodes = model->getFEEngine().getMesh().getNbNodes();

  Real t_fin = 0.5 / cs * 5;

  // Steady state initialization
  for (UInt n = 0; n < nb_nodes; ++n)
  {
#ifdef AKANTU_TRACTION_DRIVEN
    // Shear loading starts from zero, so do not initialize the body with the
    // displacement field for the final shear traction.
    displacement(n, _x) = 0.;
#else
    displacement(n, _x) =
        fss * -trac_top(_y) / shear_modulus * position(n, _y) * 0.95;
#endif
    displacement(n, _y) = normal_strain_applied * position(n, _y);
  }

  // Set boundary conditions for dynamic simulation
#ifdef AKANTU_TRACTION_DRIVEN
  // Apply normal loading initially; add the shear traction progressively in
  // the time loop below.
  auto initial_trac_top = trac_top;
  auto initial_trac_bottom = trac_bottom;
  initial_trac_top(_x) = 0.;
  initial_trac_bottom(_x) = 0.;
  model->applyBC(BC::Neumann::FromTraction(initial_trac_top), "slider_top");
  model->applyBC(BC::Neumann::FromTraction(initial_trac_bottom), "base_bottom");
#else
  model->applyBC(BC::Neumann::FromTraction(trac_top), "slider_top");
  model->applyBC(BC::Neumann::FromTraction(trac_bottom), "base_bottom");
#endif

  ///// Set to steady state
  const auto &slider_nodes =
      mesh->getElementGroup("slider").getNodeGroup().getNodes();
  const auto &base_nodes =
      mesh->getElementGroup("base").getNodeGroup().getNodes();

  // Specify initial nodal velocity

  const Real mu_s = coulomb_mus; // keep static friction from argv[1]
  const Real mu_d = 0.1;
  Real d_c = 1e-6;
    if (is_coulomb == "y")
  {
    d_c = 0;
  }


  auto &velo = model->getVelocity();
  auto &increment = model->getIncrement();
  auto friction = solver_ntn->getFriction();
  friction->set("mu_s", mu_s);
  friction->set("mu_k", mu_d);
  friction->set("d_c", d_c);
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

  contact->setBaseName(output_folder + "_contact_interface");
  contact->addDumpField("contact_pressure");

  friction->setBaseName(output_folder + "_friction_interface");
  friction->addDumpField("friction_traction");
  friction->addDumpField("frictional_strength");


  // With velocity-controlled steady sliding, a stress-controlled nucleation
  // length is not finite. Use a geometric centered precrack instead.
  const Real left = mesh->getLowerBounds()(_x);
  const Real right = mesh->getUpperBounds()(_x);
  const Real x_mid = 0.5 * (left + right);
  const Real precrack_length = (right - left) / 20.;
  const Real precrack_half_length = 0.5 * precrack_length;
  UInt weak_zone_nodes = 0;
  for (Int n = 0; n < contact->getNbContactNodes(); ++n)
  {
    const Idx slave = contact->getSlaves()(n);
    if (std::abs(position(slave, _x) - x_mid) <= precrack_half_length)
    {
      // setParam expects a mesh-node ID and maps it to its contact-array
      // index internally. Passing n selects an unrelated contact node.
      friction->setParam("mu_s", slave, mu_d);
      friction->setParam("mu_k", slave, mu_d);
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

#ifdef AKANTU_TRACTION_DRIVEN
  Real previous_traction_ramp_factor = 0.;
#endif

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
#ifdef AKANTU_TRACTION_DRIVEN
    const Real traction_ramp_factor = ramp_factor(s * time_step);
    const Real traction_ramp_increment =
        traction_ramp_factor - previous_traction_ramp_factor;

    // Neumann loads accumulate in the external-force vector, so add only the
    // change in shear traction at each step.
    Vector<Real> shear_traction_top(spatial_dimension);
    Vector<Real> shear_traction_bottom(spatial_dimension);
    shear_traction_top.setZero();
    shear_traction_bottom.setZero();
    shear_traction_top(_x) = traction_ramp_increment * trac_top(_x);
    shear_traction_bottom(_x) = traction_ramp_increment * trac_bottom(_x);
    model->applyBC(BC::Neumann::FromTraction(shear_traction_top), "slider_top");
    model->applyBC(BC::Neumann::FromTraction(shear_traction_bottom), "base_bottom");
    previous_traction_ramp_factor = traction_ramp_factor;
#else
    // Apply velocity
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
#endif

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
