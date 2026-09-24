// Bulk-velocity steady sliding test.
//
// The two elastic blocks are pre-stressed to the steady-state sliding solution
// and given a uniform bulk velocity at t = 0: the slider moves at +V/2 and the
// base at -V/2. 
//
// An optional deterministic sinusoidal velocity perturbation can be superposed
// at step 10 through the input file flags apply_initial_velocity_perturbation
// and initial_velocity_perturbation.
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
  const std::string input_file = "ras_ss_swnh_bulk.in";
  initialize(input_file, argc, argv);
  const ParserSection &data = getUserParser();
  const UInt nb_it_nodes = data.getParameter("nb_it_nodes");
  const std::string damping_mode = data.getParameter("damping_mode");
  const std::string output_prefix = data.getParameter("output_prefix");
  const bool initial_bulk_velocity = data.getParameter("initial_bulk_velocity");
  const bool apply_initial_velocity_perturbation =
      data.getParameter("apply_initial_velocity_perturbation");
  const Real initial_velocity_perturbation =
      data.getParameter("initial_velocity_perturbation");

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
  output_name << "SW_nh_bulk_" << mu_s << "_" << nb_it_nodes << "_"
              << damping_mode << "_" << output_prefix;
  const std::string output_folder = output_name.str();

  const auto &mat = model->getMaterial("slider");
  const Real E = mat.getParam("E");
  const Real nu = mat.getParam("nu");
  const Real shear_modulus = E / (2. * (1. + nu));
  const Real left = mesh->getLowerBounds()(_x);
  const Real right = mesh->getUpperBounds()(_x);
  const Real L = right - left;
  const Real x_mid = 0.5 * (left + right);

  Real cp = mat.getPushWaveSpeed(ElementNull);
  Real cs = mat.getShearWaveSpeed(ElementNull);

  std::cout << "P-wave speed = " << cp << std::endl;
  std::cout << "S-wave speed = " << cs << std::endl;
  std::cout << "mu_s = " << mu_s << ", mu_k = " << mu_k << ", d_c = " << d_c
            << std::endl;

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

  // Static analytical solution for steady sliding.
  // Plane stress with traction-free lateral boundaries: sigma_xx = 0.
  const Real fss = data.getParameter("fss");
  const Real normal_strain_applied = trac_top(1) / E;
  const Real lateral_strain_applied = -nu * normal_strain_applied;

  Array<Real> &displacement = model->getDisplacement();
  Array<Real> &position = mesh->getNodes();
  UInt nb_nodes = model->getFEEngine().getMesh().getNbNodes();

  Real t_fin = L / cs * 30;

  // Steady state initialization: uniform shear strain through the thickness
  // plus the normal compression.  Choose fss = mu_k for a stress state that
  // exactly balances kinetic friction and leaves the bulk sliding at constant
  // velocity.
  for (UInt n = 0; n < nb_nodes; ++n)
  {
    if (not mesh->isLocalOrMasterNode(n))
    {
      continue;
    }

    displacement(n, _x) = lateral_strain_applied * (position(n, _x) - x_mid);;
    displacement(n, _y) = normal_strain_applied * position(n, _y);
  }
  mesh->getNodeSynchronizer().synchronizeArray(displacement);

  // Apply normal confining tractions on the top and bottom free surfaces.
  // These remain constant during the simulation (no follower-pressure update
  // is required because the boundary normals do not rotate appreciably).
  model->applyBC(BC::Neumann::FromTraction(trac_top), "slider_top");
  model->applyBC(BC::Neumann::FromTraction(trac_bottom), "base_bottom");

  auto contact = solver_ntn->getContact();

  contact->setBaseName(output_folder + "_contact_interface");
  contact->addDumpField("normals");
  contact->addDumpField("contact_pressure");

  friction->setBaseName(output_folder + "_friction_interface");
  friction->addDumpField("friction_traction");
  friction->addDumpField("frictional_strength");

  // All interface nodes are sliding from the start.
  auto &slip_velocity = friction->getSlipVelocity();
  auto &slip_velocity_norm = friction->getSlipVelocityNorm();
  auto &is_sticking = friction->getIsSticking();

  for (auto &&[n, master, slave, slip_vel, slip_vel_n, is_sticking_flag] :
       enumerate(contact->getMasters(), contact->getSlaves(),
                 make_view(slip_velocity, slip_velocity.getNbComponent()),
                 slip_velocity_norm, is_sticking))
  {
    is_sticking_flag = false;
  }

  // Time of the simulation
  Real stable_time_step = model->getStableTimeStep();
  Real time_step = stable_time_step * time_step_factor;
  model->setTimeStep(time_step);
  UInt nb_steps = t_fin / time_step;
  UInt dump_every = nb_steps / 500;

  std::cout << "Time step = " << time_step << std::endl;
  std::cout << "Number of steps = " << nb_steps << std::endl;
  std::cout << "Dump every = " << dump_every << std::endl;

  // Initialize bulk velocity.  The slider and base move in opposite directions
  // so that the relative slip rate is shear_vel.  No boundary nodes are
  // blocked; the motion is inertial.
  auto &velo = model->getVelocity();
  auto &increment = model->getIncrement();
  velo.zero();
  increment.zero();

  if (initial_bulk_velocity)
  {
    const auto &slider_nodes =
        mesh->getElementGroup("slider").getNodeGroup().getNodes();
    const auto &base_nodes =
        mesh->getElementGroup("base").getNodeGroup().getNodes();

    for (auto n : slider_nodes)
    {
      if (not mesh->isLocalOrMasterNode(n))
      {
        continue;
      }
      velo(n, _x) = 0.5 * shear_vel;
      increment(n, _x) = velo(n, _x) * time_step;
    }
    for (auto n : base_nodes)
    {
      if (not mesh->isLocalOrMasterNode(n))
      {
        continue;
      }
      velo(n, _x) = -0.5 * shear_vel;
      increment(n, _x) = velo(n, _x) * time_step;
    }
  }

  mesh->getNodeSynchronizer().synchronizeArray(velo);
  mesh->getNodeSynchronizer().synchronizeArray(increment);

  std::cout << "rank " << prank
            << " contact nodes = " << solver_ntn->getContact()->getNbContactNodes()
            << std::endl;

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

  auto einit = 0.;
  Real external_work = 0.;
  std::cout << "Starting simulation..." << std::endl;
  //Main LOOP
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
    Array<Real> &displacement = model->getDisplacement();
    Array<bool> &blocked = model->getBlockedDOFs();


    for (UInt n = 0; n < nb_nodes; ++n)
    {
      if (not mesh->isLocalOrMasterNode(n))
      {
        continue;
      }

      if (std::abs(position(n, 1) - top) < 1e-6)
      {
          velo(n, _x) = 0.5 * shear_vel ;
      }
      if (std::abs(position(n, 1) - bottom) < 1e-6)
      {
          velo(n, _x) = -0.5 * shear_vel;
      }
    }

    Real disp_incr = shear_vel * time_step;
    for (UInt n = 0; n < nb_nodes; ++n)
    {
      if (not mesh->isLocalOrMasterNode(n))
      {
        continue;
      }

      if (std::abs(position(n, 1) - top) < 1e-6)
      {
        displacement(n, 0) += 0.5 * disp_incr ;
        increment(n, _x) += 0.5 * disp_incr ;
        blocked(n, 0) = true;
      }
      if (std::abs(position(n, 1) - bottom) < 1e-6)
      {
        displacement(n, 0) += -0.5 * disp_incr ;
        increment(n, _x) += -0.5 * disp_incr ;
        blocked(n, 0) = true;
      }
    }

    // Add a deterministic sinusoidal velocity perturbation at step 10.
    // It is applied to the interior nodes of slider and base, not to the
    // driven top/bottom boundaries.
    if (s == 10 && apply_initial_velocity_perturbation && L > 0.)
    {
      const Real pi = std::acos(-1.);
      const auto &slider_nodes =
          mesh->getElementGroup("slider").getNodeGroup().getNodes();
      const auto &base_nodes =
          mesh->getElementGroup("base").getNodeGroup().getNodes();

      for (auto n : slider_nodes)
      {
        if (not mesh->isLocalOrMasterNode(n))
        {
          continue;
        }
        if (std::abs(position(n, 1) - top) < 1e-6)
        {
          continue;
        }
        const Real perturbation = initial_velocity_perturbation *
                                  std::sin(2. * pi * (position(n, _x) - left) / L);
        velo(n, _x) += perturbation;
        increment(n, _x) += perturbation * time_step;
      }
      for (auto n : base_nodes)
      {
        if (not mesh->isLocalOrMasterNode(n))
        {
          continue;
        }
        if (std::abs(position(n, 1) - bottom) < 1e-6)
        {
          continue;
        }
        const Real perturbation = initial_velocity_perturbation *
                                  std::sin(2. * pi * (position(n, _x) - left) / L);
        velo(n, _x) += perturbation;
        increment(n, _x) += perturbation * time_step;
      }
      mesh->getNodeSynchronizer().synchronizeArray(velo);
      mesh->getNodeSynchronizer().synchronizeArray(increment);
    }

    model->solveStep(*solver_ntn, "explicit_lumped");

    auto ekin = model->getEnergy("kinetic");
    auto epot = model->getEnergy("potential");
    const auto external_work_increment =
        model->getEnergy("external work");
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

