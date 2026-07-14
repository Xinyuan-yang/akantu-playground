// Velocity boundary condition code //
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
#include "node_synchronizer.hh"
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
              << " <coulomb-mus> <nb-it-nodes> <damping: n|s|l>" << std::endl;
    return EXIT_FAILURE;
  }

  const std::string input_file = "ras_ss_sw.in";
  const std::string coulomb_mu_text = argv[1];
  const Real coulomb_mus = std::stod(coulomb_mu_text);
  const UInt nb_it_nodes = std::stoul(argv[2]);
  const std::string damping_mode = argv[3];
  initialize(input_file, argc, argv);
  const ParserSection &data = getUserParser();

  const auto &comm = Communicator::getStaticCommunicator();
  auto prank = comm.whoAmI();

  std::string output_folder =
      "steady_state_SW_local_" + coulomb_mu_text + "_" + std::to_string(nb_it_nodes) + "_" + damping_mode;
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

  // Periodic BC switch here
  // mesh->makePeriodic(_x, "slider_left", "slider_right");
  // mesh->makePeriodic(_x, "base_left", "base_right");

  model = std::make_unique<SolidMechanicsModel>(*mesh);

  Real time_step_factor = data.getParameter("time_step_factor");

  Int normal_dir = 1;

  solver_ntn = std::make_unique<NTNContactSolverCallback>(
      *model, "slider_bottom", "base_top", normal_dir, time_step_factor);

  const auto &mat = model->getMaterial("slider");

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
  Real fss = 0.30; // set residual friction to 0.30 for steady state slip weakening
  Real E = mat.getParam("E");
  Real nu = mat.getParam("nu");
  Real shear_modulus = E / (2. * (1. + nu));
  Real normal_strain_applied = trac_top(1) / E - nu * nu * trac_top(1) / E;

  Array<Real> &displacement = model->getDisplacement();
  Array<Real> &position = mesh->getNodes();
  UInt nb_nodes = model->getFEEngine().getMesh().getNbNodes();
  Array<Real> current_position(nb_nodes, spatial_dimension);

  Matrix<Real> pressure_top(spatial_dimension, spatial_dimension);
  Matrix<Real> pressure_bottom(spatial_dimension, spatial_dimension);
  pressure_top.zero();
  pressure_bottom.zero();
  const Real p_top = std::abs(trac_top(1));
  const Real p_bottom = std::abs(trac_bottom(1));
  for (UInt d = 0; d < spatial_dimension; ++d)
  {
    pressure_top(d, d) = -p_top;
    pressure_bottom(d, d) = -p_bottom;
  }

  auto apply_follower_pressure = [&]()
  {
    mesh->getNodeSynchronizer().synchronizeArray(displacement);

    for (UInt n = 0; n < nb_nodes; ++n)
    {
      for (UInt d = 0; d < spatial_dimension; ++d)
      {
        current_position(n, d) = position(n, d) + displacement(n, d);
      }
    }
    mesh->getNodeSynchronizer().synchronizeArray(current_position);

    auto &fem_boundary = model->getFEEngineBoundary();
    fem_boundary.computeNormalsOnIntegrationPoints(current_position, _not_ghost);
    fem_boundary.computeNormalsOnIntegrationPoints(current_position, _ghost);

    model->getExternalForce().zero();
    model->applyBC(BC::Neumann::FromStress(pressure_top), "slider_top");
    model->applyBC(BC::Neumann::FromStress(pressure_bottom), "base_bottom");
    mesh->getNodeSynchronizer().reduceSynchronizeArray<AddOperation>(
        model->getExternalForce());
  };

  auto update_current_position = [&]()
  {
    for (UInt n = 0; n < nb_nodes; ++n)
    {
      for (UInt d = 0; d < spatial_dimension; ++d)
      {
        current_position(n, d) = position(n, d) + displacement(n, d);
      }
    }
  };

  auto get_boundary_tangent = [&](const std::string & group_name)
  {
    const auto &nodes =
        mesh->getElementGroup(group_name).getNodeGroup().getNodes();

    const Vector<Real> &upper_bounds = mesh->getUpperBounds();
    const Vector<Real> &lower_bounds = mesh->getLowerBounds();
    const Real left = lower_bounds(_x);
    const Real right = upper_bounds(_x);
    const Real tol = std::max(1e-12, 1e-10 * std::abs(right - left));

    Vector<Real> left_point(spatial_dimension);
    Vector<Real> right_point(spatial_dimension);
    left_point.zero();
    right_point.zero();
    UInt left_count = 0;
    UInt right_count = 0;

    for (auto n : nodes)
    {
      if (not mesh->isLocalOrMasterNode(n))
      {
        continue;
      }

      if (std::abs(position(n, _x) - left) < tol)
      {
        for (UInt d = 0; d < spatial_dimension; ++d)
        {
          left_point(d) += current_position(n, d);
        }
        ++left_count;
      }

      if (std::abs(position(n, _x) - right) < tol)
      {
        for (UInt d = 0; d < spatial_dimension; ++d)
        {
          right_point(d) += current_position(n, d);
        }
        ++right_count;
      }
    }

    comm.allReduce(left_point, SynchronizerOperation::_sum);
    comm.allReduce(right_point, SynchronizerOperation::_sum);
    comm.allReduce(left_count, SynchronizerOperation::_sum);
    comm.allReduce(right_count, SynchronizerOperation::_sum);

    Vector<Real> tangent(spatial_dimension);
    tangent.zero();
    tangent(_x) = 1.;

    if (left_count > 0 and right_count > 0)
    {
      for (UInt d = 0; d < spatial_dimension; ++d)
      {
        left_point(d) /= left_count;
        right_point(d) /= right_count;
        tangent(d) = right_point(d) - left_point(d);
      }

      Real tangent_norm = 0.;
      for (UInt d = 0; d < spatial_dimension; ++d)
      {
        tangent_norm += tangent(d) * tangent(d);
      }
      tangent_norm = std::sqrt(tangent_norm);

      if (tangent_norm > 0.)
      {
        for (UInt d = 0; d < spatial_dimension; ++d)
        {
          tangent(d) /= tangent_norm;
        }
      }
    }

    return tangent;
  };

  Real t_fin = 0.5 / cs * 5;

  // Steady state initialization
  for (UInt n = 0; n < nb_nodes; ++n)
  {
    if (not mesh->isLocalOrMasterNode(n))
    {
      continue;
    }

    displacement(n, 0) = fss * -trac_top(1) / (shear_modulus)*position(n, 1);
    displacement(n, 1) = normal_strain_applied * position(n, 1);
  }

  // Set follower pressure boundary conditions for dynamic simulation
  apply_follower_pressure();

  ///// Set to steady state
  const auto &slider_nodes =
      mesh->getElementGroup("slider").getNodeGroup().getNodes();
  const auto &base_nodes =
      mesh->getElementGroup("base").getNodeGroup().getNodes();

  // Specify initial nodal velocity
  auto &velo = model->getVelocity();
  auto &increment = model->getIncrement();
  auto friction = solver_ntn->getFriction();

  const Real mu_s = coulomb_mus; // keep static friction from argv[1]
  const Real mu_d = 0.1;
  const Real d_c = 1e-3;

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

    update_current_position();
    const auto top_tangent = get_boundary_tangent("slider_top");
    const auto bottom_tangent = get_boundary_tangent("base_bottom");

    for (UInt n = 0; n < nb_nodes; ++n)
    {
      if (not mesh->isLocalOrMasterNode(n))
      {
        continue;
      }

      if (std::abs(position(n, 1) - top) < 1e-6)
      {
        for (UInt d = 0; d < spatial_dimension; ++d)
        {
          velo(n, d) = 0.5 * shear_vel * top_tangent(d);
        }
      }
      if (std::abs(position(n, 1) - bottom) < 1e-6)
      {
        for (UInt d = 0; d < spatial_dimension; ++d)
        {
          velo(n, d) = -0.5 * shear_vel * bottom_tangent(d);
        }
      }
    }

    for (UInt n = 0; n < nb_nodes; ++n)
    {
      if (not mesh->isLocalOrMasterNode(n))
      {
        continue;
      }

      if (std::abs(position(n, 1) - top) < 1e-6)
      {
        for (UInt d = 0; d < spatial_dimension; ++d)
        {
          displacement(n, d) += 0.5 * disp_incr * top_tangent(d);
          blocked(n, d) = true;
        }
      }
      if (std::abs(position(n, 1) - bottom) < 1e-6)
      {
        for (UInt d = 0; d < spatial_dimension; ++d)
        {
          displacement(n, d) += -0.5 * disp_incr * bottom_tangent(d);
          blocked(n, d) = true;
        }
      }
    }

    mesh->getNodeSynchronizer().synchronizeArray(velo);
    mesh->getNodeSynchronizer().synchronizeArray(displacement);
    mesh->getNodeSynchronizer().synchronizeArray(blocked);

    apply_follower_pressure();

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
