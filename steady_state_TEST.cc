// Fixed-normal-displacement, velocity-controlled sliding test.
// The top and bottom boundaries are prescribed in both Cartesian directions:
// y is held at the initialized normal preload and x is driven in opposite
// directions. No pressure (Neumann) boundary condition is applied.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>

#include "aka_common.hh"
#include "dumpable_iohelper.hh"
#include "dumper_text.hh"
#include "dumper_variable.hh"
#include "mesh_partition_mesh_data.hh"
#include "node_synchronizer.hh"
#include "ntn_base_contact.hh"
#include "ntn_contact_solvercallback.hh"
#include "ntn_initiation_function.hh"
#include "solid_mechanics_model.hh"
#include "sparse_matrix.hh"

using namespace akantu;

/// Read the two-column ux uy format written by simple_shear_static.
Array<Real> readDisplacementXY(const std::string & filename) {
  std::ifstream input(filename);
  if (not input) {
    AKANTU_EXCEPTION("Could not open static displacement file: " << filename);
  }

  std::vector<Real> values;
  Real ux, uy;
  while (input >> ux >> uy) {
    values.push_back(ux);
    values.push_back(uy);
  }
  if (not input.eof()) {
    AKANTU_EXCEPTION("Malformed ux uy data in static displacement file: "
                     << filename);
  }

  Array<Real> displacement(values.size() / 2, 2);
  for (Idx node = 0; node < displacement.size(); ++node) {
    displacement(node, _x) = values[2 * node];
    displacement(node, _y) = values[2 * node + 1];
  }
  return displacement;
}

using CoordinateKey = std::pair<long long, long long>;

CoordinateKey coordinateKey(Real x, Real y) {
  // Gmsh writes the same coordinates in the static and dynamic meshes.  The
  // quantisation only protects the lookup from text-format roundoff.
  constexpr Real coordinate_tolerance = 1e-10;
  return {std::llround(x / coordinate_tolerance),
          std::llround(y / coordinate_tolerance)};
}

void applyStaticInitialDisplacement(Mesh &dynamic_mesh,
                                    Array<Real> &initial_displacement,
                                    const std::string &static_mesh_file,
                                    const std::string &static_displacement_file,
                                    UInt spatial_dimension) {
  Mesh static_mesh(spatial_dimension);
  static_mesh.read(static_mesh_file);
  const Array<Real> static_displacement =
      readDisplacementXY(static_displacement_file);
  if (static_displacement.size() != static_mesh.getNbNodes()) {
    AKANTU_EXCEPTION("Static displacement file '" << static_displacement_file
                     << "' has " << static_displacement.size()
                     << " rows, but mesh '" << static_mesh_file << "' has "
                     << static_mesh.getNbNodes() << " nodes");
  }

  std::map<CoordinateKey, std::pair<Real, Real>> static_field;
  const auto &static_positions = static_mesh.getNodes();
  for (Idx node = 0; node < static_mesh.getNbNodes(); ++node) {
    static_field.emplace(coordinateKey(static_positions(node, _x),
                                       static_positions(node, _y)),
                         std::make_pair(static_displacement(node, _x),
                                        static_displacement(node, _y)));
  }

  const Real x_min = dynamic_mesh.getLowerBounds()(_x);
  const Real x_max = dynamic_mesh.getUpperBounds()(_x);
  const auto copy_static_field = [&](const Array<Int> &nodes, Real sign,
                                     bool rotate_180) {
    const auto &positions = dynamic_mesh.getNodes();
    for (const Idx node : nodes) {
      if (not dynamic_mesh.isLocalOrMasterNode(node)) {
        continue;
      }
      // In midpoint-centred coordinates this is (x, y) -> (-x, -y).
      // The mesh coordinates run from x_min to x_max, hence x -> x_min +
      // x_max - x in the stored coordinates.
      const Real source_x = rotate_180 ? x_min + x_max - positions(node, _x)
                                       : positions(node, _x);
      const Real source_y = rotate_180 ? -positions(node, _y)
                                       : positions(node, _y);
      const auto it = static_field.find(
          coordinateKey(source_x, source_y));
      if (it == static_field.end()) {
        AKANTU_EXCEPTION("No static displacement found for dynamic node "
                         << node << " at (" << positions(node, _x) << ", "
                         << positions(node, _y) << ")");
      }
      initial_displacement(node, _x) = sign * it->second.first;
      initial_displacement(node, _y) = sign * it->second.second;
    }
  };

  // The static mesh is the upper slider. The base samples the field at the
  // 180-degree mirrored point and receives the negative displacement vector.
  copy_static_field(
      dynamic_mesh.getElementGroup("slider").getNodeGroup().getNodes(), 1.,
      false);
  copy_static_field(
      dynamic_mesh.getElementGroup("base").getNodeGroup().getNodes(), -1.,
      true);
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <coulomb-mus> <nb-it-nodes> <damping: n|s|l>" << std::endl;
    return EXIT_FAILURE;
  }

  const std::string input_file = "ras_ss_swnh.in";
  const std::string coulomb_mu_text = argv[1];
  const Real coulomb_mus = std::stod(coulomb_mu_text);
  const UInt nb_it_nodes = std::stoul(argv[2]);
  const std::string damping_mode = argv[3];
  initialize(input_file, argc, argv);
  const ParserSection &data = getUserParser();

  const auto &comm = Communicator::getStaticCommunicator();
  const auto prank = comm.whoAmI();
  const UInt spatial_dimension = data.getParameter("spatial_dimension");
  const std::string output_folder =
      "steady_state_TEST_" + coulomb_mu_text + "_" +
      std::to_string(nb_it_nodes) + "_" + damping_mode;

  auto mesh = std::make_unique<Mesh>(spatial_dimension);
  const std::string mesh_file =
      "ntn_test_" + std::to_string(nb_it_nodes) + ".msh";
  const std::string static_mesh_file =
      "ntn_static_" + std::to_string(nb_it_nodes) + ".msh";
  const std::string static_displacement_file =
      "ntn_static" + std::to_string(nb_it_nodes) + ".txt";
  if (prank == 0) {
    mesh->read(mesh_file);
  }

  std::shared_ptr<MeshPartition> partition;
  const Int psize = comm.getNbProc();
  if (psize > 1 and prank == 0) {
    auto mapping =
        std::make_shared<ElementTypeMapArray<Idx>>("x_strip_partition");
    Real xmin = std::numeric_limits<Real>::max();
    Real xmax = -std::numeric_limits<Real>::max();

    for (const auto &type :
         mesh->elementTypes(spatial_dimension, _not_ghost, _ek_not_defined)) {
      for (Idx e = 0; e < mesh->getNbElement(type); ++e) {
        const auto x = mesh->getBarycenter({type, e, _not_ghost})(_x);
        xmin = std::min(xmin, x);
        xmax = std::max(xmax, x);
      }
    }

    const Real length = xmax - xmin;
    for (const auto &type :
         mesh->elementTypes(spatial_dimension, _not_ghost, _ek_not_defined)) {
      const auto nb_elements = mesh->getNbElement(type);
      auto &part = mapping->alloc(nb_elements, 1, type, _not_ghost);
      for (Idx e = 0; e < nb_elements; ++e) {
        const auto x = mesh->getBarycenter({type, e, _not_ghost})(_x);
        const Int rank = length > 0.
                             ? std::min<Int>(psize - 1,
                                             std::floor((x - xmin) / length * psize))
                             : 0;
        part(e) = rank;
      }
    }

    auto mesh_data =
        std::make_shared<MeshPartitionMeshData>(*mesh, spatial_dimension);
    mesh_data->setPartitionMapping(mapping);
    mesh_data->partitionate(psize);
    partition = mesh_data;
  }
  mesh->distribute(partition);

  auto model = std::make_unique<SolidMechanicsModel>(*mesh);
  const Real time_step_factor = data.getParameter("time_step_factor");
  auto solver_ntn = std::make_unique<NTNContactSolverCallback>(
      *model, "slider_bottom", "base_top", _y, time_step_factor);
  solver_ntn->getContact()->initParallel();

  const auto &mat = model->getMaterial("slider");
  const Real cs = mat.getShearWaveSpeed(ElementNull);
  const Real shear_vel = data.getParameter("shear_velocity");
  const Vector<Real> trac_top = data.getParameter("top_traction");

  model->setBaseName(output_folder);
  model->addDumpField("blocked_dofs");
  model->addDumpField("mass");
  model->addDumpFieldVector("velocity");
  model->addDumpFieldVector("acceleration");
  model->addDumpFieldVector("displacement");
  model->addDumpFieldVector("internal_force");
  model->addDumpFieldVector("external_force");

  auto &position = mesh->getNodes();
  auto &displacement = model->getDisplacement();
  const auto left = mesh->getLowerBounds()(_x);
  const auto right = mesh->getUpperBounds()(_x);
  const Real x_mid = 0.5 * (left + right);
  const UInt nb_nodes = mesh->getNbNodes();
  applyStaticInitialDisplacement(*mesh, displacement, static_mesh_file,
                                 static_displacement_file, spatial_dimension);

  auto &velocity = model->getVelocity();
  auto &increment = model->getIncrement();
  auto &blocked = model->getBlockedDOFs();
  velocity.zero();
  increment.zero();

  auto friction = solver_ntn->getFriction();
  const Real mu_d = 0.1;
  friction->set("mu_s", coulomb_mus);
  friction->set("mu_k", mu_d);
  friction->set("d_c", 1e-6);

  auto contact = solver_ntn->getContact();
  const Real precrack_half_length = (right - left) / 40.;
  for (Int i = 0; i < contact->getNbContactNodes(); ++i) {
    const Idx slave = contact->getSlaves()(i);
    if (std::abs(position(slave, _x) - x_mid) <= precrack_half_length) {
      friction->setParam("mu_s", static_cast<UInt>(i), mu_d);
    }
  }

  contact->setBaseName(output_folder + "_contact_interface");
  contact->addDumpField("normals");
  contact->addDumpField("contact_pressure");
  friction->setBaseName(output_folder + "_friction_interface");
  friction->addDumpField("friction_traction");
  friction->addDumpField("frictional_strength");
  friction->addDumpField("mu_s");
  friction->addDumpField("mu_k");
  friction->addDumpField("d_c");

  for (auto &&[i, master, slave, is_sticking] :
       enumerate(contact->getMasters(), contact->getSlaves(),
                 friction->getIsSticking())) {
    is_sticking = false;
  }

  const Real stable_time_step = model->getStableTimeStep();
  const Real time_step = stable_time_step * time_step_factor;
  model->setTimeStep(time_step);
  const Real t_fin = 0.5 / cs * 6.;
  const UInt nb_steps = t_fin / time_step;
  const UInt dump_every = std::max<UInt>(1, nb_steps / 500);

  Real alpha = 0.;
  Real beta = 0.;
  if (damping_mode == "s") {
    alpha = 40.;
    beta = 1e-10;
  } else if (damping_mode == "l") {
    alpha = 40.;
    beta = 5e-9;
  } else if (damping_mode != "n") {
    std::cerr << "Unknown damping mode '" << damping_mode
              << "'. Use n, s, or l." << std::endl;
    return EXIT_FAILURE;
  }
  model->assembleMass();
  auto &M = model->getDOFManager().getMatrix("M");
  model->assembleStiffnessMatrix(true);
  auto &K = model->getDOFManager().getMatrix("K");
  auto &C = model->getDOFManager().getNewMatrix("C", "K");
  C.zero();
  C.add(M, alpha);
  C.add(K, beta);

  std::ofstream energies("friction-energies-" + output_folder + ".csv",
                         std::ofstream::out | std::ofstream::trunc);
  energies << "time,ekin,epot,work,econ,efri,tot" << std::endl;
  Real external_work = 0.;
  Real einit = 0.;
  const Real ramp_time = 1. / cs;
  const Real pi = std::acos(-1.);
  const Real top = mesh->getUpperBounds()(_y);
  const Real bottom = mesh->getLowerBounds()(_y);

  mesh->getNodeSynchronizer().synchronizeArray(velocity);
  mesh->getNodeSynchronizer().synchronizeArray(displacement);
  mesh->getNodeSynchronizer().synchronizeArray(increment);
  mesh->getNodeSynchronizer().synchronizeArray(blocked);
  increment.zero();
  for (UInt s = 0; s < nb_steps; ++s) {
    const Real t = s * time_step;
    const Real ramp = t >= ramp_time
                          ? 1.
                          : 0.5 * (1. - std::cos(pi * std::max(t, Real{0.}) /
                                                   ramp_time));
    const Real velocity_x = ramp * shear_vel;
    const Real displacement_x = velocity_x * time_step;

    for (UInt n = 0; n < nb_nodes; ++n) {
      if (not mesh->isLocalOrMasterNode(n)) {
        continue;
      }
      if (std::abs(position(n, _y) - top) < 1e-6) {
        velocity(n, _x) = 0.5 * velocity_x;
        velocity(n, _y) = 0.;
        displacement(n, _x) += 0.5 * displacement_x;
        increment(n, _x) = 0.5 * displacement_x;
        blocked(n, _x) = true;
        blocked(n, _y) = true;
      }
      if (std::abs(position(n, _y) - bottom) < 1e-6) {
        velocity(n, _x) = -0.5 * velocity_x;
        velocity(n, _y) = 0.;
        displacement(n, _x) -= 0.5 * displacement_x;
        increment(n, _x) = -0.5 * displacement_x;
        blocked(n, _x) = true;
        blocked(n, _y) = true;
      }
    }

    mesh->getNodeSynchronizer().synchronizeArray(velocity);
    mesh->getNodeSynchronizer().synchronizeArray(displacement);
    mesh->getNodeSynchronizer().synchronizeArray(increment);
    mesh->getNodeSynchronizer().synchronizeArray(blocked);

    model->solveStep(*solver_ntn, "explicit_lumped");
    const auto ekin = model->getEnergy("kinetic");
    const auto epot = model->getEnergy("potential");
    external_work += model->getEnergy("external work");
    const auto econ = solver_ntn->getExternalWork();
    if (s == 0) {
      einit = ekin + epot - (external_work + econ[0] + econ[1]);
    }
    energies << t << ',' << ekin << ',' << epot << ',' << external_work << ','
             << econ[0] << ',' << econ[1] << ','
             << ekin + epot - (external_work + econ[0] + econ[1]) - einit
             << std::endl;

    if (s % dump_every == 0) {
      const Real dump_time = (s + 1) * time_step;
      model->dump(dump_time, s + 1);
      contact->dump(dump_time, s + 1);
      friction->dump(dump_time, s + 1);
      std::cout << "Step " << s << "\t\r" << std::flush;
    }
  }

  return EXIT_SUCCESS;
}
