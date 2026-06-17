/**
 * Copyright (C) 2013-2023 EPFL (Ecole Polytechnique Federale de Lausanne)
 * Laboratory (LSMS - Laboratoire de Simulation en Mecanique des Solides)
 *
 * This file is part of Akantu
 */

/* -------------------------------------------------------------------------- */
#include "coupler_solid_contact.hh"
#include "non_linear_solver.hh"
/* -------------------------------------------------------------------------- */
using namespace akantu;
/* -------------------------------------------------------------------------- */

int main(int argc, char *argv[])
{

  // Initialize the material database
  initialize("material.dat", argc, argv);

  // Create the mesh
  Mesh mesh(2); // Dimension 2

  // Read the mesh
  mesh.read("bloc.msh");

  // Create the model
  CouplerSolidContact coupler(mesh);

  // Initialize each model
  auto &solid = coupler.getSolidMechanicsModel();
  auto &contact = coupler.getContactMechanicsModel();
  auto &&selector = std::make_shared<MeshDataMaterialSelector<std::string>>(
      "physical_names", solid);
  solid.setMaterialSelector(selector);

  // Initialize the coupler
  coupler.initFull(_analysis_method = _explicit_lumped_mass);
  Real time_step = solid.getStableTimeStep() * 0.3;
  coupler.setTimeStep(time_step);
  std::cout << "Time step: " << time_step << std::endl;

  // Setup the contact between the two elastic solids
  auto &&surface_selector = std::make_shared<PhysicalSurfaceSelector>(mesh);
  contact.getContactDetector().setSurfaceSelector(surface_selector);

  // Configuration of the dumper
  coupler.setBaseName("BM_0");
  coupler.addDumpFieldVector("displacement");
  coupler.addDumpFieldVector("velocity");
  coupler.addDumpFieldVector("normals");
  coupler.addDumpFieldVector("tangents");
  coupler.addDumpFieldVector("contact_force");
  coupler.addDumpFieldVector("external_force");
  coupler.addDumpFieldVector("internal_force");
  coupler.addDumpField("areas");
  coupler.addDumpField("stress");
  coupler.addDumpField("blocked_dofs");

  // Add the boundary conditions
  solid.applyBC(BC::Dirichlet::FixedValue(0.0, _x), "XFixed");
  solid.applyBC(BC::Dirichlet::FixedValue(0.0, _y), "YFixed");
  solid.applyBC(BC::Dirichlet::FixedValue(0.0, _x), "loading");
  solid.applyBC(BC::Dirichlet::FixedValue(0.0, _y), "loading");

  // Register velocity and gaps for future damping
  auto &velocity = solid.getVelocity();
  auto &gaps = contact.getGaps();

  auto damp_contact_zone = [&]() {
    for (auto &&tuple : zip(gaps, make_view(velocity, 2))) {
      auto &gap = std::get<0>(tuple);
      auto &vel = std::get<1>(tuple);
      if (gap > 0) {
        vel *= 0.99;
      }
    }
  };

  auto dump_step = [&](const std::string &stage, int s) {
    if (s % 100 == 0) {
      std::cout << stage << " step " << s << "\t\r" << std::flush;
      coupler.dump();
    }
  };

  // Dump the initial state
  coupler.dump();

  const int loading_steps = 2000;
  const Real final_compression = -1.0;

  // First loop: ramped normal loading of the upper solid onto the lower solid
  for (int s = 0; s < loading_steps; ++s) {
    const Real increment = final_compression / loading_steps;

    solid.applyBC(BC::Dirichlet::IncrementValue(increment, _y), "loading");

    coupler.solveStep();
    damp_contact_zone();
    dump_step("Loading", s);
  }

  std::cout << std::endl << "Ramped loading done !" << std::endl;

  const int speed_ramp_steps = 2000;
  const int sliding_steps = 4000;
  const Real final_sliding_speed = 1.0;

  // Second loop: ramped tangential speed, imposed as dx = v(t) * dt
  for (int s = 0; s < speed_ramp_steps; ++s) {
    const Real ramp = static_cast<Real>(s + 1) / speed_ramp_steps;
    const Real imposed_speed = ramp * final_sliding_speed;

    solid.applyBC(BC::Dirichlet::IncrementValue(imposed_speed * time_step, _x),
                  "loading");

    coupler.solveStep();
    damp_contact_zone();
    dump_step("Speed ramp", s);
  }

  std::cout << std::endl << "Ramped speed reached !" << std::endl;

  // Third loop: keep sliding at the final prescribed speed
  for (int s = 0; s < sliding_steps; ++s) {
    solid.applyBC(
        BC::Dirichlet::IncrementValue(final_sliding_speed * time_step, _x),
        "loading");

    coupler.solveStep();
    damp_contact_zone();
    dump_step("Sliding", s);
  }

  std::cout << std::endl << "BM_0 done !" << std::endl;

  return 0;
}
