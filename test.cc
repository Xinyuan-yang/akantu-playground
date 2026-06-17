/**
 * Copyright (©) 2013-2023 EPFL (Ecole Polytechnique Fédérale de Lausanne)
 * Laboratory (LSMS - Laboratoire de Simulation en Mécanique des Solides)
 *
 * This file is part of Akantu
 *
 * Akantu is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * Akantu is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Akantu. If not, see <http://www.gnu.org/licenses/>.
 */

/* -------------------------------------------------------------------------- */
#include "coupler_solid_contact.hh"
#include "non_linear_solver.hh"
#include <fstream>
#include <iomanip>
#include <vector>
/* -------------------------------------------------------------------------- */
using namespace akantu;
/* -------------------------------------------------------------------------- */

int main(int argc, char *argv[])
{

  // Initialize the material database
  initialize("material.dat", argc, argv);

  // Create the mesh>
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

  // Setup the contact
  auto &&surface_selector = std::make_shared<PhysicalSurfaceSelector>(mesh);
  contact.getContactDetector().setSurfaceSelector(surface_selector);

  // Configuration of the dumper
  coupler.setBaseName("bloc_friction_test1");
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
  solid.applyBC(BC::Dirichlet::FixedValue(0.0, _y), "loading"); // see the bloc.geo for the definition of the "loading" physical group

  // Register velocity and gaps for future damping
  auto &velocity = solid.getVelocity();
  auto &gaps = contact.getGaps();

  // Dump the initial state
  coupler.dump();

  // First loop : compression of the bloc against the wall
  for (int s = 0; s < 2000; s++)
  {
    solid.applyBC(BC::Dirichlet::IncrementValue(-1.0 / 2000, _y), "loading");

    coupler.solveStep();

    // damping velocities only along the contacting zone
    for (auto &&tuple : zip(gaps, make_view(velocity, 2)))
    {
      auto &gap = std::get<0>(tuple);
      auto &vel = std::get<1>(tuple);
      if (gap > 0)
      {
        vel *= 0.99;
      }
    }

    if (s % 100 == 0)
    {
      std::cout << "Step " << s << "\t\r" << std::flush;
      coupler.dump();
    }
  }

  std::cout << "Compression done !" << std::endl;

  // Second big loop : sliding of the bloc against the wall, stick-slip repeating 5 times
  for (int cycle = 0; cycle < 5; ++cycle)
  {
    for (int s = 0; s < 2000; s++)
    {
      solid.applyBC(BC::Dirichlet::IncrementValue(2.0 / 2000, _x), "loading");

      coupler.solveStep();

      // damping velocities only along the contacting zone
      for (auto &&tuple : zip(gaps, make_view(velocity, 2)))
      {
        auto &gap = std::get<0>(tuple);
        auto &vel = std::get<1>(tuple);
        if (gap > 0)
        {
          vel *= 0.99;
        }
      }

      if (s % 100 == 0)
      {
        std::cout << "Step " << s << "\t\r" << std::flush;
        coupler.dump();
      }
    }

    std::cout << "Sliding done !" << std::endl;

    // Third loop : stabilization of the system (no external force)
    for (int s = 0; s < 2000; s++)
    {
      coupler.solveStep();

      // damping velocities only along the contacting zone
      for (auto &&tuple : zip(gaps, make_view(velocity, 2)))
      {
        auto &gap = std::get<0>(tuple);
        auto &vel = std::get<1>(tuple);
        if (gap > 0)
        {
          vel *= 0.99;
        }
      }

      if (s % 100 == 0)
      {
        std::cout << "Step " << s << "\t\r" << std::flush;
        coupler.dump();
      }
    }
  }

  return 0;
}