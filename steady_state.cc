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
#include "ntn_contact_solver.hh"
#include "ntn_contact_solvercallback.hh"
#include "ntn_initiation_function.hh"

using namespace akantu;

namespace akantu {
namespace {

auto registerSelectableFrictionLaws() -> bool {
  registerFriction<NTNFricLawCoulomb>("coulomb");
  registerFriction<NTNFricLawLinearSlipWeakening>("linear_slip_weakening");
  registerFriction<NTNFricLawLinearSlipWeakeningNoHealing>(
      "linear_slip_weakening_no_healing");
  registerFriction<NTNFricLawLinearCohesive>("linear_cohesive");
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
int main(int /*argc*/, char * /*argv*/[]) {

  // TODO read this from input file
  std::stringstream output_folder;
  output_folder << "steady_state";

  getStaticParser().parse("ras_ss.in");
  const ParserSection & data = getUserParser();
  UInt spatial_dimension = data.getParameter("spatial_dimension");
  UInt dump_every = data.getParameter("dump_every");
  std::unique_ptr<Mesh> mesh;
  std::unique_ptr<SolidMechanicsModel> model;
  std::unique_ptr<NTNContactSolverCallback> solver_ntn;
  mesh = std::make_unique<Mesh>(spatial_dimension);
  mesh->read("ntn_test_ras.msh");

  mesh->makePeriodic(_x, "slider_left", "slider_right");
  mesh->makePeriodic(_x, "base_left", "base_right");

  model = std::make_unique<SolidMechanicsModel>(*mesh);

  Real time_step_factor = data.getParameter("time_step_factor");

  Int normal_dir = 1;

  solver_ntn = std::make_unique<NTNContactSolverCallback>(
      *model, "slider_bottom", "base_top", normal_dir, time_step_factor);

  const auto & mat = model->getMaterial("slider");

  Real shear_vel = data.getParameter("shear_velocity");
  Vector<Real> trac_top = data.getParameter("top_traction");
  Vector<Real> trac_bottom = data.getParameter("bot_traction");

  model->setBaseName(output_folder.str());
  model->addDumpField("blocked_dofs");
  model->addDumpField("mass");
  model->addDumpFieldVector("velocity");
  model->addDumpFieldVector("acceleration");
  model->addDumpFieldVector("displacement");
  model->addDumpFieldVector("internal_force");
  model->addDumpFieldVector("external_force");

  // Static analytical solution
  Real fss = data.getParameter("fss");
  Real E = mat.getParam("E");
  Real nu = mat.getParam("nu");
  Real shear_modulus = E / (2. * (1. + nu));
  Real normal_strain_applied = trac_top(1) / E - nu * nu * trac_top(1) / E;

  Array<Real> & displacement = model->getDisplacement();
  Array<Real> & position = mesh->getNodes();
  UInt nb_nodes = model->getFEEngine().getMesh().getNbNodes();

  for (UInt n = 0; n < nb_nodes; ++n) {
    displacement(n, 0) = fss * -trac_top(1) / (shear_modulus)*position(n, 1);
    displacement(n, 1) = normal_strain_applied * position(n, 1);
  }

  // Set boundary conditions for dynamic simulation
  model->applyBC(BC::Neumann::FromTraction(trac_top), "slider_top");
  model->applyBC(BC::Neumann::FromTraction(trac_bottom), "base_bottom");

  ///// Set to steady state
  const auto & slider_nodes =
      mesh->getElementGroup("slider").getNodeGroup().getNodes();
  const auto & base_nodes =
      mesh->getElementGroup("base").getNodeGroup().getNodes();

  // Specify initial nodal velocity
  auto & velo = model->getVelocity();
  auto & increment = model->getIncrement();
  auto friction = solver_ntn->getFriction();
  auto dt = model->getTimeStep();

  for (auto n : slider_nodes) {
    velo(n, _x) = 0.5 * shear_vel;
    increment(n, _x) = 0.5 * shear_vel * dt;
  }
  for (auto n : base_nodes) {
    velo(n, _x) = -0.5 * shear_vel;
    increment(n, _x) = -0.5 * shear_vel * dt;
  }

  auto contact = solver_ntn->getContact();

  auto vel_it =
      make_view(model->getVelocity(), model->getSpatialDimension()).begin();
  auto & slip_velocity = friction->getSlipVelocity();
  auto & slip_velocity_norm = friction->getSlipVelocityNorm();
  auto & is_sticking = friction->getIsSticking();
  Real phi = friction->get("friction_state");

  for (auto && [n, master, slave, slip_vel, slip_vel_n, is_sticking] :
       enumerate(contact->getMasters(), contact->getSlaves(),
                 make_view(slip_velocity, slip_velocity.getNbComponent()),
                 slip_velocity_norm, is_sticking)) {

    is_sticking = false;

    friction->updateFrictionState(n, phi);
  }
  //////

  // Time of the simulation
  Real stable_time_step = model->getStableTimeStep();
  Real time_step = stable_time_step * time_step_factor;
  model->setTimeStep(time_step);
  UInt nb_steps = data.getParameter("nb_steps");

  model->dump();

  std::ofstream energies;
  auto file_name = std::filesystem::path(output_folder.str());
  file_name.replace_extension("csv");
  file_name = std::string("friction-energies-") + file_name.string();
  energies.open(file_name.c_str(), std::ofstream::out | std::ofstream::trunc);

  energies << "time,ekin,epot,work,econ,efri,tot" << std::endl;

  auto einit = 0.;

  for (UInt s = 0; s < nb_steps; ++s) {
    // Apply velocity
    UInt nb_nodes = model->getFEEngine().getMesh().getNbNodes();
    Array<Real> & position = mesh->getNodes();
    Array<Real> & velo = model->getVelocity();
    const Vector<Real> & upperBounds = mesh->getUpperBounds();
    const Vector<Real> & lowerBounds = mesh->getLowerBounds();
    Real top = upperBounds(1);
    Real bottom = lowerBounds(1);
    Real stable_time_step = model->getStableTimeStep();
    Real time_step = stable_time_step * time_step_factor;
    Real disp_incr = shear_vel * time_step;
    Array<Real> & displacement = model->getDisplacement();
    Array<bool> & blocked = model->getBlockedDOFs();
    for (UInt n = 0; n < nb_nodes; ++n) {
      if (std::abs(position(n, 1) - top) < 1e-6) {
        velo(n, _x) = 0.5 * shear_vel;
      }
      if (std::abs(position(n, 1) - bottom) < 1e-6) {
        velo(n, _x) = -0.5 * shear_vel;
      }
    }

    for (UInt n = 0; n < nb_nodes; ++n) {
      if (std::abs(position(n, 1) - top) < 1e-6) {
        displacement(n, 0) += 0.5 * disp_incr;
        blocked(n, 0) = true;
      }
      if (std::abs(position(n, 1) - bottom) < 1e-6) {
        displacement(n, 0) += -0.5 * disp_incr;
        blocked(n, 0) = true;
      }
    }

    model->solveStep(*solver_ntn, "explicit_lumped");

    auto ekin = model->getEnergy("kinetic");
    auto epot = model->getEnergy("potential");
    auto work = model->getEnergy("external work new");
    auto econ = solver_ntn->getExternalWork();
    if (s == 0) {
      einit = ekin + epot - (work + econ[0] + econ[1]);
    }
    energies << s * time_step << "," << ekin << "," << epot << "," << work
             << "," << econ[0] << "," << econ[1] << ","
             << ekin + epot - (work + econ[0] + econ[1]) - einit << std::endl;

    if (s % dump_every == 0) {
      model->dump();
    }

    if (s == 10) {
      Real left = lowerBounds(0);
      Real right = upperBounds(0);
      Real contact_l = right - left;
      Real value;
      Real amplitude = 0.01;
      auto & cur_pos = contact->getModel().getCurrentPosition();
      auto & slaves = contact->getSlaves();

      auto & friction_state = friction->getState();
      UInt nb_contact_nodes = contact->getNbContactNodes();
      for (UInt n = 0; n < nb_contact_nodes; ++n) {
        UInt slave = slaves(n);
        value = friction_state(n) *
                (1 + amplitude * sin(2 * M_PI / contact_l * cur_pos(slave)));

        friction->updateFrictionState(n, value);
      }
    }
  }

  return EXIT_SUCCESS;
}

  return true;
}

const bool selectable_friction_laws_are_registered [[maybe_unused]] =
    registerSelectableFrictionLaws();

auto initializeSelectedNTNFriction(NTNBaseContact &contact,
                                   const ID &selected_friction_law,
                                   const ID &selected_friction_option = "")
    -> std::shared_ptr<NTNBaseFriction> {
  const ParserSection *selected_section = nullptr;
  std::stringstream available_laws;

  auto sub_sections = getStaticParser().getSubSections(ParserType::_friction);
  for (auto it = sub_sections.first; it != sub_sections.second; ++it) {
    const ParserSection &section = *it;
    const auto option = section.getOption();

    available_laws << "\n  friction " << section.getName();
    if (not option.empty()) {
      available_laws << " " << option;
    }

    if (section.getName() != selected_friction_law) {
      continue;
    }

    if (not selected_friction_option.empty() and
        option != selected_friction_option) {
      continue;
    }

    selected_section = &section;
    break;
  }

  if (selected_section == nullptr) {
    AKANTU_EXCEPTION("Could not find selected friction law `"
                     << selected_friction_law << "`"
                     << (selected_friction_option.empty()
                             ? ""
                             : " with option `" + selected_friction_option +
                                   "`")
                     << ". Available laws are:" << available_laws.str());
  }

  const auto friction_law = selected_section->getName();
  const auto friction_reg = selected_section->getOption("no_regularisation");
  auto friction =
      NTNFactory::getInstance().allocate(friction_law, friction_reg, contact);
  friction->parseSection(*selected_section);

  std::cout << "Selected friction law: friction " << friction_law;
  if (not friction_reg.empty()) {
    std::cout << " " << friction_reg;
  }
  std::cout << std::endl;

  return std::shared_ptr<NTNBaseFriction>(std::move(friction));
}

class SelectableNTNContactSolverCallback : public InterceptSolverCallback {
public:
  SelectableNTNContactSolverCallback(SolidMechanicsModel &solid,
                                     const ID &slave, const ID &master,
                                     Int surface_normal_dir,
                                     Real time_step_factor,
                                     const ID &selected_friction_law,
                                     const ID &selected_friction_option = "")
      : InterceptSolverCallback(solid), solid(solid) {
    contact = std::make_shared<NTNContact>(solid);

    solid.initFull(SolidMechanicsModelOptions(_explicit_lumped_mass));
    solid.assembleMassLumped();

    contact->addSurfacePair(slave, master, surface_normal_dir);
    contact->updateNormals();
    contact->updateLumpedBoundary();

    Real stable_time_step = solid.getStableTimeStep();
    Real time_step = stable_time_step * time_step_factor;
    solid.setTimeStep(time_step);
    contact->updateImpedance();

    friction = initializeSelectedNTNFriction(
        *contact, selected_friction_law, selected_friction_option);
  }

  auto getFriction() const -> std::shared_ptr<NTNBaseFriction> {
    return friction;
  }

  auto getContact() const -> std::shared_ptr<NTNContact> { return contact; }

  [[nodiscard]] auto getExternalWorkIncrement() const -> Vector<Real, 2> {
    Vector<Real, 2> work{0., 0.};
    const auto &contact_pressure = contact->getGlobalContactPressure();
    const auto &friction_traction = friction->getGlobalFrictionTraction();
    const auto &mesh = solid.getMesh();
    auto spatial_dimension = mesh.getSpatialDimension();

    for (auto &&[n, inc, f_cont, f_fric] :
         enumerate(make_view(solid.getIncrement(), spatial_dimension),
                   make_view(contact_pressure, spatial_dimension),
                   make_view(friction_traction, spatial_dimension))) {
      auto is_local_node = mesh.isLocalOrMasterNode(n);
      auto is_not_pbc_slave_node = not mesh.isPeriodicSlave(n);
      Real count_node = is_local_node && is_not_pbc_slave_node ? 1. : 0.;

      work[0] -= count_node * f_cont.dot(inc);
      work[1] -= count_node * f_fric.dot(inc);
    }

    return work;
  }

  [[nodiscard]] auto getExternalWork() const -> Vector<Real, 2> {
    Vector<Real, 2> work = external_work;
    solid.getMesh().getCommunicator().allReduce(work,
                                                SynchronizerOperation::_sum);
    return work;
  }

  void assembleResidual() override {
    auto &&dof_manager = solid.getDOFManager();

    auto &contact_pressure = contact->getGlobalContactPressure();
    auto &friction_traction = friction->getGlobalFrictionTraction();

    if (external_work_release != solid.getDisplacementRelease()) {
      external_work += this->getExternalWorkIncrement();
      external_work_release = solid.getDisplacementRelease();
    }

    aka::as_type<SolverCallback>(solid).assembleResidual();

    contact->computeContactPressure();
    friction->updateSlip();
    friction->computeFrictionTraction();

    contact->assembleGlobalContactPressure();
    friction->assembleGlobalFrictionTraction();

    dof_manager.assembleToResidual("displacement", contact_pressure, 1);
    dof_manager.assembleToResidual("displacement", friction_traction, 1);
  }

  void beforeSolveStep() override {
    aka::as_type<SolverCallback>(solid).beforeSolveStep();

    friction->savePreviousState();
    contact->savePreviousState();
  }

  void afterSolveStep(bool converged = true) override {
    aka::as_type<SolverCallback>(solid).afterSolveStep(converged);

    if (not converged) {
      friction->restorePreviousState();
      contact->restorePreviousState();
    }
  }

private:
  SolidMechanicsModel &solid;
  std::shared_ptr<NTNContact> contact;
  std::shared_ptr<NTNBaseFriction> friction;
  Vector<Real, 2> external_work{0., 0.};
  Int external_work_release{-1};
};

} // namespace
} // namespace akantu

/* ------------------------------------------------------------------------ */
/* Main                                                                     */
/* ------------------------------------------------------------------------ */
int main(int /*argc*/, char * /*argv*/[]) {

  // TODO read this from input file
  std::stringstream output_folder;
  output_folder << "steady_state";

  getStaticParser().parse("ras_ss.in");
  const ParserSection & data = getUserParser();
  UInt spatial_dimension = data.getParameter("spatial_dimension");
  UInt dump_every = data.getParameter("dump_every");
  std::unique_ptr<Mesh> mesh;
  std::unique_ptr<SolidMechanicsModel> model;
  std::unique_ptr<SelectableNTNContactSolverCallback> solver_ntn;
  mesh = std::make_unique<Mesh>(spatial_dimension);
  mesh->read("ntn_test_ras.msh");

  //Periodic BC switch here
  //mesh->makePeriodic(_x, "slider_left", "slider_right");
  //mesh->makePeriodic(_x, "base_left", "base_right");

  model = std::make_unique<SolidMechanicsModel>(*mesh);

  Real time_step_factor = data.getParameter("time_step_factor");

  Int normal_dir = 1;

  const ID selected_friction_law =
      "rate_and_state:revised_dieterich:ageing_regularized";
  const ID selected_friction_option = "";

  solver_ntn = std::make_unique<SelectableNTNContactSolverCallback>(
      *model, "slider_bottom", "base_top", normal_dir, time_step_factor,
      selected_friction_law, selected_friction_option);

  const auto & mat = model->getMaterial("slider");

  Real shear_vel = data.getParameter("shear_velocity");
  Vector<Real> trac_top = data.getParameter("top_traction");
  Vector<Real> trac_bottom = data.getParameter("bot_traction");

  model->setBaseName(output_folder.str());
  model->addDumpField("blocked_dofs");
  model->addDumpField("mass");
  model->addDumpFieldVector("velocity");
  model->addDumpFieldVector("acceleration");
  model->addDumpFieldVector("displacement");
  model->addDumpFieldVector("internal_force");
  model->addDumpFieldVector("external_force");

  // Static analytical solution
  Real fss = data.getParameter("fss");
  Real E = mat.getParam("E");
  Real nu = mat.getParam("nu");
  Real shear_modulus = E / (2. * (1. + nu));
  Real normal_strain_applied = trac_top(1) / E - nu * nu * trac_top(1) / E;

  Array<Real> & displacement = model->getDisplacement();
  Array<Real> & position = mesh->getNodes();
  UInt nb_nodes = model->getFEEngine().getMesh().getNbNodes();

  for (UInt n = 0; n < nb_nodes; ++n) {
    displacement(n, 0) = fss * -trac_top(1) / (shear_modulus)*position(n, 1);
    displacement(n, 1) = normal_strain_applied * position(n, 1);
  }

  // Set boundary conditions for dynamic simulation
  model->applyBC(BC::Neumann::FromTraction(trac_top), "slider_top");
  model->applyBC(BC::Neumann::FromTraction(trac_bottom), "base_bottom");

  ///// Set to steady state
  const auto & slider_nodes =
      mesh->getElementGroup("slider").getNodeGroup().getNodes();
  const auto & base_nodes =
      mesh->getElementGroup("base").getNodeGroup().getNodes();

  // Specify initial nodal velocity
  auto & velo = model->getVelocity();
  auto & increment = model->getIncrement();
  auto friction = solver_ntn->getFriction();
  auto dt = model->getTimeStep();

  for (auto n : slider_nodes) {
    velo(n, _x) = 0.5 * shear_vel;
    increment(n, _x) = 0.5 * shear_vel * dt;
  }
  for (auto n : base_nodes) {
    velo(n, _x) = -0.5 * shear_vel;
    increment(n, _x) = -0.5 * shear_vel * dt;
  }

  auto contact = solver_ntn->getContact();

  auto vel_it =
      make_view(model->getVelocity(), model->getSpatialDimension()).begin();
  auto & slip_velocity = friction->getSlipVelocity();
  auto & slip_velocity_norm = friction->getSlipVelocityNorm();
  auto & is_sticking = friction->getIsSticking();
  Real phi = friction->get("friction_state");

  for (auto && [n, master, slave, slip_vel, slip_vel_n, is_sticking] :
       enumerate(contact->getMasters(), contact->getSlaves(),
                 make_view(slip_velocity, slip_velocity.getNbComponent()),
                 slip_velocity_norm, is_sticking)) {

    is_sticking = false;

    friction->updateFrictionState(n, phi);
  }
  //////

  // Time of the simulation
  Real stable_time_step = model->getStableTimeStep();
  Real time_step = stable_time_step * time_step_factor;
  model->setTimeStep(time_step);
  UInt nb_steps = data.getParameter("nb_steps");

  model->dump();

  std::ofstream energies;
  auto file_name = std::filesystem::path(output_folder.str());
  file_name.replace_extension("csv");
  file_name = std::string("friction-energies-") + file_name.string();
  energies.open(file_name.c_str(), std::ofstream::out | std::ofstream::trunc);

  energies << "time,ekin,epot,work,econ,efri,tot" << std::endl;

  auto einit = 0.;

  for (UInt s = 0; s < nb_steps; ++s) {
    // Apply velocity
    UInt nb_nodes = model->getFEEngine().getMesh().getNbNodes();
    Array<Real> & position = mesh->getNodes();
    Array<Real> & velo = model->getVelocity();
    const Vector<Real> & upperBounds = mesh->getUpperBounds();
    const Vector<Real> & lowerBounds = mesh->getLowerBounds();
    Real top = upperBounds(1);
    Real bottom = lowerBounds(1);
    Real stable_time_step = model->getStableTimeStep();
    Real time_step = stable_time_step * time_step_factor;
    Real disp_incr = shear_vel * time_step;
    Array<Real> & displacement = model->getDisplacement();
    Array<bool> & blocked = model->getBlockedDOFs();
    for (UInt n = 0; n < nb_nodes; ++n) {
      if (std::abs(position(n, 1) - top) < 1e-6) {
        velo(n, _x) = 0.5 * shear_vel;
      }
      if (std::abs(position(n, 1) - bottom) < 1e-6) {
        velo(n, _x) = -0.5 * shear_vel;
      }
    }

    for (UInt n = 0; n < nb_nodes; ++n) {
      if (std::abs(position(n, 1) - top) < 1e-6) {
        displacement(n, 0) += 0.5 * disp_incr;
        blocked(n, 0) = true;
      }
      if (std::abs(position(n, 1) - bottom) < 1e-6) {
        displacement(n, 0) += -0.5 * disp_incr;
        blocked(n, 0) = true;
      }
    }

    model->solveStep(*solver_ntn, "explicit_lumped");

    auto ekin = model->getEnergy("kinetic");
    auto epot = model->getEnergy("potential");
    auto work = model->getEnergy("external work new");
    auto econ = solver_ntn->getExternalWork();
    if (s == 0) {
      einit = ekin + epot - (work + econ[0] + econ[1]);
    }
    energies << s * time_step << "," << ekin << "," << epot << "," << work
             << "," << econ[0] << "," << econ[1] << ","
             << ekin + epot - (work + econ[0] + econ[1]) - einit << std::endl;

    if (s % dump_every == 0) {
      model->dump();
    }

    if (s == 10) {
      Real left = lowerBounds(0);
      Real right = upperBounds(0);
      Real contact_l = right - left;
      Real value;
      Real amplitude = 0.01;
      auto & cur_pos = contact->getModel().getCurrentPosition();
      auto & slaves = contact->getSlaves();

      auto & friction_state = friction->getState();
      UInt nb_contact_nodes = contact->getNbContactNodes();
      for (UInt n = 0; n < nb_contact_nodes; ++n) {
        UInt slave = slaves(n);
        value = friction_state(n) *
                (1 + amplitude * sin(2 * M_PI / contact_l * cur_pos(slave)));

        friction->updateFrictionState(n, value);
      }
    }
  }

  return EXIT_SUCCESS;
}
