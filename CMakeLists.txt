cmake_minimum_required(VERSION 3.16)

project(MyAkantuPlayground)


list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message(STATUS "CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")
find_package(GMSH REQUIRED MODULE)
find_package(Akantu REQUIRED)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ras_ss_sw.in
    ${CMAKE_CURRENT_BINARY_DIR}/ras_ss_sw.in
    COPYONLY
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ras_ss.in
    ${CMAKE_CURRENT_BINARY_DIR}/ras_ss.in
    COPYONLY
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ras_ss_coulomb.in
    ${CMAKE_CURRENT_BINARY_DIR}/ras_ss_coulomb.in
    COPYONLY
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ras_ss_coulomb_ve.in
    ${CMAKE_CURRENT_BINARY_DIR}/ras_ss_coulomb_ve.in
    COPYONLY
)
if(NOT COMMAND add_mesh)
  message(FATAL_ERROR "add_mesh was not defined")
endif()

add_akantu_simulation(bloc_friction test.cc)
add_akantu_simulation(BM_0 BM_0.cc)
add_akantu_simulation(steady_state steady_state.cc)
add_akantu_simulation(steady_state_sw steady_state_slipweakening.cc)
add_akantu_simulation(steady_state_sw_local steady_state_SW_local.cc)

add_mesh(bloc bloc.geo 2 1)
add_dependencies(bloc_friction bloc)

add_mesh(ntn_test_100 ntn_test_100.geo 2 1)
add_mesh(ntn_test_200 ntn_test_200.geo 2 1)
add_mesh(ntn_test_500 ntn_test_500.geo 2 1)
add_mesh(ntn_test_1000 ntn_test_1000.geo 2 1)
add_dependencies(steady_state ntn_test_100 ntn_test_200 ntn_test_500 ntn_test_1000)
add_dependencies(steady_state_sw_local ntn_test_100 ntn_test_200 ntn_test_500 ntn_test_1000)

add_akantu_simulation(steady_state_Method1 steady_state_Method1.cc)
