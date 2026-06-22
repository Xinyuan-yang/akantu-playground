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

set(MU_VALUES 0.10 0.15 0.20 0.30 0.33)

foreach(MU ${MU_VALUES})

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/ras_ss_coulomb_${MU}.in
        ${CMAKE_CURRENT_BINARY_DIR}/ras_ss_coulomb_${MU}.in
        COPYONLY
    )

endforeach()

if(NOT COMMAND add_mesh)
  message(FATAL_ERROR "add_mesh was not defined")
endif()

add_akantu_simulation(bloc_friction test.cc)
add_akantu_simulation(BM_0 BM_0.cc)
add_akantu_simulation(steady_state steady_state.cc)
add_akantu_simulation(steady_state_sw steady_state_slipweakening.cc)

add_mesh(bloc bloc.geo 2 1)
add_dependencies(bloc_friction bloc)

add_mesh(ntn_test_ras ntn_test_ras.geo 2 1)
add_dependencies(steady_state ntn_test_ras)
add_dependencies(steady_state_sw ntn_test_ras)