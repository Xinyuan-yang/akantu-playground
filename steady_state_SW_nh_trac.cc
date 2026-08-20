// Traction-driven periodic, no-healing slip-weakening simulation.
// The shared implementation keeps this case mechanically identical to
// steady_state_SW_nh_peri except that tangential loading is a Neumann
// traction, not a prescribed boundary velocity.
#define AKANTU_TRACTION_DRIVEN
#include "steady_state_SW_nh_peri.cc"
