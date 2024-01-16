#include <pybind11/pybind11.h>
#include <xtensor/xarray.hpp>

namespace py = pybind11;
constexpr double FOUR_THIRDS_PI = 4.1887902;
constexpr double N_THREE_HALVES_SQRT_3 = -2.59807621;
constexpr double TWO_OVER_SQRT_THREE = 1.154700538;

void ge_41rt_inverse_distortion() {
  return;
}

PYBIND11_MODULE(inverse_distortion, m)
{
  m.doc() = "HEXRD inverse distribution plugin";

  m.def("ge_41rt_inverse_distortion", &ge_41rt_inverse_distortion, "Inverse distortion for ge_41rt");
}
