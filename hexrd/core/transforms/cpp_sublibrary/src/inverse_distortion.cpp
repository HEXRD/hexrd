#define _USE_MATH_DEFINES
#include <Eigen/Core>
#include <cmath>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <xsimd/xsimd.hpp>

namespace py = pybind11;
const double FOUR_THIRDS_PI = M_PI * 4.0 / 3.0;
const double N_THREE_HALVES_SQRT_3 = -3.0 / 2.0 * std::sqrt(3.0);
const double TWO_OVER_SQRT_THREE = 2.0 / std::sqrt(3.0);

Eigen::ArrayXXd ge_41rt_inverse_distortion(const Eigen::ArrayXXd &inputs,
                                           const double rhoMax,
                                           const Eigen::ArrayXd &params) {
  Eigen::ArrayXd radii = inputs.matrix().rowwise().norm();
  Eigen::ArrayXd inverted_radii = radii.cwiseInverse();
  Eigen::ArrayXd cosines = inputs.col(0) * inverted_radii;
  Eigen::ArrayXd cosine_double_angles = 2 * cosines.square() - 1;
  Eigen::ArrayXd cosine_quadruple_angles =
      2 * cosine_double_angles.square() - 1;
  Eigen::ArrayXd sqrt_p_is =
      rhoMax / (-params[0] * cosine_double_angles -
                params[1] * cosine_quadruple_angles - params[2])
                   .sqrt();
  Eigen::ArrayXd solutions =
      TWO_OVER_SQRT_THREE * sqrt_p_is *
      (acos(N_THREE_HALVES_SQRT_3 * radii / sqrt_p_is) / 3 + FOUR_THIRDS_PI)
          .cos();
  Eigen::ArrayXXd results =
      solutions.rowwise().replicate(inputs.cols()).array() * inputs *
      inverted_radii.rowwise().replicate(inputs.cols()).array();

  return results;
}

PYBIND11_MODULE(inverse_distortion, m) {
  m.doc() = "HEXRD inverse distribution plugin";

  m.def("ge_41rt_inverse_distortion", &ge_41rt_inverse_distortion,
        "Inverse distortion for ge_41rt");
}