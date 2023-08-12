#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <xsimd/xsimd.hpp>

namespace py = pybind11;
constexpr double FOUR_THIRDS_PI = 4.1887902;
constexpr double N_THREE_HALVES_SQRT_3 = -2.59807621;
constexpr double TWO_OVER_SQRT_THREE = 1.154700538;

Eigen::ArrayXXd ge_41rt_inverse_distortion(const Eigen::ArrayXXd& inputs, const double rhoMax, const Eigen::ArrayXd& params) {
  Eigen::ArrayXd radii = inputs.matrix().rowwise().norm();
  Eigen::ArrayXd inverted_radii = radii.cwiseInverse();
  Eigen::ArrayXd cosines = inputs.col(0) * inverted_radii;
  Eigen::ArrayXd cosine_double_angles = 2*cosines.square() - 1;
  Eigen::ArrayXd cosine_quadruple_angles = 2*cosine_double_angles.square() - 1;
  Eigen::ArrayXd sqrt_p_is = rhoMax / (-params[0]*cosine_double_angles - params[1]*cosine_quadruple_angles - params[2]).sqrt();
  Eigen::ArrayXd solutions = TWO_OVER_SQRT_THREE*sqrt_p_is*(acos(N_THREE_HALVES_SQRT_3*radii/sqrt_p_is)/3 + FOUR_THIRDS_PI).cos();
  Eigen::ArrayXXd results = solutions.rowwise().replicate(inputs.cols()).array() * inputs * inverted_radii.rowwise().replicate(inputs.cols()).array();

  return results;
}

PYBIND11_MODULE(inverse_distortion, m)
{
  m.doc() = "HEXRD inverse distribution plugin";

  m.def("ge_41rt_inverse_distortion", &ge_41rt_inverse_distortion, "Inverse distortion for ge_41rt");
}