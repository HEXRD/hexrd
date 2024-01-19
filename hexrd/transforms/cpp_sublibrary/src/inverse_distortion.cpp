#define FORCE_IMPORT_ARRAY

#include <pybind11/pybind11.h>
#include <xtensor/xmath.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor-python/pyarray.hpp>
#include <xtensor/xview.hpp>

namespace py = pybind11;
constexpr double FOUR_THIRDS_PI = 4.1887902;
constexpr double N_THREE_HALVES_SQRT_3 = -2.59807621;
constexpr double TWO_OVER_SQRT_THREE = 1.154700538;

xt::pyarray<double> ge_41rt_inverse_distortion(const xt::pyarray<double>& inputs, const double rhoMax, const xt::pyarray<double>& params) {
    auto radii = xt::sqrt(xt::sum(xt::square(inputs), {1}));

    if (xt::amax(radii)() < 1e-10) {
        return xt::zeros<double>({inputs.shape()[0], inputs.shape()[1]});
    }

    auto inverted_radii = xt::pow(radii, -1);
    xt::pyarray<double> cosines = xt::view(inputs, xt::all(), 0) * inverted_radii;
    auto cosine_double_angles = 2 * xt::square(cosines) - 1;
    auto cosine_quadruple_angles = 2 * xt::square(cosine_double_angles) - 1;
    auto sqrt_p_is = rhoMax / xt::sqrt(-params[0] * cosine_double_angles - params[1] * cosine_quadruple_angles - params[2]);
    auto solutions = TWO_OVER_SQRT_THREE * sqrt_p_is * xt::cos(xt::acos(N_THREE_HALVES_SQRT_3 * radii / sqrt_p_is) / 3 + FOUR_THIRDS_PI);
    xt::pyarray<double> results = xt::view(solutions, xt::all(), xt::newaxis()) * inputs * xt::view(inverted_radii, xt::all(), xt::newaxis());

    return results;
}

PYBIND11_MODULE(inverse_distortion, m)
{
    xt::import_numpy();
    m.doc() = "HEXRD inverse distribution plugin";

    m.def("ge_41rt_inverse_distortion", &ge_41rt_inverse_distortion, "Inverse distortion for ge_41rt");
}
