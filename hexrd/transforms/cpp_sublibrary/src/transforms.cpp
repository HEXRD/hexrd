#define FORCE_IMPORT_ARRAY
#define XTENSOR_USE_XSIMD

#include <pybind11/pybind11.h>
#include <xtensor/xmath.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor-python/pyarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xmanipulation.hpp>
#include <iostream>

namespace py = pybind11;

const xt::xarray<double> Zl {0.0, 0.0, 1.0};
const float eps = std::numeric_limits<float>::epsilon();
const float sqrtEps = std::sqrt(std::numeric_limits<float>::epsilon());

const static xt::xarray<double> makeBinaryRotMat(const xt::xarray<double>& a) {
  xt::xarray<double> r = 2.0 * xt::linalg::outer(a, xt::transpose(a));
  r(0, 0) -= 1;
  r(1, 1) -= 1;
  r(2, 2) -= 1;
  return r;
}

const static xt::xarray<double> gvecToDetectorXYOne(
                                    const xt::xarray<double>& gVec_c, // 3d vector
                                    const xt::xarray<double>& rMat_d, // 3x3 matrix
                                    const xt::xarray<double>& rMat_sc, // 3x3 matrix
                                    const xt::xarray<double>& tVec_d, // 3d vector
                                    const xt::xarray<double>& bHat_l, // 3d vector
                                    const xt::xarray<double>& nVec_l, // 3d vector
                                    float num,
                                    const xt::xarray<double>& P0_l)
{
  const xt::xarray<double> gHat_c = gVec_c / xt::linalg::norm(gVec_c);
  const xt::xarray<double> gVec_l = rMat_sc * gHat_c;
  const double bDot = -xt::linalg::vdot(bHat_l, gVec_l);

  if (bDot >= eps && bDot <= 1.0 - eps) {
    const xt::xarray<double> brMat = makeBinaryRotMat(gVec_l);

    const xt::xarray<double> dVec_l = -brMat * bHat_l;
    const double denom = xt::linalg::vdot(nVec_l, dVec_l);

    if (denom < -eps) {
      const xt::xarray<double> P2_l = P0_l + dVec_l * num / denom;
      const xt::xarray<double> result {
        (xt::linalg::vdot(xt::view(rMat_d, xt::all(), 0), P2_l - tVec_d)),
        (xt::linalg::vdot(xt::view(rMat_d, xt::all(), 1), P2_l - tVec_d))
      };

      return result;
    }
  }

  return xt::xarray<double> {NAN, NAN};
}

const static xt::xarray<double> gvecToDetectorXY(
                                 const xt::xarray<double>& gVec_c, // 3xN matrix
                                 const xt::xarray<double>& rMat_d,
                                 const xt::xarray<double>& rMat_s, // 3x3xN matrix
                                 const xt::xarray<double>& rMat_c,
                                 const xt::xarray<double>& tVec_d,
                                 const xt::xarray<double>& tVec_s,
                                 const xt::xarray<double>& tVec_c,
                                 const xt::xarray<double>& beamVec)
{
  if (gVec_c.shape()[1] != 3) {
    std::cerr << "Error: gVec_c should have 3 columns!" << std::endl;
    return xt::xarray<double>(); // Return an empty matrix
  }

  if (rMat_s.shape()[0] % 3 != 0 || rMat_s.shape()[1] != 3) {
    std::cerr << "Error: rMat_s dimensions are not valid!" << std::endl;
    std::cerr << "Dimensions are: " << rMat_s.shape()[0] << ", " << rMat_s.shape()[1] << std::endl;
    return xt::xarray<double>(); // Return an empty matrix
  }

  xt::xarray<double> result = xt::zeros<double>({int(gVec_c.shape()[0]), 2});
  xt::xarray<double> bHat_l = beamVec / xt::linalg::norm(beamVec);

  for (int i = 0; i < gVec_c.shape()[0]; i++) {
    if (i * 3 + 2 >= rMat_s.shape()[0]) {
      std::cerr << "Error: Index out of bounds when accessing rMat_s!" << std::endl;
      continue;
    }

    xt::xarray<double> nVec_l = rMat_d * Zl;
    xt::xarray<double> P0_l = tVec_s + xt::view(rMat_s, xt::all(), xt::all(), i) * tVec_c;
    xt::xarray<double> rMat_sc = xt::view(rMat_s, xt::all(), xt::all(), i) * rMat_c;

    xt::row(result, i) = xt::transpose(gvecToDetectorXYOne(xt::view(gVec_c, xt::all(), i),
                                       rMat_d, rMat_sc, tVec_d,
                                       bHat_l, nVec_l,
                                       xt::linalg::vdot(nVec_l, P0_l),
                                       P0_l));
  }

  return result;
}

int main() {
  const xt::xarray<double>& gVec_c {{0, 0, 1}, {0, 0, 1}};
  const xt::xarray<double>& rMat_d {{1, 0, 0}, {0, 0, 1}, {0, 1, 0}};
  const xt::xarray<double>& rMat_s {{{1, 0, 0}, {0, 0, 1}, {0, 1, 0}},
                                    {{1, 0, 0}, {0, 0, 1}, {0, 1, 0}}}; // 3x3xN matrix
  const xt::xarray<double>& rMat_c {{1, 0, 0}, {0, 0, 1}, {0, 1, 0}};
  const xt::xarray<double>& tVec_d {1, 0, 0};
  const xt::xarray<double>& tVec_s {1, 0, 0};
  const xt::xarray<double>& tVec_c {1, 0, 0};
  const xt::xarray<double>& beamVec {1, 0, 0};

  auto x = gvecToDetectorXY(gVec_c, rMat_d, rMat_s, rMat_c, tVec_d, tVec_s, tVec_c, beamVec);
  std::cout << x << std::endl;
  return 0;
}