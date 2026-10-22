#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>
#include <string>
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <xsimd/xsimd.hpp>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#if defined(__GNUC__) || defined(__clang__)
#define HEXRD_FORCEINLINE __attribute__((always_inline))
#else
#define HEXRD_FORCEINLINE
#endif

#if (defined(__x86_64__) || defined(__i386__)) && !defined(_MSC_VER) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define HEXRD_AVX2 __attribute__((target("avx2")))
#endif

namespace py = pybind11;

const Eigen::Vector3d Zl = {0.0, 0.0, 1.0};
const double eps = std::numeric_limits<double>::epsilon();
const double sqrtEps = std::sqrt(std::numeric_limits<double>::epsilon());

const static Eigen::MatrixXd makeBinaryRotMat(const Eigen::Vector3d &a) {
  Eigen::Matrix3d r = 2.0 * a * a.transpose();
  r.diagonal() -= Eigen::Vector3d::Ones();
  return r;
}

Eigen::MatrixXd makeRotMatOfExpMap(const Eigen::Vector3d &e) {
  Eigen::AngleAxisd rotation(e.norm(), e.normalized());
  return rotation.toRotationMatrix();
}

static Eigen::MatrixX3d makeOscillRotMat(const double chi,
                                         const Eigen::VectorXd &ome) {
  Eigen::MatrixX3d rots(3 * ome.size(), 3);
  const double cchi = cos(chi), schi = sin(chi);
  xsimd::batch<double> come_v, some_v;

  const size_t batch_size = xsimd::simd_traits<double>::size;
  const size_t vec_size = ome.size();

  double come_arr[batch_size], some_arr[batch_size];

  for (size_t i = 0; i < vec_size; i += batch_size) {
    auto ome_v = xsimd::load_unaligned(&ome[i]);
    std::tie(some_v, come_v) = xsimd::sincos(ome_v);
    come_v.store_aligned(come_arr);
    some_v.store_aligned(some_arr);

    for (size_t j = 0; j < batch_size && (i + j) < vec_size; ++j) {
      rots.block<3, 3>(3 * (i + j), 0) << come_arr[j], 0, some_arr[j],
          schi * some_arr[j], cchi, -schi * come_arr[j], -cchi * some_arr[j],
          schi, cchi * come_arr[j];
    }
  }
  return rots;
}

// Both anglesToGvec and anglesToDvec, depending on if the last argument is true
// or false
const static Eigen::MatrixX3d
anglesToVec(const Eigen::MatrixXd &angs, const Eigen::Vector3d &bHat_l,
            const Eigen::Vector3d &eHat_l, double chi,
            const Eigen::Matrix3d &rMat_c, bool toGVec) noexcept {
  constexpr size_t batch_size = xsimd::simd_traits<double>::size;
  const size_t vec_size = angs.rows();
  const size_t num_full_batches = vec_size / batch_size;

  auto cchi = cos(chi);
  auto schi = sin(chi);
  Eigen::MatrixX3d gVec_c(angs.rows(), 3);

  const Eigen::Vector3d yHat = eHat_l.cross(bHat_l).normalized();
  const Eigen::Vector3d bHat = bHat_l.normalized();
  const Eigen::Vector3d xHat = bHat.cross(yHat);
  Eigen::Matrix3d rotMat;
  rotMat << xHat, yHat, -bHat;

  auto rotMat00 = rotMat(0, 0);
  auto rotMat01 = rotMat(0, 1);
  auto rotMat02 = rotMat(0, 2);
  auto rotMat10 = rotMat(1, 0);
  auto rotMat11 = rotMat(1, 1);
  auto rotMat12 = rotMat(1, 2);
  auto rotMat20 = rotMat(2, 0);
  auto rotMat21 = rotMat(2, 1);
  auto rotMat22 = rotMat(2, 2);

  auto mat00 = rMat_c(0, 0);
  auto mat01 = rMat_c(0, 1);
  auto mat02 = rMat_c(0, 2);
  auto mat10 = rMat_c(1, 0);
  auto mat11 = rMat_c(1, 1);
  auto mat12 = rMat_c(1, 2);
  auto mat20 = rMat_c(2, 0);
  auto mat21 = rMat_c(2, 1);
  auto mat22 = rMat_c(2, 2);

  // SIMD loop for full batches
  for (size_t i = 0; i < num_full_batches * batch_size; i += batch_size) {
    auto angs1_v = xsimd::load_unaligned(angs.data() + vec_size + i);
    auto angs2_v = xsimd::load_unaligned(angs.data() + 2 * vec_size + i);

    auto cosAngs2 = xsimd::cos(angs2_v);
    auto sinAngs2 = xsimd::sin(angs2_v);
    xsimd::batch<double> gVec_row_0, gVec_row_1, gVec_row_2;

    if (toGVec) {
      auto half_angs_v = xsimd::load_unaligned(angs.data() + i) * 0.5;
      auto cosHalfAngs = xsimd::cos(half_angs_v);

      auto preMult_gVec_row_0 = cosHalfAngs * xsimd::cos(angs1_v);
      auto preMult_gVec_row_1 = cosHalfAngs * xsimd::sin(angs1_v);
      auto preMult_gVec_row_2 = xsimd::sin(half_angs_v);

      gVec_row_0 = preMult_gVec_row_0 * rotMat00 +
                   preMult_gVec_row_1 * rotMat01 +
                   preMult_gVec_row_2 * rotMat02;
      gVec_row_1 = preMult_gVec_row_0 * rotMat10 +
                   preMult_gVec_row_1 * rotMat11 +
                   preMult_gVec_row_2 * rotMat12;
      gVec_row_2 = preMult_gVec_row_0 * rotMat20 +
                   preMult_gVec_row_1 * rotMat21 +
                   preMult_gVec_row_2 * rotMat22;
    } else {
      auto angs_v = xsimd::load_unaligned(angs.data() + i);
      xsimd::batch<double> sinAngs, cosAngs, sinAngs1, cosAngs1;
      std::tie(sinAngs, cosAngs) = xsimd::sincos(angs_v);
      std::tie(sinAngs1, cosAngs1) = xsimd::sincos(angs1_v);

      auto preMult_gVec_row_0 = sinAngs * cosAngs1;
      auto preMult_gVec_row_1 = sinAngs * sinAngs1;
      auto preMult_gVec_row_2 = -cosAngs;

      gVec_row_0 = preMult_gVec_row_0 * rotMat00 +
                   preMult_gVec_row_1 * rotMat01 +
                   preMult_gVec_row_2 * rotMat02;
      gVec_row_1 = preMult_gVec_row_0 * rotMat10 +
                   preMult_gVec_row_1 * rotMat11 +
                   preMult_gVec_row_2 * rotMat12;
      gVec_row_2 = preMult_gVec_row_0 * rotMat20 +
                   preMult_gVec_row_1 * rotMat21 +
                   preMult_gVec_row_2 * rotMat22;
    }

    auto dot0 =
        (mat00 * cosAngs2 + mat20 * sinAngs2) * gVec_row_0 +
        (mat00 * schi * sinAngs2 + mat10 * cchi - mat20 * schi * cosAngs2) *
            gVec_row_1 +
        (-mat00 * cchi * sinAngs2 + mat10 * schi + mat20 * cchi * cosAngs2) *
            gVec_row_2;
    auto dot1 =
        (mat01 * cosAngs2 + mat21 * sinAngs2) * gVec_row_0 +
        (mat01 * schi * sinAngs2 + mat11 * cchi - mat21 * schi * cosAngs2) *
            gVec_row_1 +
        (-mat01 * cchi * sinAngs2 + mat11 * schi + mat21 * cchi * cosAngs2) *
            gVec_row_2;
    auto dot2 =
        (mat02 * cosAngs2 + mat22 * sinAngs2) * gVec_row_0 +
        (mat02 * schi * sinAngs2 + mat12 * cchi - mat22 * schi * cosAngs2) *
            gVec_row_1 +
        (-mat02 * cchi * sinAngs2 + mat12 * schi + mat22 * cchi * cosAngs2) *
            gVec_row_2;

    xsimd::store_unaligned(gVec_c.data() + i, dot0);
    xsimd::store_unaligned(gVec_c.data() + vec_size + i, dot1);
    xsimd::store_unaligned(gVec_c.data() + 2 * vec_size + i, dot2);
  }

  // Loop for remaining elements, if any
  for (size_t i = num_full_batches * batch_size; i < vec_size; ++i) {
    auto angs1_v = *(angs.data() + vec_size + i);
    auto angs2_v = *(angs.data() + 2 * vec_size + i);

    auto cosAngs2 = std::cos(angs2_v);
    auto sinAngs2 = std::sin(angs2_v);
    double gVec_row_0, gVec_row_1, gVec_row_2;

    if (toGVec) {
      auto half_angs_v = *(angs.data() + i) * 0.5;
      auto cosHalfAngs = std::cos(half_angs_v);

      auto preMult_gVec_row_0 = cosHalfAngs * std::cos(angs1_v);
      auto preMult_gVec_row_1 = cosHalfAngs * std::sin(angs1_v);
      auto preMult_gVec_row_2 = std::sin(half_angs_v);

      gVec_row_0 = preMult_gVec_row_0 * rotMat00 +
                   preMult_gVec_row_1 * rotMat01 +
                   preMult_gVec_row_2 * rotMat02;
      gVec_row_1 = preMult_gVec_row_0 * rotMat10 +
                   preMult_gVec_row_1 * rotMat11 +
                   preMult_gVec_row_2 * rotMat12;
      gVec_row_2 = preMult_gVec_row_0 * rotMat20 +
                   preMult_gVec_row_1 * rotMat21 +
                   preMult_gVec_row_2 * rotMat22;
    } else {
      auto angs_v = *(angs.data() + i);
      double sinAngs = std::sin(angs_v);

      auto preMult_gVec_row_0 = sinAngs * std::cos(angs1_v);
      auto preMult_gVec_row_1 = sinAngs * std::sin(angs1_v);
      auto preMult_gVec_row_2 = -std::cos(angs_v);

      gVec_row_0 = preMult_gVec_row_0 * rotMat00 +
                   preMult_gVec_row_1 * rotMat01 +
                   preMult_gVec_row_2 * rotMat02;
      gVec_row_1 = preMult_gVec_row_0 * rotMat10 +
                   preMult_gVec_row_1 * rotMat11 +
                   preMult_gVec_row_2 * rotMat12;
      gVec_row_2 = preMult_gVec_row_0 * rotMat20 +
                   preMult_gVec_row_1 * rotMat21 +
                   preMult_gVec_row_2 * rotMat22;
    }

    auto dot0 =
        (mat00 * cosAngs2 + mat20 * sinAngs2) * gVec_row_0 +
        (mat00 * schi * sinAngs2 + mat10 * cchi - mat20 * schi * cosAngs2) *
            gVec_row_1 +
        (-mat00 * cchi * sinAngs2 + mat10 * schi + mat20 * cchi * cosAngs2) *
            gVec_row_2;
    auto dot1 =
        (mat01 * cosAngs2 + mat21 * sinAngs2) * gVec_row_0 +
        (mat01 * schi * sinAngs2 + mat11 * cchi - mat21 * schi * cosAngs2) *
            gVec_row_1 +
        (-mat01 * cchi * sinAngs2 + mat11 * schi + mat21 * cchi * cosAngs2) *
            gVec_row_2;
    auto dot2 =
        (mat02 * cosAngs2 + mat22 * sinAngs2) * gVec_row_0 +
        (mat02 * schi * sinAngs2 + mat12 * cchi - mat22 * schi * cosAngs2) *
            gVec_row_1 +
        (-mat02 * cchi * sinAngs2 + mat12 * schi + mat22 * cchi * cosAngs2) *
            gVec_row_2;

    *(gVec_c.data() + i) = dot0;
    *(gVec_c.data() + vec_size + i) = dot1;
    *(gVec_c.data() + 2 * vec_size + i) = dot2;
  }

  return gVec_c;
}

const static Eigen::MatrixX3d
anglesToGVec(const Eigen::MatrixXd &angs, const Eigen::Vector3d &bHat_l,
             const Eigen::Vector3d &eHat_l, double chi,
             const Eigen::Matrix3d &rMat_c) noexcept {
  return anglesToVec(angs, bHat_l, eHat_l, chi, rMat_c, true);
}

const static Eigen::MatrixX3d
anglesToDVec(const Eigen::MatrixXd &angs, const Eigen::Vector3d &bHat_l,
             const Eigen::Vector3d &eHat_l, double chi,
             const Eigen::Matrix3d &rMat_c) noexcept {
  return anglesToVec(angs, bHat_l, eHat_l, chi, rMat_c, false);
}

const static Eigen::Vector2d gvecToDetectorXYOne(
    const Eigen::Vector3d &gVec_c, const Eigen::Matrix3d &rMat_d,
    const Eigen::Matrix3d &rMat_sc, const Eigen::Vector3d &tVec_d,
    const Eigen::Vector3d &bHat_l, const Eigen::Vector3d &nVec_l, double num,
    const Eigen::Vector3d &P0_l) {
  Eigen::Vector3d gHat_c = gVec_c.normalized();
  Eigen::Vector3d gVec_l = rMat_sc * gHat_c;
  double bDot = -bHat_l.dot(gVec_l);

  if (bDot >= eps && bDot <= 1.0 - eps) {
    Eigen::Matrix3d brMat = makeBinaryRotMat(gVec_l);

    Eigen::Vector3d dVec_l = -brMat * bHat_l;
    double denom = nVec_l.dot(dVec_l);

    if (denom < -eps) {
      Eigen::Vector3d P2_l = P0_l + dVec_l * num / denom;
      Eigen::Vector2d result;
      result[0] = (rMat_d.col(0).dot(P2_l - tVec_d));
      result[1] = (rMat_d.col(1).dot(P2_l - tVec_d));

      return result;
    }
  }

  return Eigen::Vector2d(NAN, NAN);
}

const static Eigen::MatrixXd
gvecToDetectorXY(const Eigen::MatrixXd &gVec_c, const Eigen::Matrix3d &rMat_d,
                 const Eigen::MatrixXd &rMat_s, const Eigen::Matrix3d &rMat_c,
                 const Eigen::Vector3d &tVec_d, const Eigen::Vector3d &tVec_s,
                 const Eigen::Vector3d &tVec_c,
                 const Eigen::Vector3d &beamVec) {
  if (gVec_c.cols() != 3) {
    std::cerr << "Error: gVec_c should have 3 columns!" << std::endl;
    return Eigen::MatrixXd(); // Return an empty matrix
  }

  if (rMat_s.rows() % 3 != 0 || rMat_s.cols() != 3) {
    std::cerr << "Error: rMat_s dimensions are not valid!" << std::endl;
    return Eigen::MatrixXd(); // Return an empty matrix
  }

  Eigen::MatrixXd result(gVec_c.rows(), 2);
  Eigen::Vector3d bHat_l = beamVec.normalized();

  for (int i = 0; i < gVec_c.rows(); i++) {
    if (i * 3 + 2 >= rMat_s.rows()) {
      std::cerr << "Error: Index out of bounds when accessing rMat_s!"
                << std::endl;
      continue; // Skip this iteration
    }

    Eigen::Vector3d nVec_l = rMat_d * Zl;
    Eigen::Vector3d P0_l = tVec_s + rMat_s.block<3, 3>(i * 3, 0) * tVec_c;
    Eigen::Matrix3d rMat_sc = rMat_s.block<3, 3>(i * 3, 0) * rMat_c;

    result.row(i) = gvecToDetectorXYOne(gVec_c.row(i), rMat_d, rMat_sc, tVec_d,
                                        bHat_l, nVec_l, nVec_l.dot(P0_l), P0_l)
                        .transpose();
  }

  return result;
}

const static Eigen::Vector2d gvecToDetectorXYOneSimple(
    const Eigen::Vector3d &gVec_l, const Eigen::Matrix3d &rMat_d,
    const Eigen::Vector3d &tVec_d, const Eigen::Vector3d &bHat_l,
    const Eigen::Vector3d &nVec_l, double num, const Eigen::Vector3d &P0_l) {
  double bDot = -bHat_l.dot(gVec_l);

  if (bDot >= eps && bDot <= 1.0 - eps) {
    Eigen::Vector3d dVec_l = -makeBinaryRotMat(gVec_l) * bHat_l;
    double denom = nVec_l.dot(dVec_l);

    if (denom < -eps) {
      Eigen::Vector3d P2_l = P0_l + dVec_l * num / denom - tVec_d;
      return Eigen::Vector2d(rMat_d.col(0).dot(P2_l), rMat_d.col(1).dot(P2_l));
    }
  }

  return Eigen::Vector2d(NAN, NAN);
}

const static Eigen::MatrixXd gvecToDetectorXYFromAngles(
    const double chi, const Eigen::VectorXd omes, const Eigen::MatrixXd &gVec_c,
    const Eigen::Matrix3d &rMat_d, const Eigen::Matrix3d &rMat_c,
    const Eigen::Vector3d &tVec_d, const Eigen::Vector3d &tVec_s,
    const Eigen::Vector3d &tVec_c, const Eigen::Vector3d &beamVec) {
  const int npts = gVec_c.rows();
  Eigen::MatrixXd result(npts, 2);
  Eigen::MatrixX3d rMat_s = makeOscillRotMat(chi, omes);

  Eigen::Vector3d bHat_l = beamVec.normalized();

  for (int i = 0; i < npts; i++) {
    Eigen::Vector3d nVec_l = rMat_d * Zl;
    Eigen::Vector3d P0_l = tVec_s + rMat_s.block<3, 3>(i * 3, 0) * tVec_c;
    Eigen::Vector3d P3_l = tVec_d;

    double num = nVec_l.dot(P3_l - P0_l);

    Eigen::Matrix3d rMat_sc = rMat_s.block<3, 3>(i * 3, 0) * rMat_c;

    result.row(i) = gvecToDetectorXYOne(gVec_c.row(i), rMat_d, rMat_sc, tVec_d,
                                        bHat_l, nVec_l, num, P0_l)
                        .transpose();
  }

  return result;
}

const static Eigen::MatrixX2d anglesToDetectorXY(
    const double chi, const Eigen::MatrixXd omes, const Eigen::Matrix3d &rMat_d,
    const Eigen::Matrix3d &rMat_c, const Eigen::Vector3d &tVec_d,
    const Eigen::Vector3d &tVec_s, const Eigen::Vector3d &tVec_c,
    const Eigen::Vector3d &beamVec) {
  Eigen::MatrixX2d result(omes.rows(), 2);
  const Eigen::MatrixX3d rMat_s = makeOscillRotMat(chi, omes.col(2));
  const Eigen::MatrixX3d gVec_c =
      anglesToGVec(omes, beamVec, {1, 0, 0}, chi, rMat_c)
          .rowwise()
          .normalized();
  const Eigen::Vector3d bHat_l = beamVec.normalized();

  for (int i = 0; i < omes.rows(); i++) {
    Eigen::Matrix3d current_rmat = rMat_s.block<3, 3>(i * 3, 0);
    Eigen::Vector3d nVec_l = rMat_d * Zl;
    Eigen::Vector3d P0_l = tVec_s + current_rmat * tVec_c;

    Eigen::Vector3d norm = gVec_c.row(i);
    Eigen::Vector3d gVec_l = current_rmat * rMat_c * norm;

    result.row(i) =
        gvecToDetectorXYOneSimple(gVec_l, rMat_d, tVec_d, bHat_l, nVec_l,
                                  nVec_l.dot(tVec_d - P0_l), P0_l)
            .transpose();
  }

  return result;
}

// Scalar ordering is intentional: these routines preserve the historical
// diffraction predictor's rounding, including inverse stretch and beam frames.
using Index = py::ssize_t;
constexpr double pi = 3.14159265358979323846;
template <typename T>
using CArray = py::array_t<T, py::array::c_style | py::array::forcecast | py::detail::npy_api::NPY_ARRAY_ALIGNED_>;
template <typename T>
using StridedArray = py::array_t<T, py::array::forcecast | py::detail::npy_api::NPY_ARRAY_ALIGNED_>;

static void checkShape(const py::array &array, std::initializer_list<Index> dims) {
  if (array.ndim() != static_cast<Index>(dims.size()))
    throw py::value_error("incorrect array dimensions");
  Index axis = 0;
  for (Index dim : dims) {
    if (dim >= 0 && array.shape(axis) != dim)
      throw py::value_error("inconsistent array shape");
    ++axis;
  }
}

static double normalize3(const double *src, double *dst) {
  double norm = 0;
  for (int j = 0; j < 3; ++j) norm += src[j]*src[j];
  norm = std::sqrt(norm);
  for (int j = 0; j < 3; ++j) dst[j] = norm > eps ? src[j]/norm : src[j];
  return norm;
}

static void cross3(const double *a, const double *b, double *out) {
  for (int j = 0; j < 3; ++j)
    out[j] = a[(j+1)%3]*b[(j+2)%3] - b[(j+1)%3]*a[(j+2)%3];
}

static void matvec3(const double *matrix, const double *vec, double *out) {
  for (int j = 0; j < 3; ++j) {
    out[j] = 0;
    for (int k = 0; k < 3; ++k) out[j] += matrix[3*j+k]*vec[k];
  }
}

static std::vector<double> reciprocalVectors(const CArray<double> &hkls,
                                              const CArray<double> &bmat) {
  checkShape(hkls, {-1, 3});
  checkShape(bmat, {3, 3});
  std::vector<double> result(hkls.size());
  for (Index i = 0; i < hkls.shape(0); ++i)
    matvec3(bmat.data(), hkls.data()+3*i, result.data()+3*i);
  return result;
}

static std::array<double, 9> beamFrame(const double *beam, const double *eta) {
  double b[3], y[3], x[3];
  if (!(normalize3(beam, b) > eps)) throw std::runtime_error("bvec_l MUST NOT be ZERO!");
  cross3(eta, beam, y);
  if (!(normalize3(y, y) > eps))
    throw std::runtime_error("bvec_l and evec_l MUST NOT be collinear!");
  cross3(b, y, x);
  std::array<double, 9> frame;
  for (int j = 0; j < 3; ++j) {
    frame[3*j] = x[j]; frame[3*j+1] = y[j]; frame[3*j+2] = -b[j];
  }
  return frame;
}

static py::array_t<double> makeEtaFrameRotMat(CArray<double> beam, CArray<double> eta) {
  checkShape(beam, {3}); checkShape(eta, {3});
  py::array_t<double> output({3, 3});
  auto frame = beamFrame(beam.data(), eta.data());
  std::copy(frame.begin(), frame.end(), output.mutable_data());
  return output;
}

static void rotateOne(double angle, const double *axis, const double *vec, double *out) {
  double co = std::cos(angle), so = std::sin(angle), norm = 0, projection = 0, cross[3];
  for (int j = 0; j < 3; ++j) {
    norm += axis[j]*axis[j]; projection += axis[j]*vec[j];
  }
  norm = std::sqrt(norm);
  cross3(axis, vec, cross);
  for (int j = 0; j < 3; ++j)
    out[j] = co*vec[j]+(so/norm)*cross[j]+(1.-co)*projection*axis[j]/(norm*norm);
}

static py::array_t<double> rotateVectors(CArray<double> angles, CArray<double> axes,
                                         CArray<double> vectors) {
  checkShape(vectors, {-1, 3});
  Index n = vectors.shape(0);
  if (angles.ndim() > 1 || axes.ndim() < 1 || axes.ndim() > 2 ||
      axes.shape(axes.ndim()-1) != 3 ||
      (angles.ndim() == 1 && angles.size() != n) ||
      (axes.ndim() == 2 && axes.shape(0) != n))
    throw py::value_error("angles, axes and vectors have mismatching dimensions");
  py::array_t<double> output({n, Index(3)});
  auto result = output.mutable_data();
  py::gil_scoped_release release;
  for (Index i = 0; i < n; ++i)
    rotateOne(angles.data()[angles.ndim() ? i : 0], axes.data()+(axes.ndim()==2 ? 3*i : 0),
              vectors.data()+3*i, result+3*i);
  return output;
}

static py::tuple detectorXYToGvec(CArray<double> xy, CArray<double> rd, CArray<double> rs,
    CArray<double> td, CArray<double> ts, CArray<double> tc,
    CArray<double> beam, CArray<double> etaVector) {
  checkShape(xy, {-1, 2}); checkShape(rd, {3, 3}); checkShape(rs, {3, 3});
  checkShape(td, {3}); checkShape(ts, {3}); checkShape(tc, {3});
  checkShape(beam, {3}); checkShape(etaVector, {3});
  Index n = xy.shape(0);
  py::array_t<double> tth(n), eta(n), gvec({n, Index(3)});
  double *angles = tth.mutable_data(), *azimuth = eta.mutable_data(), *out = gvec.mutable_data();
  auto frame = beamFrame(beam.data(), etaVector.data());
  { py::gil_scoped_release release;
    double b[3], shift[3];
    normalize3(beam.data(), b);
    for (int j = 0; j < 3; ++j) {
      shift[j] = td.data()[j]-ts.data()[j];
      for (int k = 0; k < 3; ++k) shift[j] -= rs.data()[3*j+k]*tc.data()[k];
    }
    for (Index i = 0; i < n; ++i) {
      double direction[3], norm = 0;
      for (int j = 0; j < 3; ++j) {
        direction[j] = shift[j];
        for (int k = 0; k < 2; ++k) direction[j] += rd.data()[3*j+k]*xy.data()[2*i+k];
        norm += direction[j]*direction[j];
      }
      if (norm > eps) for (int j = 0; j < 3; ++j) direction[j] /= std::sqrt(norm);
      double dot = 0, transverse[2] = {}, axis[3];
      for (int j = 0; j < 3; ++j) dot += b[j]*direction[j];
      angles[i] = std::acos(dot);
      for (int j = 0; j < 2; ++j)
        for (int k = 0; k < 3; ++k) transverse[j] += frame[3*k+j]*direction[k];
      azimuth[i] = std::atan2(transverse[1], transverse[0]);
      cross3(b, direction, axis);
      norm = 0;
      for (int j = 0; j < 3; ++j) norm += axis[j]*axis[j];
      norm = std::sqrt(norm);
      for (int j = 0; j < 3; ++j) axis[j] /= norm;
      rotateOne(0.5*(pi-angles[i]), axis, direction, out+3*i);
    }
  }
  return py::make_tuple(py::make_tuple(tth, eta), gvec);
}

// Legacy projection uses unnormalized input g-vectors and a distinct beam fast
// path. Keep that contract alongside the existing Eigen projection interface.
static void rotateExact3(const double *r, const double *v, double *out) {
  for (int j = 0; j < 3; ++j)
    out[j] = r[3*j]*v[0]+r[3*j+1]*v[1]+r[3*j+2]*v[2];
}

template <bool single>
static py::array_t<double> projectGvecs(CArray<double> vectors, CArray<double> rd,
    CArray<double> rs, CArray<double> rc, CArray<double> td, CArray<double> ts,
    CArray<double> tc, py::object beamObject) {
  checkShape(vectors, {-1, 3}); checkShape(rd, {3, 3}); checkShape(rc, {3, 3});
  checkShape(td, {3}); checkShape(ts, {3}); checkShape(tc, {3});
  Index n = vectors.shape(0);
  if (single) checkShape(rs, {3, 3}); else checkShape(rs, {n, 3, 3});
  CArray<double> beamArray;
  double beam[3] = {}, basis[3][4], combined[9], fixedOrigin[3];
  bool generalBeam = !beamObject.is_none();
  if (generalBeam) {
    beamArray = CArray<double>::ensure(beamObject);
    if (!beamArray) throw py::value_error("invalid beam array");
    checkShape(beamArray, {3});
    const double *b = beamArray.data();
    double norm = b[0]*b[0]+b[1]*b[1]+b[2]*b[2];
    if (norm > eps*eps) {
      double inverse = -1./std::sqrt(norm);
      beam[2] = b[2]*inverse;
      generalBeam = !(std::abs(beam[2]-1.) < eps);
      beam[0] = b[0]*inverse; beam[1] = b[1]*inverse;
    } else std::copy(b, b+3, beam);
  }
  py::array_t<double> output({n, Index(2)}); auto result = output.mutable_data();
  py::gil_scoped_release release;
  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 3; ++k) basis[j][k] = rd.data()[3*k+j];
    basis[j][3] = -(td.data()[0]*basis[j][0]+td.data()[1]*basis[j][1]+td.data()[2]*basis[j][2]);
  }
  if (single) {
    for (int j = 0; j < 3; ++j) {
      fixedOrigin[j] = ts.data()[j];
      for (int k = 0; k < 3; ++k) fixedOrigin[j] += rs.data()[3*j+k]*tc.data()[k];
      for (int k = 0; k < 3; ++k)
        combined[3*j+k] = rs.data()[3*j]*rc.data()[k]+rs.data()[3*j+1]*rc.data()[3+k]+rs.data()[3*j+2]*rc.data()[6+k];
    }
  }
  constexpr Index chunk = 128;
  for (Index first = 0; first < n; first += chunk) {
    Index count = std::min(chunk, n-first);
    double crystal[3*chunk], lab[3*chunk], rays[3*chunk], origins[3*chunk];
    if (single) {
      for (Index i = 0; i < count; ++i) rotateExact3(combined, vectors.data()+3*(first+i), lab+3*i);
    } else {
      for (Index i = 0; i < count; ++i)
        for (int j = 0; j < 3; ++j) {
          const double *r = rs.data()+9*(first+i)+3*j;
          origins[3*i+j] = ts.data()[j]+r[0]*tc.data()[0]+r[1]*tc.data()[1]+r[2]*tc.data()[2];
        }
      for (Index i = 0; i < count; ++i) rotateExact3(rc.data(), vectors.data()+3*(first+i), crystal+3*i);
      for (Index i = 0; i < count; ++i) rotateExact3(rs.data()+9*(first+i), crystal+3*i, lab+3*i);
    }
    for (Index i = 0; i < count; ++i) {
      double x = lab[3*i], y = lab[3*i+1], z = lab[3*i+2];
      double xz = x*z*2., yz = y*z*2., zz = z*z*2.-1.;
      if (generalBeam) {
        double xx = x*x*2.-1., yy = y*y*2.-1., xy = x*y*2.;
        rays[3*i] = xz*beam[2]+xy*beam[1]+xx*beam[0];
        rays[3*i+1] = yz*beam[2]+yy*beam[1]+xy*beam[0];
        rays[3*i+2] = zz*beam[2]+yz*beam[1]+xz*beam[0];
      } else { rays[3*i] = xz; rays[3*i+1] = yz; rays[3*i+2] = zz; }
    }
    for (Index i = 0; i < count; ++i) {
      const double *origin = single ? fixedOrigin : origins+3*i, *ray = rays+3*i;
      const double *plane = basis[2];
      double numerator = plane[3]+plane[0]*origin[0]+plane[1]*origin[1]+plane[2]*origin[2];
      double denominator = plane[0]*ray[0]+plane[1]*ray[1]+plane[2]*ray[2];
      double factor = numerator/denominator;
      if (!(factor < 0.)) factor = NAN;
      for (int j = 0; j < 2; ++j) {
        const double *b = basis[j];
        double base = b[3]+b[0]*origin[0]+b[1]*origin[1]+b[2]*origin[2];
        double direction = b[0]*ray[0]+b[1]*ray[1]+b[2]*ray[2];
        result[2*(first+i)+j] = base-factor*direction;
      }
    }
  }
  return output;
}

// Reduce atan to a table anchor and a residual <= 1/512. The quadratic
// residual's truncation error is below 2.5e-9 radians; bin boundaries still
// use the original predictor, guarding both rounding and geometry reduction.
static const std::array<std::array<double, 2>, 513> scoringAtans = [] {
  std::array<std::array<double, 2>, 513> result;
  for (Index i = 0; i < 513; ++i) {
    double t = double(i)/512.;
    result[i] = {std::atan(t), 1./(1.+t*t)};
  }
  return result;
}();

static double scoringAtan2(double y, double x) {
  double ax = std::abs(x), ay = std::abs(y), maximum = std::max(ax, ay);
  if (!(maximum > 0) || !std::isfinite(maximum)) return std::atan2(y, x);
  double ratio = std::min(ax, ay)/maximum;
  Index i = Index(ratio*512.);
  double anchor = double(i)/512.;
  double residual = ratio-anchor, derivative = scoringAtans[i][1];
  double angle = scoringAtans[i][0]+residual*derivative*(1.-residual*anchor*derivative);
  angle += double(ay > ax)*(pi/2.-2.*angle);
  angle += double(x < 0)*(pi-2.*angle);
  return std::copysign(angle, y);
}

#ifdef HEXRD_AVX2
namespace scoring_simd {
struct Batch {
  __m256d value;
  HEXRD_AVX2 Batch(double x) : value(_mm256_set1_pd(x)) {}
  HEXRD_AVX2 Batch(__m256d x) : value(x) {}
  HEXRD_AVX2 static Batch load_unaligned(const double *p) { return _mm256_loadu_pd(p); }
  HEXRD_AVX2 void store_unaligned(double *p) const { _mm256_storeu_pd(p, value); }
  friend HEXRD_AVX2 Batch operator+(Batch a, Batch b) { return _mm256_add_pd(a.value, b.value); }
  friend HEXRD_AVX2 Batch operator-(Batch a, Batch b) { return _mm256_sub_pd(a.value, b.value); }
  friend HEXRD_AVX2 Batch operator*(Batch a, Batch b) { return _mm256_mul_pd(a.value, b.value); }
  friend HEXRD_AVX2 Batch operator~(Batch a) { return _mm256_xor_pd(a.value, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))); }
  friend HEXRD_AVX2 Batch operator/(Batch a, Batch b) { return _mm256_div_pd(a.value, b.value); }
  friend HEXRD_AVX2 Batch operator-(Batch a) { return _mm256_xor_pd(a.value, _mm256_set1_pd(-0.)); }
  friend HEXRD_AVX2 Batch operator&(Batch a, Batch b) { return _mm256_and_pd(a.value, b.value); }
  friend HEXRD_AVX2 Batch operator|(Batch a, Batch b) { return _mm256_or_pd(a.value, b.value); }
  friend HEXRD_AVX2 Batch operator>(Batch a, Batch b) { return _mm256_cmp_pd(a.value, b.value, _CMP_GT_OQ); }
  friend HEXRD_AVX2 Batch operator<(Batch a, Batch b) { return _mm256_cmp_pd(a.value, b.value, _CMP_LT_OQ); }
  friend HEXRD_AVX2 Batch operator<=(Batch a, Batch b) { return _mm256_cmp_pd(a.value, b.value, _CMP_LE_OQ); }
};
HEXRD_AVX2 static Batch abs(Batch a) { return _mm256_andnot_pd(_mm256_set1_pd(-0.), a.value); }
HEXRD_AVX2 static Batch sqrt(Batch a) { return _mm256_sqrt_pd(a.value); }
HEXRD_AVX2 static Batch floor(Batch a) { return _mm256_floor_pd(a.value); }
HEXRD_AVX2 static Batch min(Batch a, Batch b) { return _mm256_min_pd(a.value, b.value); }
HEXRD_AVX2 static Batch isfinite(Batch a) { return abs(a) <= Batch(std::numeric_limits<double>::max()); }
HEXRD_AVX2 static Batch select(Batch mask, Batch yes, Batch no) { return _mm256_blendv_pd(no.value, yes.value, mask.value); }
HEXRD_AVX2 static Batch max(Batch a, Batch b) { return _mm256_max_pd(a.value, b.value); }
HEXRD_AVX2 static bool any(Batch a) { return _mm256_movemask_pd(a.value) != 0; }
HEXRD_AVX2 static __m128i mask128(Batch a) {
  auto lo = _mm256_castpd256_pd128(a.value), hi = _mm256_extractf128_pd(a.value, 1);
  return _mm_castps_si128(_mm_shuffle_ps(_mm_castpd_ps(lo), _mm_castpd_ps(hi), _MM_SHUFFLE(2, 0, 2, 0)));
}
HEXRD_AVX2 static Batch copysign(Batch a, Batch b) {
  auto sign = _mm256_set1_pd(-0.);
  return _mm256_or_pd(_mm256_andnot_pd(sign, a.value), _mm256_and_pd(sign, b.value));
}
HEXRD_AVX2 static Batch atan2(Batch y, Batch x) {
  auto ax = abs(x), ay = abs(y), ratio = min(ax, ay)/max(ax, ay);
  // Degree-9 Chebyshev fit of atan(sqrt(t))/sqrt(t); angle error <2e-9 rad.
  auto t = ratio*ratio, t2 = t*t;
  auto p0 = Batch(0.9999999971605468)+t*Batch(-0.3333327629200022);
  auto p1 = Batch(0.19998075281505315)+t*Batch(-0.1426001608570026);
  auto p2 = Batch(0.1093234151327955)+t*Batch(-0.08349724996759872);
  auto p3 = Batch(0.05708955633078856)+t*Batch(-0.030351865114653258);
  auto p4 = Batch(0.010487649399874501)+t*Batch(-0.0017011700903822203);
  auto angle = ratio*(p0+t2*(p1+t2*(p2+t2*(p3+t2*p4))));
  angle = select(ay > ax, Batch(pi/2.)-angle, angle);
  angle = select(x < Batch(0.), Batch(pi)-angle, angle);
  return copysign(angle, y);
}
}
#endif

struct OscillationGeometry {
  double beam[3], cchi, schi, wavelength;
  std::array<double, 9> frame = {};
  std::array<double, 6> stretch;
  bool consistent;

  OscillationGeometry(double chi, double wave, const double *stretch,
                      const double *bvec, const double *evec)
      : cchi(std::cos(chi)), schi(std::sin(chi)), wavelength(wave) {
    std::copy(stretch, stretch+6, this->stretch.begin());
    double eta[3], dot = 0;
    normalize3(bvec, beam);
    normalize3(evec, eta);
    for (int j = 0; j < 3; ++j) dot += beam[j]*eta[j];
    consistent = std::abs(dot) < 1.0 - 1.5e-8;
    if (consistent) frame = beamFrame(beam, eta);
  }

  void predict(Index n, const double *reciprocal, const double *rmat,
               double *out0, double *out1, const double *lengths = nullptr,
               const Index *opposites = nullptr, const double *visibility = nullptr) const {
    for (Index i = 0; i < n; ++i) {
      if (visibility) {
        double y = std::abs(rmat[3]*reciprocal[3*i]+rmat[4]*reciprocal[3*i+1]+
                            rmat[5]*reciprocal[3*i+2]);
        if (y < visibility[2*i] || y > visibility[2*i+1]) {
          std::fill(out0+3*i, out0+3*i+3, NAN);
          std::fill(out1+3*i, out1+3*i+3, NAN);
          continue;
        }
      }
      if (lengths && opposites && opposites[i] >= 0 && out0[3*opposites[i]] == -INFINITY) {
        // Opposite HKLs reflect eta and shift omega by pi in the reduced geometry.
        for (auto out : {out0, out1}) {
          out[3*i] = -INFINITY;
          out[3*i+1] = -out[3*opposites[i]+1];
          out[3*i+2] = out[3*opposites[i]+2]+pi;
        }
        continue;
      }
      double gc[3], gs[3], tmp[3];
      matvec3(rmat, reciprocal+3*i, gs);
      if (lengths && lengths[i] > eps) {
        double x = gs[0], y = gs[1], z = gs[2], squared = lengths[i]*lengths[i];
        double st = 0.5*wavelength*squared, root2 = x*x+z*z-st*st;
        if (root2 > 1e-8*squared && std::abs(0.5*wavelength*lengths[i]) <= 1.) {
          double root = std::sqrt(root2), eta = scoringAtan2(y, root);
          double ome0 = scoringAtan2(-st*x+root*z, root*x+st*z);
          double ome1 = scoringAtan2(-st*x-root*z, -root*x+st*z);
          if (std::isfinite(eta) && std::isfinite(ome0) && std::isfinite(ome1)) {
            out0[3*i] = out1[3*i] = -INFINITY; // Mark reduced, rather than scalar, predictions.
            out0[3*i+1] = eta;
            out1[3*i+1] = std::copysign(pi, y)-eta;
            out0[3*i+2] = ome0; out1[3*i+2] = ome1;
            continue;
          }
        }
      }
      tmp[0] = stretch[0]*gs[0] + (stretch[5]*gs[1] + stretch[4]*gs[2])/std::sqrt(2.);
      tmp[1] = stretch[1]*gs[1] + (stretch[5]*gs[0] + stretch[3]*gs[2])/std::sqrt(2.);
      tmp[2] = stretch[2]*gs[2] + (stretch[4]*gs[0] + stretch[3]*gs[1])/std::sqrt(2.);
      double norm = 0;
      for (int j = 0; j < 3; ++j) {
        gc[j] = 0;
        for (int k = 0; k < 3; ++k) gc[j] += rmat[j+3*k]*tmp[k];
        norm += gc[j]*gc[j];
      }
      norm = std::sqrt(norm);
      if (norm > eps) for (int j = 0; j < 3; ++j) gs[j] = tmp[j]/norm;
      double sintht = 0.5*wavelength*norm;
      double a = gs[2]*beam[0] + schi*gs[0]*beam[1] - cchi*gs[0]*beam[2];
      double b = gs[0]*beam[0] - schi*gs[2]*beam[1] + cchi*gs[2]*beam[2];
      double c = -sintht - cchi*gs[1]*beam[1] - schi*gs[1]*beam[2];
      double phase = std::atan2(b, a), rhs = c/std::sqrt(a*a+b*b);
      double *outputs[2] = {out0+3*i, out1+3*i};
      if (std::abs(rhs) > 1.) {
        for (auto out : outputs) std::fill(out, out+3, NAN);
        continue;
      }
      double angle = std::asin(rhs);
      outputs[0][2] = angle-phase;
      outputs[1][2] = pi-angle-phase;
      double tth = consistent ? 2.*std::asin(sintht) : NAN;
      if (consistent) for (auto out : outputs) {
        double co = std::cos(out[2]), so = std::sin(out[2]);
        double sample[9] = {co, 0, so, schi*so, cchi, -schi*co,
                           -cchi*so, schi, cchi*co};
        double lab[3], transverse[2] = {};
        matvec3(sample, gs, lab);
        for (int j = 0; j < 2; ++j)
          for (int k = 0; k < 3; ++k) transverse[j] += frame[3*k+j]*lab[k];
        out[1] = std::atan2(transverse[1], transverse[0]);
        out[0] = tth;
      } else for (auto out : outputs) std::fill(out, out+2, NAN);
    }
  }
};

static py::tuple oscillAnglesOfHKLs(CArray<double> hkls, double chi,
    CArray<double> rmat, CArray<double> bmat, double wavelength,
    CArray<double> vinv, CArray<double> beam, CArray<double> eta) {
  checkShape(rmat, {3, 3}); checkShape(vinv, {6});
  checkShape(beam, {3}); checkShape(eta, {3});
  auto reciprocal = reciprocalVectors(hkls, bmat);
  py::array_t<double> out0({hkls.shape(0), Index(3)}), out1({hkls.shape(0), Index(3)});
  OscillationGeometry geometry(chi, wavelength, vinv.data(), beam.data(), eta.data());
  { py::gil_scoped_release release;
    geometry.predict(hkls.shape(0), reciprocal.data(), rmat.data(),
                     out0.mutable_data(), out1.mutable_data()); }
  return py::make_tuple(out0, out1);
}

static py::object reduceFiber(StridedArray<double> fiber, StridedArray<double> symmetry) {
  checkShape(fiber, {4, -1}); checkShape(symmetry, {-1, 4, 4});
  Index n = fiber.shape(1), count = symmetry.shape(0);
  if (!count) throw py::value_error("fiber symmetry cannot be empty");
  // NumPy's dot kernel preserves the active BLAS implementation's rounding.
  auto array = reinterpret_cast<PyArrayObject *>(fiber.ptr());
#if NPY_ABI_VERSION >= 0x02000000
  auto dot = PyDataType_GetArrFuncs(PyArray_DESCR(array))->dotfunc;
#else
  auto dot = PyArray_DESCR(array)->f->dotfunc;
#endif
  std::vector<double> reduced(4*n);
  std::vector<std::array<Index, 4>> groups(n);
  std::vector<Index> order(n), unique;
  bool invalid_norm = false;
  {
    py::gil_scoped_release release;
    using Strides = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    using View = Eigen::Map<const Eigen::MatrixXd, 0, Strides>;
    View scalar_rows(symmetry.data(), count, 4,
        Strides(symmetry.strides(2)/Index(sizeof(double)),
                symmetry.strides(0)/Index(sizeof(double))));
    View fibers(fiber.data(), 4, n,
        Strides(fiber.strides(1)/Index(sizeof(double)),
                fiber.strides(0)/Index(sizeof(double))));
    constexpr Index batch_size = 64;
    Eigen::MatrixXd scalars;
    for (Index q = 0; q < n; ++q) {
      Index column = q%batch_size;
      if (!column) {
        auto block = fibers.middleCols(q, std::min(batch_size, n-q));
        // Accumulate components in reference order to preserve rounding.
        scalars.noalias() = scalar_rows.col(0)*block.row(0);
        for (int r = 1; r < 4; ++r) scalars.noalias() += scalar_rows.col(r)*block.row(r);
        scalars = scalars.cwiseAbs();
      }
      double v[4]; Index selected;
      for (int r = 0; r < 4; ++r) v[r] = *fiber.data(r, q);
      double best = scalars.col(column).maxCoeff(&selected);
      scalars(selected, column) = -1.; // Exclude the winner to find the runner-up.
      double second = scalars.col(column).maxCoeff();
      auto evaluate = [&](Index s, double *value) {
        for (int r = 0; r < 4; ++r)
          dot(const_cast<double *>(symmetry.data(s, r, 0)), symmetry.strides(2),
              v, sizeof(double), value+r, 4, array);
        double norm = std::sqrt(((value[0]*value[0]+value[1]*value[1])+value[2]*value[2])+value[3]*value[3]);
        if (!(norm > 10.*eps) || !std::isfinite(norm)) { invalid_norm = true; return; }
        bool negative = value[0] < 0;
        for (int r = 0; r < 4; ++r) {
          value[r] /= norm;
          if (negative) value[r] = -value[r];
        }
      };
      double value[4];
      if (best-second <= 1e-12) {
        // Resolve near ties using every normalized candidate, in reference order.
        best = -1.;
        for (Index s = 0; s < count; ++s) {
          double candidate[4]; evaluate(s, candidate);
          if (candidate[0] > best) {
            best = candidate[0]; std::copy(candidate, candidate+4, value);
          }
        }
      } else evaluate(selected, value);
      if (invalid_norm) break;
      for (int r = 0; r < 4; ++r) reduced[r*n+q] = value[r];
    }
    if (!invalid_norm) {
      for (Index q = 0; q < n; ++q) order[q] = q;
      for (int r = 0; r < 4; ++r) {
        std::stable_sort(order.begin(), order.end(), [&](Index a, Index b) {
          return reduced[r*n+a] < reduced[r*n+b];
        });
        Index label = 0;
        for (Index j = 0; j < n; ++j) {
          if (!j || reduced[r*n+order[j]]-reduced[r*n+order[j-1]] > 1e-12) ++label;
          groups[order[j]][r] = label;
        }
      }
      std::sort(order.begin(), order.end(), [&](Index a, Index b) {
        for (int r = 3; r >= 0; --r)
          if (groups[a][r] != groups[b][r]) return groups[a][r] < groups[b][r];
        return a < b;
      });
      for (Index q : order)
        if (unique.empty() || groups[q] != groups[unique.back()]) unique.push_back(q);
    }
  }
  if (invalid_norm) return py::none(); // Non-finite/degenerate inputs use Python's reference.
  py::array_t<double> output({Index(4), Index(unique.size())});
  for (int r = 0; r < 4; ++r) for (Index j = 0; j < Index(unique.size()); ++j)
    output.mutable_data()[r*unique.size()+j] = reduced[r*n+unique[j]];
  return std::move(output);
}

static void quaternionRotation(double a, double b, double c, double d, double *r) {
  r[0] = a*a + b*b - c*c - d*d;
  r[1] = 2*b*c - 2*a*d; r[2] = 2*a*c + 2*b*d;
  r[3] = 2*a*d + 2*b*c; r[4] = a*a - b*b + c*c - d*d;
  r[5] = 2*c*d - 2*a*b; r[6] = 2*b*d - 2*a*c;
  r[7] = 2*a*b + 2*c*d; r[8] = a*a - b*b - c*c + d*d;
}

static py::array_t<double> quaternionRotations(CArray<double> quats) {
  checkShape(quats, {-1, 4});
  py::array_t<double> output({quats.shape(0), Index(3), Index(3)});
  const double *q = quats.data(); double *r = output.mutable_data();
  py::gil_scoped_release release;
  for (Index i = 0; i < quats.shape(0); ++i, q += 4, r += 9)
    quaternionRotation(q[0], q[1], q[2], q[3], r);
  return output;
}

// Fast path for ordinary numeric NPZ directories; zipfile handles extensions
// such as ZIP64, encryption, non-ASCII names and self-extracting prefixes.
static py::object npzIndex(py::buffer buffer, bool metadataOnly) {
  auto view = buffer.request();
  if (view.ndim != 1 || view.itemsize != 1 || view.strides[0] != 1)
    throw py::value_error("expected a contiguous byte buffer");
  const auto *data = static_cast<const unsigned char *>(view.ptr);
  Index size = view.size*view.itemsize;
  auto word = [&](Index offset, int bytes = 4) -> uint32_t {
    if (offset < 0 || offset > size-bytes) throw py::value_error("truncated ZIP directory");
    uint32_t value = 0;
    for (int i = 0; i < bytes; ++i) value |= uint32_t(data[offset+i]) << (8*i);
    return value;
  };
  Index end = size-22;
  while (end >= std::max(Index(0), size-65557) &&
         !(word(end) == 0x06054b50 && end+22+word(end+20, 2) == size)) --end;
  if (end < std::max(Index(0), size-65557)) return py::none();
  Index count = word(end+10, 2), pos = word(end+16), stop = end;
  if (word(end+4) || word(end+8, 2) != count || count == 65535 ||
      pos+word(end+12) != end) return py::none();
  py::dict members;
  for (Index i = 0; i < count; ++i) {
    if (word(pos) != 0x02014b50 || pos > stop-46) return py::none();
    Index length = word(pos+28, 2), next = pos+46+length+word(pos+30, 2)+word(pos+32, 2);
    if (next > stop || word(pos+20) == UINT32_MAX || word(pos+24) == UINT32_MAX ||
        word(pos+42) == UINT32_MAX || (word(pos+8, 2)&1)) return py::none();
    std::string name(reinterpret_cast<const char *>(data+pos+46), length);
    if (std::any_of(name.begin(), name.end(), [](unsigned char c) { return c >= 128; })) return py::none();
    if (!metadataOnly || name == "shape.npy" || name == "dtype.npy" ||
        name == "nframes.npy" || name == "omega.npy")
      members[py::str(name)] = py::make_tuple(word(pos+42), word(pos+20), word(pos+10, 2), word(pos+16));
    pos = next;
  }
  return pos == stop ? py::object(members) : py::none();
}

static void csrPixelIds(CArray<Index> offsets, CArray<Index> columns, Index width,
                        py::array_t<int64_t, py::array::c_style> output) {
  checkShape(offsets, {-1}); checkShape(columns, {-1});
  checkShape(output, {columns.size()});
  auto ids = output.mutable_data();
  py::gil_scoped_release release;
  for (Index row = 0; row+1 < offsets.size(); ++row) {
    Index lo = offsets.data()[row], hi = offsets.data()[row+1];
    if (lo < 0 || hi < lo || hi > columns.size()) throw py::value_error("invalid CSR offsets");
    for (Index p = lo; p < hi; ++p) ids[p] = row*width+columns.data()[p];
  }
}

static py::tuple ringMasks(CArray<double> tth, CArray<Index> eta,
                           CArray<double> ranges, Index nEta) {
  checkShape(tth, {-1}); checkShape(eta, {tth.size()}); checkShape(ranges, {-1, 2});
  if (nEta < 0) throw py::value_error("negative eta-bin count");
  Index pixels = tth.size(), rings = ranges.shape(0);
  py::array_t<int64_t> masks({(rings+62)/63, pixels});
  py::array_t<bool> coverage({rings, nEta});
  auto bits = masks.mutable_data(); auto covered = coverage.mutable_data();
  { py::gil_scoped_release release;
    std::fill(bits, bits+masks.size(), 0);
    std::fill(covered, covered+coverage.size(), false);
    for (Index ring = 0; ring < rings; ++ring) for (Index p = 0; p < pixels; ++p) {
      Index bin = eta.data()[p];
      if (bin >= nEta) throw py::value_error("eta bin is out of bounds");
      if (bin >= 0 && ranges.data()[2*ring] <= tth.data()[p] &&
          tth.data()[p] <= ranges.data()[2*ring+1]) {
        bits[(ring/63)*pixels+p] |= int64_t(1) << (ring%63);
        covered[ring*nEta+bin] = true;
      }
    }
  }
  return py::make_tuple(masks, coverage);
}

static void binPixels(CArray<Index> offsets, CArray<Index> ids, CArray<double> values,
    CArray<Index> rows, CArray<int64_t> masks, CArray<int32_t> eta, double threshold,
    py::array_t<double, py::array::c_style> output) {
  checkShape(rows, {-1}); checkShape(offsets, {rows.size()+1}); checkShape(ids, {-1});
  checkShape(values, {ids.size()}); checkShape(masks, {-1}); checkShape(eta, {masks.size()});
  checkShape(output, {-1, -1, -1});
  Index nRing = output.shape(0), nRow = output.shape(1), nEta = output.shape(2);
  Index nPixels = masks.size(), nValues = ids.size(), nFrames = rows.size();
  auto out = output.mutable_data();
  py::gil_scoped_release release;
  for (Index f = 0; f < nFrames; ++f) {
    Index row = rows.data()[f], lo = offsets.data()[f], hi = offsets.data()[f+1];
    if (row < 0 || row >= nRow || lo < 0 || hi < lo || hi > nValues)
      throw py::value_error("invalid frame row or pixel offsets");
    for (Index p = lo; p < hi; ++p) {
      double value = values.data()[p];
      if (value < threshold) continue;
      Index pixel = ids.data()[p];
      if (pixel < 0 || pixel >= nPixels) throw py::value_error("pixel is out of bounds");
      int64_t mask = masks.data()[pixel];
      Index bin = eta.data()[pixel], ring = 0;
      if (mask < 0 || (mask && (bin < 0 || bin >= nEta))) throw py::value_error("invalid ring bin");
      while (mask) {
        if (ring >= nRing) throw py::value_error("ring is out of bounds");
        if (mask & 1) out[(ring*nRow+row)*nEta+bin] += value;
        mask >>= 1; ++ring;
      }
    }
  }
}

// First hit or NaN in row-major window order, in two linear scans per ring.
static py::array_t<int8_t> orientationHitStatus(CArray<double> maps,
                                               Index de, Index dome, double threshold) {
  checkShape(maps, {-1, -1, -1});
  Index height = maps.shape(1), width = maps.shape(2);
  if (de < 0 || dome < 0 || height < 1 || width < 1 || height > INT32_MAX/width)
    throw py::value_error("unsupported ring shape or negative tolerance");
  Index size = height*width;
  py::array_t<int8_t> output(maps.request().shape);
  std::vector<int32_t> scratch(size);
  const double *input = maps.data(); auto status = output.mutable_data();
  py::gil_scoped_release release;
  for (Index ring = 0; ring < maps.shape(0); ++ring) {
    const double *data = input+ring*size; auto out = status+ring*size;
    for (Index row = 0; row < height; ++row) {
      Index next = 0;
      for (Index col = 0; col < width; ++col) {
        Index right = de < width-1-col ? col+de+1 : width;
        next = std::max(next, col > de ? col-de : Index(0));
        while (next < width && !(data[row*width+next] > threshold ||
                                 std::isnan(data[row*width+next]))) ++next;
        scratch[row*width+col] = next < right ? int32_t(row*width+next) : INT32_MAX;
      }
    }
    for (Index col = 0; col < width; ++col) {
      Index next = 0;
      for (Index row = 0; row < height; ++row) {
        Index bottom = dome < height-1-row ? row+dome+1 : height;
        next = std::max(next, row > dome ? row-dome : Index(0));
        while (next < height && scratch[next*width+col] == INT32_MAX) ++next;
        int32_t first = next < bottom ? scratch[next*width+col] : INT32_MAX;
        out[row*width+col] = first == INT32_MAX ? 0 : (data[first] > threshold ? 1 : -1);
      }
    }
  }
  return output;
}

static Index rangeIndex(double value, const double *bounds, Index count) {
  if (value < bounds[0] || value >= bounds[count-1]) return -2;
  return std::upper_bound(bounds, bounds+count, value)-bounds;
}

static double orientationMod(double value, double offset) {
  double result = value-offset;
  if (result >= 0 && result < 2.*pi) return result+offset;
  if (result >= -2.*pi && result < 0) return result+2.*pi+offset;
  if (result >= 2.*pi && result < 4.*pi) return result-2.*pi+offset;
  result = std::fmod(result, 2.*pi);
  return result < 0 ? result+2.*pi+offset : result+offset;
}

static py::array_t<double> scoreOrientationBins(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> bins, CArray<int8_t> status) {
  checkShape(bins, {-1, 2, -1}); checkShape(status, {-1, -1, -1});
  Index n = bins.shape(0), count = 2*bins.shape(2), size = status.size();
  py::array_t<double> scores(n); auto out = scores.mutable_data();
  const auto *indices = static_cast<const char *>(bins.request().ptr);
  py::gil_scoped_release release;
  for (Index q = 0; q < n; ++q) {
    Index hits = 0, total = 0;
    for (Index h = 0; h < count; ++h) {
      int32_t cell;
      std::memcpy(&cell, indices+sizeof(cell)*(q*count+h), sizeof(cell));
      if (cell < 0) continue;
      if (cell >= size) throw py::value_error("reflection bin is out of bounds");
      int hit = status.data()[cell];
      total += hit >= 0; hits += std::max(hit, 0);
    }
    out[q] = total ? double(hits)/total : 0.;
  }
  return scores;
}

static Index orientationBin(double value, const double *edges, Index count, double scale) {
  if (!(value >= edges[0] && value < edges[count])) return -1;
  double estimate = (value-edges[0])*scale;
  Index guess = estimate >= 0 && estimate < count ? Index(estimate) : 0;
  return value >= edges[guess] && value < edges[guess+1] ? guess :
         rangeIndex(value, edges, count+1)-1;
}

static bool nearBinEdge(double value, const double *edges, Index count, Index bin) {
  if (bin < 0 || bin >= count) bin = value < edges[0] ? 0 : count-1;
  return std::abs(value-edges[bin]) <= 1e-7 || std::abs(value-edges[bin+1]) <= 1e-7;
}

static int packedHitStatus(const int8_t *lookup, Index ring, Index o, Index e,
                           Index nome, Index neta) {
  Index tile = (ring*((nome+7)/8)+o/8)*((neta+7)/8)+e/8;
  Index local = (o%8)*8+e%8;
  int code = (uint8_t(lookup[16*tile+local/4]) >> (2*(local%4))) & 3;
  return (code >> 1)-int(!(code & 1));
}

#ifdef HEXRD_AVX2
struct VectorScorer {
  const double *etaEdges, *omeEdges, *validEta, *validOme;
  const int *etaValid, *omeValid;
  const double *reciprocal, *lengths, *visibility;
  const Index *rings, *opposites, *mates;
  const int8_t *lookup;
  Index neta, nome, nhkl, nvalidEta, nvalidOme;
  double etaScale, omeScale, offset, wavelength;
  const OscillationGeometry *geometry;

  HEXRD_AVX2 unsigned score(const double *q, Index stride, Index step, double *output) const {
    using B = scoring_simd::Batch;
    using namespace scoring_simd;
    auto load = [&](const double *p) HEXRD_AVX2 HEXRD_FORCEINLINE {
      return step == 1 ? B::load_unaligned(p) :
          B(_mm256_i64gather_pd(p, _mm256_set_epi64x(3*step, 2*step, step, 0), 8));
    };
    B a = load(q), b = load(q+stride), c = load(q+2*stride), d = load(q+3*stride);
    B r[9] = {a*a+b*b-c*c-d*d, B(2.)*b*c-B(2.)*a*d, B(2.)*a*c+B(2.)*b*d,
              B(2.)*a*d+B(2.)*b*c, a*a-b*b+c*c-d*d, B(2.)*c*d-B(2.)*a*b,
              B(2.)*b*d-B(2.)*a*c, B(2.)*a*b+B(2.)*c*d, a*a-b*b-c*c+d*d};
    B active = B(0.) <= B(1.);
    const bool halfPeriod = omeEdges[0] == 0. &&
        std::abs(omeEdges[nome]-2.*pi) < 1e-12 && offset == -pi;
    for (int i = 0; i < 3; ++i) for (int j = 0; j <= i; ++j) {
      B dot(0.);
      for (int k = 0; k < 3; ++k) dot = dot+r[3*k+i]*r[3*k+j];
      active = active & (abs(dot-B(i == j ? 1. : 0.)) <= B(1e-14));
    }
    __m128i hits = _mm_setzero_si128(), total = _mm_setzero_si128();
    auto exact = [&](B mask, Index h, int solution, B opposite) HEXRD_AVX2 __attribute__((noinline, cold)) {
      unsigned lanes = _mm256_movemask_pd(mask.value);
      if (!lanes) return;
      unsigned matesMask = _mm256_movemask_pd(opposite.value);
      double rotations[9][4];
      for (int k = 0; k < 9; ++k) r[k].store_unaligned(rotations[k]);
      int32_t hv[4] = {}, tv[4] = {};
      for (int j = 0; j < 4; ++j) if (lanes & (1u << j)) {
        Index reflection = matesMask & (1u << j) ? mates[h] : h;
        double rotation[9], angles[6];
        for (int k = 0; k < 9; ++k) rotation[k] = rotations[k][j];
        geometry->predict(1, reciprocal+3*reflection, rotation, angles, angles+3);
        const double *a = angles+3*solution;
        if (std::isnan(a[0])) continue;
        double e = orientationMod(a[1], -pi), o = orientationMod(a[2], offset);
        Index ei = orientationBin(e, etaEdges, neta, etaScale);
        Index oi = orientationBin(o, omeEdges, nome, omeScale);
        if (ei < 0 || oi < 0 || !(rangeIndex(e, validEta, nvalidEta)&1) ||
            !(rangeIndex(o, validOme, nvalidOme)&1)) continue;
        int hit = packedHitStatus(lookup, rings[reflection], oi, ei, nome, neta);
        tv[j] = hit >= 0; hv[j] = hit > 0;
      }
      hits = _mm_add_epi32(hits, _mm_loadu_si128(reinterpret_cast<const __m128i *>(hv)));
      total = _mm_add_epi32(total, _mm_loadu_si128(reinterpret_cast<const __m128i *>(tv)));
    };
    auto bin = [&](B eta, B omega, B present, Index h, int solution, B opposite) HEXRD_AVX2 HEXRD_FORCEINLINE {
      auto wrap = [](B angle, double origin) HEXRD_AVX2 HEXRD_FORCEINLINE {
        auto x = angle-B(origin);
        x = select(x < B(0.), x+B(2.*pi), x);
        x = select(B(2.*pi) <= x, x-B(2.*pi), x);
        return x+B(origin);
      };
      present = present & active;
      if (!any(present)) return;
      // Predicted eta is in [-pi, pi]; folded omega is in (0, pi).
      auto e = select(B(pi) <= eta, eta-B(2.*pi), eta);
      auto o = halfPeriod && mates[h] >= 0 ? omega : wrap(omega, offset);
      auto normalized = (B(-pi) <= e) & (e < B(pi)) & (B(offset) <= o) & (o < B(offset+2.*pi));
      active = active & ~(present & ~normalized);
      present = present & active;
      auto es = (e-B(etaEdges[0]))*B(etaScale), os = (o-B(omeEdges[0]))*B(omeScale);
      auto ed = es-floor(es), od = os-floor(os);
      auto inside = present & (es > B(0.)) & (es < B(double(neta))) &
                    (os > B(0.)) & (os < B(double(nome))) &
                    (min(ed, B(1.)-ed) > B(2e-7*etaScale)) &
                    (min(od, B(1.)-od) > B(2e-7*omeScale));
      auto outside = present & ((e < B(etaEdges[0]-2e-7)) | (e > B(etaEdges[neta]+2e-7)) |
                                 (o < B(omeEdges[0]-2e-7)) | (o > B(omeEdges[nome]+2e-7)));
      auto boundary = present & ~(inside | outside);
      if (any(boundary)) exact(boundary, h, solution, opposite);
      inside = inside & active;
      if (!any(inside)) return;
      auto ei = _mm256_cvttpd_epi32(select(inside, es, B(0.)).value);
      auto oi = _mm256_cvttpd_epi32(select(inside, os, B(0.)).value);
      auto ve = _mm_i32gather_epi32(etaValid, ei, 4), vo = _mm_i32gather_epi32(omeValid, oi, 4);
      auto zero = _mm_setzero_si128(), mask = mask128(inside);
      auto accepted = _mm_and_si128(mask, _mm_and_si128(_mm_cmpgt_epi32(ve, zero), _mm_cmpgt_epi32(vo, zero)));
      auto unknown = _mm_and_si128(mask, _mm_or_si128(_mm_cmpgt_epi32(zero, ve), _mm_cmpgt_epi32(zero, vo)));
      // Resolve bins that straddle valid-range boundaries with the exact predictor.
      if (!_mm_testz_si128(unknown, unknown))
        exact(B(_mm256_castsi256_pd(_mm256_cvtepi32_epi64(unknown))), h, solution, opposite);
      auto set = [](Index x) HEXRD_AVX2 { return _mm_set1_epi32(int32_t(x)); };
      auto row = _mm_add_epi32(_mm_srli_epi32(oi, 3), set(rings[h]*((nome+7)/8)));
      auto tile = _mm_add_epi32(_mm_mullo_epi32(row, set((neta+7)/8)), _mm_srli_epi32(ei, 3));
      auto word = _mm_add_epi32(_mm_slli_epi32(tile, 2), _mm_srli_epi32(_mm_and_si128(oi, set(7)), 1));
      auto shift = _mm_add_epi32(_mm_slli_epi32(_mm_and_si128(oi, set(1)), 4),
                                 _mm_slli_epi32(_mm_and_si128(ei, set(7)), 1));
      auto bits = _mm_mask_i32gather_epi32(zero, reinterpret_cast<const int *>(lookup), word, accepted, 4);
      auto codes = _mm_and_si128(_mm_srlv_epi32(bits, shift), set(3));
      total = _mm_sub_epi32(total, _mm_cmpgt_epi32(codes, zero));
      hits = _mm_sub_epi32(hits, _mm_cmpeq_epi32(codes, set(3)));
    };
    for (Index h = 0; h < nhkl; ++h) {
      if (!any(active)) break;
      if (opposites[h] >= 0) continue;
      B gx(reciprocal[3*h]), gy(reciprocal[3*h+1]), gz(reciprocal[3*h+2]);
      auto x = r[0]*gx+r[1]*gy+r[2]*gz, y = r[3]*gx+r[4]*gy+r[5]*gz;
      auto z = r[6]*gx+r[7]*gy+r[8]*gz;
      auto present = active & (B(visibility[2*h]) <= abs(y)) & (abs(y) <= B(visibility[2*h+1]));
      if (!any(present)) continue;
      double squared = lengths[h]*lengths[h], st = 0.5*wavelength*squared;
      auto root2 = x*x+z*z-B(st*st);
      active = active & ~(present & (~isfinite(root2) | (abs(root2) <= B(1e-8*squared))));
      present = present & active & (root2 > B(1e-8*squared));
      if (!any(present)) continue;
      x = select(present, x, B(1.)); y = select(present, y, B(0.)); z = select(present, z, B(0.));
      auto sine = select(present, B(st), B(0.));
      auto root = sqrt(select(present, root2, B(1.)));
      auto p = -sine*x+root*z, u = root*x+sine*z;
      auto v = -sine*x-root*z, w = -root*x+sine*z;
      auto finite = isfinite(p) & isfinite(u) & isfinite(v) & isfinite(w);
      if (any(present & ~finite)) {
        active = active & ~(present & ~finite); present = present & active;
        p = select(present, p, B(0.)); u = select(present, u, B(1.));
        v = select(present, v, B(0.)); w = select(present, w, B(1.));
      }
      auto eta0 = atan2(y, root), eta1 = copysign(B(pi), y)-eta0;
      auto omega0 = atan2(p, u), omega1 = atan2(v, w);
      for (int solution = 0; solution < 2; ++solution) {
        B e = solution ? eta1 : eta0, o = solution ? omega1 : omega0;
        if (halfPeriod && mates[h] >= 0) {
          // Only one opposite reflection lies inside both periods.
          active = active & ~(present & (abs(o) <= B(2e-7)));
          auto negative = o < B(0.);
          bin(select(negative, -e, e), select(negative, o+B(pi), o), present, h, solution, negative);
        } else {
          bin(e, o, present, h, solution, B(0.));
          if (mates[h] >= 0) bin(-e, o+B(pi), present, mates[h], solution, B(0.));
        }
      }
    }
    int32_t hv[4], tv[4];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(hv), hits);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(tv), total);
    unsigned ready = _mm256_movemask_pd(active.value);
    for (int j = 0; j < 4; ++j) if (ready & (1u << j))
      output[j] = tv[j] ? double(hv[j])/tv[j] : 0.;
    return ready;
  }
};
#endif

static py::array_t<double> scoreOrientations(
    StridedArray<double> rmats,
    CArray<double> hkls,
    CArray<double> bmat, double wavelength, CArray<double> beam,
    CArray<double> etaEdges, CArray<double> omeEdges, CArray<double> validEta,
    CArray<double> validOme, double offset, StridedArray<double> maps, CArray<Index> rings,
    Index de, Index dome, double threshold, py::object statusObject, py::object binsObject) {
  bool quaternions = rmats.ndim() == 2;
  if (quaternions) checkShape(rmats, {4, -1});
  else { checkShape(rmats, {-1, 3, 3}); rmats = CArray<double>::ensure(rmats); }
  checkShape(beam, {3}); checkShape(hkls, {-1, 3});
  checkShape(maps, {-1, -1, -1}); checkShape(rings, {hkls.shape(0)});
  Index n = rmats.shape(quaternions ? 1 : 0), nhkl = hkls.shape(0);
  Index nome = maps.shape(1), neta = maps.shape(2);
  Index componentStride = rmats.strides(0)/Index(sizeof(double)), trialStride = rmats.strides(1)/Index(sizeof(double));
  checkShape(etaEdges, {neta+1}); checkShape(omeEdges, {nome+1});
  checkShape(validEta, {-1}); checkShape(validOme, {-1});
  Index nvalidEta = validEta.size(), nvalidOme = validOme.size();
  double etaScale = neta/(etaEdges.data()[neta]-etaEdges.data()[0]);
  double omeScale = nome/(omeEdges.data()[nome]-omeEdges.data()[0]);
  double norm = std::sqrt(beam.data()[0]*beam.data()[0] + beam.data()[1]*beam.data()[1] +
                          beam.data()[2]*beam.data()[2]);
  if (de < 0 || dome < 0 || nome < 1 || neta < 1 || nvalidEta < 2 ||
      nvalidOme < 2 || nvalidEta%2 || nvalidOme%2 ||
      !std::isfinite(norm) || norm <= eps || std::abs(beam.data()[0]/norm) >= 1.-1.5e-8)
    throw py::value_error("invalid scoring ranges, beam or tolerances");
  for (Index i = 0; i < rings.size(); ++i) if (rings.data()[i] < 0 || rings.data()[i] >= maps.shape(0))
    throw py::value_error("reflection ring index is out of bounds");
  CArray<int8_t> status;
  bool packed = false;
  if (!statusObject.is_none()) {
    status = CArray<int8_t>::ensure(statusObject);
    if (!status) throw py::value_error("invalid hit status map");
    packed = status.ndim() == 1;
    if (packed) checkShape(status, {maps.shape(0)*((nome+7)/8)*((neta+7)/8)*16});
    else checkShape(status, {maps.shape(0), nome, neta});
  } else maps = CArray<double>::ensure(maps);
  auto reciprocal = reciprocalVectors(hkls, bmat);
  std::vector<Index> opposites(nhkl, -1);
  for (Index h = 0; h < nhkl; ++h) for (Index j = 0; j < h; ++j)
    if (reciprocal[3*h] == -reciprocal[3*j] && reciprocal[3*h+1] == -reciprocal[3*j+1] &&
        reciprocal[3*h+2] == -reciprocal[3*j+2]) { opposites[h] = j; break; }
  std::vector<double> lengths(nhkl);
  for (Index h = 0; h < nhkl; ++h) {
    double squared = 0;
    for (int j = 0; j < 3; ++j) squared += reciprocal[3*h+j]*reciprocal[3*h+j];
    lengths[h] = std::sqrt(squared);
  }
  bool nominal = std::abs(beam.data()[0]/norm) <= 1e-14 &&
                 std::abs(beam.data()[1]/norm) <= 1e-14 && beam.data()[2] < 0 &&
                 std::abs(offset) < 100.;
  auto validity = [](const double *edges, Index count, const double *bounds, Index size) {
    std::vector<int> result(count, -1);
    for (Index i = 0; i < count; ++i) {
      Index lo = std::upper_bound(bounds, bounds+size, edges[i]-1e-7)-bounds;
      Index hi = std::upper_bound(bounds, bounds+size, edges[i+1]+1e-7)-bounds;
      if (lo == hi) result[i] = lo%2;
    }
    return result;
  };
  auto etaValid = validity(etaEdges.data(), neta, validEta.data(), nvalidEta);
  auto omeValid = validity(omeEdges.data(), nome, validOme.data(), nvalidOme);
  bool pairedBins = nome%2 == 0 && std::abs(omeEdges.data()[0]-offset) < 1e-12 &&
      std::abs(etaEdges.data()[0]+pi) < 1e-12 &&
      std::abs(etaEdges.data()[neta]-pi) < 1e-12;
  for (Index i = 0; i <= neta && pairedBins; ++i)
    pairedBins &= std::abs(etaEdges.data()[i]-(-pi+2.*pi*i/neta)) < 1e-12;
  for (Index i = 0; i <= nome && pairedBins; ++i)
    pairedBins &= std::abs(omeEdges.data()[i]-(omeEdges.data()[0]+2.*pi*i/nome)) < 1e-12;
#ifdef HEXRD_AVX2
  bool vectorScores = nominal && quaternions && packed && binsObject.is_none() &&
      __builtin_cpu_supports("avx2") && status.size()/4 <= INT32_MAX && nhkl < INT32_MAX/2 &&
      neta < INT32_MAX && nome < INT32_MAX && etaScale > 0 && omeScale > 0 &&
      etaScale < 1e6 && omeScale < 1e6;
  std::vector<Index> mates(nhkl, -1);
  for (Index h = 0; h < nhkl && vectorScores; ++h) {
    double st = 0.5*wavelength*lengths[h];
    vectorScores &= lengths[h] > eps && std::isfinite(lengths[h]*lengths[h]) && 1.-st*st > 1e-8;
    Index parent = opposites[h];
    if (parent >= 0) {
      vectorScores &= opposites[parent] < 0 && mates[parent] < 0 && rings.data()[h] == rings.data()[parent];
      mates[parent] = h;
    }
  }
  for (Index i = 0; i <= neta && vectorScores; ++i)
    vectorScores &= std::abs(etaEdges.data()[i]-(etaEdges.data()[0]+i/etaScale)) < 1e-12;
  for (Index i = 0; i <= nome && vectorScores; ++i)
    vectorScores &= std::abs(omeEdges.data()[i]-(omeEdges.data()[0]+i/omeScale)) < 1e-12;
#endif
  std::vector<Index> reflectionBins(4*nhkl, -1);
  std::vector<double> angles(std::max(Index(1), 6*nhkl));
  py::array_t<double> scores(n); double *output = scores.mutable_data();
  const double vinv[6] = {1, 1, 1, 0, 0, 0}, eta[3] = {1, 0, 0};
  OscillationGeometry geometry(0., wavelength, vinv, beam.data(), eta);
  const int8_t *lookup = statusObject.is_none() ? nullptr : status.data();
  CArray<int32_t> bins;
  int32_t *cells = nullptr;
  if (!binsObject.is_none()) {
    bins = CArray<int32_t>::ensure(binsObject);
    if (!bins || maps.size() > INT32_MAX) throw py::value_error("unsupported reflection-bin array");
    checkShape(bins, {n, 2, nhkl});
    cells = bins.mutable_data();
    std::fill(cells, cells+bins.size(), -1);
  }
  py::gil_scoped_release release;
  auto statusAt = [&](Index ring, Index o, Index e) {
    if (!packed) return int(lookup[(ring*nome+o)*neta+e]);
    return packedHitStatus(lookup, ring, o, e, nome, neta);
  };
  // |g_y| = |g| cos(theta) |sin(eta)|; prune by coverage, not intensity.
  // Expand angular and vector bounds to retain boundary cases.
  std::vector<double> visibility(2*nhkl);
  if (lookup && !cells && nominal) {
    std::vector<double> limits(2*maps.shape(0));
    for (Index ring = 0; ring < maps.shape(0); ++ring) {
      double lower = 1., upper = 0.;
      for (Index e = 0; e < neta; ++e) {
        if (etaValid[e] == 0) continue;
        bool visible = false;
        for (Index o = 0; o < nome && !visible; ++o)
          visible = omeValid[o] != 0 && statusAt(ring, o, e) >= 0;
        if (!visible) continue;
        double lo = etaEdges.data()[e]-1e-7, hi = etaEdges.data()[e+1]+1e-7;
        lower = std::min(lower, std::min(std::abs(std::sin(lo)), std::abs(std::sin(hi))));
        upper = std::max(upper, std::max(std::abs(std::sin(lo)), std::abs(std::sin(hi))));
        if (std::ceil(lo/pi) <= std::floor(hi/pi)) lower = 0.;
        if (std::ceil((lo-pi/2.)/pi) <= std::floor((hi-pi/2.)/pi)) upper = 1.;
      }
      limits[2*ring] = lower; limits[2*ring+1] = upper;
    }
    for (Index h = 0; h < nhkl; ++h) {
      double st = 0.5*wavelength*lengths[h];
      double transverse = lengths[h] > eps && 1.-st*st > 1e-8 ?
          lengths[h]*std::sqrt(1.-st*st) : NAN;
      visibility[2*h] = limits[2*rings.data()[h]]*transverse-1e-7*lengths[h];
      visibility[2*h+1] = limits[2*rings.data()[h]+1]*transverse+1e-7*lengths[h];
    }
  }
#ifdef HEXRD_AVX2
  VectorScorer vectorScorer{etaEdges.data(), omeEdges.data(), validEta.data(), validOme.data(),
      etaValid.data(), omeValid.data(), reciprocal.data(), lengths.data(), visibility.data(),
      rings.data(), opposites.data(), mates.data(), lookup, neta, nome, nhkl, nvalidEta, nvalidOme,
      etaScale, omeScale, offset, wavelength, &geometry};
  Index vectorEnd = 0; unsigned vectorReady = 0;
#endif
  for (Index q = 0; q < n; ++q) {
#ifdef HEXRD_AVX2
    if (vectorScores) {
      if (q >= vectorEnd && q+4 <= n) {
        vectorReady = vectorScorer.score(rmats.data()+q*trialStride, componentStride, trialStride, output+q);
        vectorEnd = q+4;
        if (vectorReady == 15) { q += 3; continue; }
      }
      if (q < vectorEnd) {
        bool ready = vectorReady & 1; vectorReady >>= 1;
        if (ready) continue;
      }
    }
#endif
    double rotation[9];
    const double *rmat = rotation;
    if (quaternions) {
      const double *v = rmats.data()+q*trialStride;
      quaternionRotation(v[0], v[componentStride], v[2*componentStride], v[3*componentStride], rotation);
    } else rmat = rmats.data()+9*q;
    bool fast = nominal;
    for (int i = 0; i < 3 && fast; ++i) for (int j = 0; j <= i; ++j) {
      double dot = 0;
      for (int k = 0; k < 3; ++k) dot += rmat[3*k+i]*rmat[3*k+j];
      fast &= std::abs(dot-(i == j ? 1. : 0.)) <= 1e-14;
    }
    geometry.predict(nhkl, reciprocal.data(), rmat, angles.data(), angles.data()+3*nhkl,
                     fast ? lengths.data() : nullptr, opposites.data(),
                     fast && lookup && !cells ? visibility.data() : nullptr);
    Index hits = 0, total = 0;
    std::fill(reflectionBins.begin(), reflectionBins.end(), -1);
    for (Index solution = 0; solution < 2; ++solution) for (Index h = 0; h < nhkl; ++h) {
      const double *a = angles.data()+3*(solution*nhkl+h);
      if (std::isnan(a[0])) continue;
      Index ei = -1, oi = -1, parent = opposites[h];
      if (fast && pairedBins && parent >= 0 && a[0] == -INFINITY) {
        const Index *previous = reflectionBins.data()+2*(solution*nhkl+parent);
        if (previous[0] >= 0) {
          ei = neta-1-previous[0];
          oi = previous[1] < nome/2 ? previous[1]+nome/2 : previous[1]-nome/2;
          if (etaValid[ei] < 0 || omeValid[oi] < 0) ei = -1;
        }
      }
      Index ve, vo;
      if (ei >= 0) { ve = etaValid[ei]; vo = omeValid[oi]; }
      else {
        double e = orientationMod(a[1], -pi), o = orientationMod(a[2], offset);
        ei = orientationBin(e, etaEdges.data(), neta, etaScale);
        oi = orientationBin(o, omeEdges.data(), nome, omeScale);
        ve = ei >= 0 ? etaValid[ei] : -1;
        vo = oi >= 0 ? omeValid[oi] : -1;
        bool boundary = fast && (nearBinEdge(e, etaEdges.data(), neta, ei) ||
            nearBinEdge(o, omeEdges.data(), nome, oi) ||
            (ve < 0 && nearBinEdge(e, validEta.data(), nvalidEta-1,
                                  rangeIndex(e, validEta.data(), nvalidEta)-1)) ||
            (vo < 0 && nearBinEdge(o, validOme.data(), nvalidOme-1,
                                  rangeIndex(o, validOme.data(), nvalidOme)-1)));
        if (boundary) {
          double exact[6];
          geometry.predict(1, reciprocal.data()+3*h, rmat, exact, exact+3);
          a = exact+3*solution;
          if (std::isnan(a[0])) continue;
          e = orientationMod(a[1], -pi); o = orientationMod(a[2], offset);
          ei = orientationBin(e, etaEdges.data(), neta, etaScale);
          oi = orientationBin(o, omeEdges.data(), nome, omeScale);
          ve = rangeIndex(e, validEta.data(), nvalidEta);
          vo = rangeIndex(o, validOme.data(), nvalidOme);
        } else if (pairedBins && a[0] == -INFINITY && ei >= 0 && oi >= 0) {
          reflectionBins[2*(solution*nhkl+h)] = ei;
          reflectionBins[2*(solution*nhkl+h)+1] = oi;
        }
        if (ve < 0) ve = rangeIndex(e, validEta.data(), nvalidEta);
        if (vo < 0) vo = rangeIndex(o, validOme.data(), nvalidOme);
      }
      if (!(ve&1) || !(vo&1)) continue;
      if (ei < 0 || oi < 0 || ei >= neta || oi >= nome) continue;
      Index cell = (rings.data()[h]*nome+oi)*neta+ei;
      if (cells) cells[(2*q+solution)*nhkl+h] = int32_t(cell);
      int hit = lookup ? statusAt(rings.data()[h], oi, ei) : 0;
      if (!lookup) {
        Index ostop = dome < nome-1-oi ? oi+dome+1 : nome;
        Index estop = de < neta-1-ei ? ei+de+1 : neta;
        for (Index i = std::max(Index(0), oi-dome); i < ostop && hit == 0; ++i) {
          const double *row = maps.data()+(rings.data()[h]*nome+i)*neta;
          for (Index j = std::max(Index(0), ei-de); j < estop; ++j) {
            if (row[j] > threshold) { hit = 1; break; }
            if (std::isnan(row[j])) { hit = -1; break; }
          }
        }
      }
      if (hit >= 0) { ++total; hits += hit; }
    }
    output[q] = total ? double(hits)/total : 0.;
  }
  return scores;
}

static double quatDistance(CArray<double> q1, CArray<double> q2, CArray<double> sym) {
  checkShape(q1, {4}); checkShape(q2, {4}); checkShape(sym, {-1, 4});
  const double *a = q1.data(), *b = q2.data(); double maximum = 0;
  py::gil_scoped_release release;
  for (Index i = 0; i < sym.shape(0); ++i) {
    const double *s = sym.data()+4*i;
    double r[4] = {b[0]*s[0]-b[1]*s[1]-b[2]*s[2]-b[3]*s[3],
                   b[1]*s[0]+b[0]*s[1]-b[3]*s[2]+b[2]*s[3],
                   b[2]*s[0]+b[3]*s[1]+b[0]*s[2]-b[1]*s[3],
                   b[3]*s[0]-b[2]*s[1]+b[1]*s[2]+b[0]*s[3]};
    double value = std::abs(a[0]*r[0]+a[1]*r[1]+a[2]*r[2]+a[3]*r[3]);
    if (value > maximum) maximum = value;
  }
  return maximum <= 1. ? 2.*std::acos(maximum) : (maximum-1. < 1e-12 ? 0. : NAN);
}

PYBIND11_MODULE(transforms, m) {
  if (_import_array() < 0) throw py::error_already_set();
  m.def("_reduceFiber", &reduceFiber);
  m.doc() = "HEXRD transforms plugin";
  m.def("oscillAnglesOfHKLs", &oscillAnglesOfHKLs);
  m.def("projectGvecs", &projectGvecs<true>, py::arg("vectors"), py::arg("rmat_d"),
        py::arg("rmat_s"), py::arg("rmat_c"), py::arg("tvec_d"), py::arg("tvec_s"),
        py::arg("tvec_c"), py::arg("beam") = py::none());
  m.def("projectGvecsArray", &projectGvecs<false>, py::arg("vectors"), py::arg("rmat_d"),
        py::arg("rmat_s"), py::arg("rmat_c"), py::arg("tvec_d"), py::arg("tvec_s"),
        py::arg("tvec_c"), py::arg("beam") = py::none());
  m.def("detectorXYToGvec", &detectorXYToGvec);
  m.def("makeEtaFrameRotMat", &makeEtaFrameRotMat);
  m.def("rotate_vecs_about_axis", &rotateVectors);
  m.def("quat_distance", &quatDistance);
  m.def("quaternionRotations", &quaternionRotations);
  m.def("_npzIndex", &npzIndex);
  m.def("_csrPixelIds", &csrPixelIds, py::arg("offsets"), py::arg("columns"),
        py::arg("width"), py::arg("output").noconvert());
  m.def("_ringMasks", &ringMasks);
  m.def("_binPixels", &binPixels, py::arg("offsets"), py::arg("ids"), py::arg("values"),
        py::arg("rows"), py::arg("masks"), py::arg("eta"), py::arg("threshold"),
        py::arg("output").noconvert());
  m.def("orientationHitStatus", &orientationHitStatus);
  m.def("_scoreOrientationBins", &scoreOrientationBins);
  m.def("scoreOrientations", &scoreOrientations,
        py::arg("rmats"), py::arg("hkls"), py::arg("bmat"), py::arg("wavelength"),
        py::arg("beam"), py::arg("eta_edges"), py::arg("omega_edges"),
        py::arg("valid_eta"), py::arg("valid_omega"), py::arg("omega_offset"),
        py::arg("ring_maps"), py::arg("ring_for_hkl"), py::arg("dpix_eta"),
        py::arg("dpix_omega"), py::arg("threshold"), py::arg("hit_status") = py::none(),
        py::arg("bin_indices") = py::none());

  m.def("make_binary_rot_mat", &makeBinaryRotMat,
        "Function that computes a rotation matrix from a binary vector");
  m.def("make_rot_mat_of_exp_map", &makeRotMatOfExpMap,
        "Function that computes a rotation matrix from an exponential map");
  m.def("makeOscillRotMat", &makeOscillRotMat,
        "Function that generates a collection of rotation matrices from two "
        "angles (chi, ome)");
  m.def("anglesToDetectorXY", &anglesToDetectorXY,
        "Converts angles to detector xy coordinates.");
  m.def("anglesToGVec", &anglesToGVec,
        "Function that converts angles to g-vectors");
  m.def("anglesToDVec", &anglesToDVec,
        "Function that converts angles to d-vectors");
  m.def("gvecToDetectorXY", &gvecToDetectorXY,
        "A function that converts gVec to detector XY");
  m.def("gvec_to_detector_xy_one", &gvecToDetectorXYOne,
        "Function that converts g-vectors to detector xy coordinates");
  m.def("gvecToDetectorXYFromAngles", &gvecToDetectorXYFromAngles,
        "Function that converts g-vectors to detector xy coordinates, given "
        "rotation axes");
}
