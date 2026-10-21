#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <xsimd/xsimd.hpp>

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

struct OscillationGeometry {
  double beam[3], cchi, schi, wavelength;
  std::array<double, 9> frame = {};
  const double *vinv;
  bool consistent;

  OscillationGeometry(double chi, double wave, const double *stretch,
                      const double *bvec, const double *evec)
      : cchi(std::cos(chi)), schi(std::sin(chi)), wavelength(wave), vinv(stretch) {
    double eta[3], dot = 0;
    normalize3(bvec, beam);
    normalize3(evec, eta);
    for (int j = 0; j < 3; ++j) dot += beam[j]*eta[j];
    consistent = std::abs(dot) < 1.0 - 1.5e-8;
    if (consistent) frame = beamFrame(beam, eta);
  }

  void predict(Index n, const double *reciprocal, const double *rmat,
               double *out0, double *out1) const {
    const double cchi = this->cchi, schi = this->schi, wavelength = this->wavelength;
    const auto frame = this->frame;
    const std::array<double, 3> beam = {this->beam[0], this->beam[1], this->beam[2]};
    std::array<double, 6> stretch;
    std::copy(vinv, vinv+6, stretch.begin());
    for (Index i = 0; i < n; ++i) {
      double gc[3], gs[3], tmp[3];
      matvec3(rmat, reciprocal+3*i, gs);
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

static py::array_t<double> quaternionRotations(CArray<double> quats) {
  checkShape(quats, {-1, 4});
  py::array_t<double> output({quats.shape(0), Index(3), Index(3)});
  const double *q = quats.data(); double *r = output.mutable_data();
  py::gil_scoped_release release;
  for (Index i = 0; i < quats.shape(0); ++i, q += 4, r += 9) {
    double a = q[0], b = q[1], c = q[2], d = q[3];
    r[0] = a*a + b*b - c*c - d*d;
    r[1] = 2*b*c - 2*a*d; r[2] = 2*a*c + 2*b*d;
    r[3] = 2*a*d + 2*b*c; r[4] = a*a - b*b + c*c - d*d;
    r[5] = 2*c*d - 2*a*b; r[6] = 2*b*d - 2*a*c;
    r[7] = 2*a*b + 2*c*d; r[8] = a*a - b*b - c*c + d*d;
  }
  return output;
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
  double result = std::fmod(value-offset, 2.*pi);
  return result < 0 ? result+2.*pi+offset : result+offset;
}

static py::array_t<double> scoreOrientations(CArray<double> rmats, CArray<double> hkls,
    CArray<double> bmat, double wavelength, CArray<double> beam,
    CArray<double> etaEdges, CArray<double> omeEdges, CArray<double> validEta,
    CArray<double> validOme, double offset, CArray<double> maps, CArray<Index> rings,
    Index de, Index dome, double threshold, py::object statusObject) {
  checkShape(rmats, {-1, 3, 3}); checkShape(beam, {3});
  checkShape(hkls, {-1, 3});
  checkShape(maps, {-1, -1, -1}); checkShape(rings, {hkls.shape(0)});
  Index n = rmats.shape(0), nhkl = hkls.shape(0), nome = maps.shape(1), neta = maps.shape(2);
  checkShape(etaEdges, {neta+1}); checkShape(omeEdges, {nome+1});
  checkShape(validEta, {-1}); checkShape(validOme, {-1});
  Index nvalidEta = validEta.size(), nvalidOme = validOme.size();
  double norm = std::sqrt(beam.data()[0]*beam.data()[0] + beam.data()[1]*beam.data()[1] +
                          beam.data()[2]*beam.data()[2]);
  if (de < 0 || dome < 0 || nome < 1 || neta < 1 || nvalidEta < 2 ||
      nvalidOme < 2 || nvalidEta%2 || nvalidOme%2 ||
      !std::isfinite(norm) || norm <= eps || std::abs(beam.data()[0]/norm) >= 1.-1.5e-8)
    throw py::value_error("invalid scoring ranges, beam or tolerances");
  for (Index i = 0; i < rings.size(); ++i) if (rings.data()[i] < 0 || rings.data()[i] >= maps.shape(0))
    throw py::value_error("reflection ring index is out of bounds");
  CArray<int8_t> status;
  if (!statusObject.is_none()) {
    status = CArray<int8_t>::ensure(statusObject);
    if (!status) throw py::value_error("invalid hit status map");
    checkShape(status, {maps.shape(0), nome, neta});
  }
  auto reciprocal = reciprocalVectors(hkls, bmat);
  std::vector<double> angles(std::max(Index(1), 6*nhkl));
  py::array_t<double> scores(n); double *output = scores.mutable_data();
  const double vinv[6] = {1, 1, 1, 0, 0, 0}, eta[3] = {1, 0, 0};
  OscillationGeometry geometry(0., wavelength, vinv, beam.data(), eta);
  const int8_t *lookup = statusObject.is_none() ? nullptr : status.data();
  py::gil_scoped_release release;
  for (Index q = 0; q < n; ++q) {
    geometry.predict(nhkl, reciprocal.data(), rmats.data()+9*q, angles.data(), angles.data()+3*nhkl);
    Index hits = 0, total = 0;
    for (Index solution = 0; solution < 2; ++solution) for (Index h = 0; h < nhkl; ++h) {
      const double *a = angles.data()+3*(solution*nhkl+h);
      if (std::isnan(a[0])) continue;
      double e = orientationMod(a[1], -pi), o = orientationMod(a[2], offset);
      if (!(rangeIndex(e, validEta.data(), nvalidEta)&1) ||
          !(rangeIndex(o, validOme.data(), nvalidOme)&1)) continue;
      Index ei = rangeIndex(e, etaEdges.data(), neta+1)-1;
      Index oi = rangeIndex(o, omeEdges.data(), nome+1)-1;
      if (ei < 0 || oi < 0 || ei >= neta || oi >= nome) continue;
      Index cell = (rings.data()[h]*nome+oi)*neta+ei;
      int hit = lookup ? lookup[cell] : 0;
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
  m.def("orientationHitStatus", &orientationHitStatus);
  m.def("scoreOrientations", &scoreOrientations,
        py::arg("rmats"), py::arg("hkls"), py::arg("bmat"), py::arg("wavelength"),
        py::arg("beam"), py::arg("eta_edges"), py::arg("omega_edges"),
        py::arg("valid_eta"), py::arg("valid_omega"), py::arg("omega_offset"),
        py::arg("ring_maps"), py::arg("ring_for_hkl"), py::arg("dpix_eta"),
        py::arg("dpix_omega"), py::arg("threshold"), py::arg("hit_status") = py::none());

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
