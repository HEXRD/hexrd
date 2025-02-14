#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
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

PYBIND11_MODULE(transforms, m) {
  m.doc() = "HEXRD transforms plugin";

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
