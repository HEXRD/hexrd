#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <cmath>

namespace py = pybind11;

using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

Eigen::MatrixX3d anglesToGVec(const MatrixXdR &angs,
                              const Eigen::Vector3d &bHat,
                              const Eigen::Vector3d &eHat, double chi,
                              const Eigen::Matrix3d &rMat_c) noexcept {
  const size_t vec_size = angs.rows();
  Eigen::MatrixX3d gVec_c(vec_size, 3);

  const Eigen::Vector3d yHat = eHat.cross(bHat);
  const Eigen::Vector3d xHat = bHat.cross(yHat);
  Eigen::Matrix3d rotMat = (Eigen::Matrix3d() << xHat, yHat, -bHat).finished();

  double cchi, schi;
  sincos(chi, &schi, &cchi);
  Eigen::Matrix3d preComputedMatrix = rMat_c * Eigen::DiagonalMatrix<double, 3>(1.0, cchi, cchi) * rotMat.transpose();

  for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(vec_size); ++i) {
    double cosAlpha, sinAlpha, cosBeta, sinBeta, cosGamma, sinGamma;
    sincos(angs(i, 0) / 2.0, &sinAlpha, &cosAlpha);
    sincos(angs(i, 1), &sinBeta, &cosBeta);
    sincos(angs(i, 2), &sinGamma, &cosGamma);

    Eigen::Vector3d preMult_gVec(cosAlpha * cosBeta, cosAlpha * sinBeta, sinAlpha);

    Eigen::Vector3d postRot(
      cosGamma * preMult_gVec.x() + sinGamma * preMult_gVec.y(),
      schi * (sinGamma * preMult_gVec.x() + preMult_gVec.z()) + cchi * preMult_gVec.y(),
      cchi * (preMult_gVec.z() - sinGamma * preMult_gVec.x()) - schi * preMult_gVec.y()
    );

    gVec_c.row(i).noalias() = preComputedMatrix * postRot;
  }

  return gVec_c;
}

PYBIND11_MODULE(transforms, m) {
  m.doc() = "HEXRD transforms plugin";
  m.def("anglesToGVec", &anglesToGVec, "Convert angles to g-vectors with optimized memory handling and computational efficiency using sincos.");
}
