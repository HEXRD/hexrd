// Fit-local native residual evaluation. Preserve NumPy's BLAS calls for the
// three matrix products; scalar assembly retains the reference operation order.
#pragma once

namespace grain_objective {
namespace {

static py::array dot(py::array left, py::array right) {
  auto product = py::reinterpret_steal<py::object>(
      PyArray_MatrixProduct2(left.ptr(), right.ptr(), nullptr));
  if (!product)
    throw py::error_already_set();
  return py::cast<py::array>(product);
}

static py::array transpose(py::array array) {
  auto result = py::reinterpret_steal<py::object>(PyArray_Transpose(
      reinterpret_cast<PyArrayObject *>(array.ptr()), nullptr));
  if (!result)
    throw py::error_already_set();
  return py::cast<py::array>(result);
}

struct Panel {
  CArray<double> reciprocal, measured, rd, td, ts;
  std::vector<double> oscillation_reciprocal;
  double chi;
};

struct PanelGeometry {
  CArray<double> vectors, rs, omegas;
};
struct CachedGeometry {
  std::array<double, 10> key;
  CArray<double> rc;
  std::vector<PanelGeometry> panels;
};

class Objective {
  std::vector<Panel> panels;
  CArray<double> beam, eta;
  Index total = 0;
  mutable std::array<std::shared_ptr<CachedGeometry>, 4> cache;

public:
  Objective(py::list input, CArray<double> bmat, CArray<double> beam,
            CArray<double> eta)
      : beam(beam), eta(eta) {
    checkShape(bmat, {3, 3});
    checkShape(beam, {3});
    checkShape(eta, {3});
    for (auto entry : input) {
      auto item = py::cast<py::tuple>(entry);
      if (item.size() != 7)
        throw py::value_error("invalid grain panel fields");
      CArray<double> hkls = CArray<double>::ensure(item[0]);
      Panel panel{
          CArray<double>::ensure(item[1]), CArray<double>::ensure(item[2]),
          CArray<double>::ensure(item[3]), CArray<double>::ensure(item[4]),
          CArray<double>::ensure(item[6]), {},
          py::cast<double>(item[5])};
      if (!hkls || !panel.reciprocal || !panel.measured || !panel.rd ||
          !panel.td || !panel.ts)
        throw py::type_error("grain panel fields must be numeric arrays");
      checkShape(hkls, {-1, 3});
      checkShape(panel.reciprocal, {3, hkls.shape(0)});
      checkShape(panel.measured, {hkls.shape(0), 3});
      checkShape(panel.rd, {3, 3});
      checkShape(panel.td, {3});
      checkShape(panel.ts, {3});
      panel.oscillation_reciprocal = reciprocalVectors(hkls, bmat);
      if (hkls.shape(0) > PY_SSIZE_T_MAX / 3 - total)
        throw py::value_error("too many grain reflections");
      total += hkls.shape(0);
      panels.push_back(std::move(panel));
    }
  }

  py::object residual(CArray<double> parameters, double wavelength) const {
    checkShape(parameters, {12});
    const double *p = parameters.data();
    std::array<double, 10> key;
    std::copy(p, p + 3, key.begin());
    std::copy(p + 6, p + 12, key.begin() + 3);
    key[9] = wavelength;
    std::shared_ptr<CachedGeometry> geometry;
    for (Index i = 0; i < Index(cache.size()); ++i)
      if (cache[i] &&
          std::memcmp(cache[i]->key.data(), key.data(), sizeof(key)) == 0) {
        geometry = cache[i];
        for (Index j = i; j > 0; --j)
          cache[j] = cache[j - 1];
        cache[0] = geometry;
        break;
      }
    if (!geometry) {
      geometry = std::make_shared<CachedGeometry>();
      geometry->key = key;
      Eigen::Vector3d expmap(p[0], p[1], p[2]);
      py::array rotation = py::cast(makeRotMatOfExpMap(expmap));
      CArray<double> rc = CArray<double>::ensure(rotation);
      py::array rt = transpose(rotation);
      py::array_t<double> stretch({Index(3), Index(3)});
      auto v = stretch.mutable_data();
      v[0] = p[6];
      v[4] = p[7];
      v[8] = p[8];
      v[5] = v[7] = p[9] / std::sqrt(2.);
      v[2] = v[6] = p[10] / std::sqrt(2.);
      v[1] = v[3] = p[11] / std::sqrt(2.);
      for (const auto &panel : panels) {
        Index n = panel.measured.shape(0);
        auto sample = dot(stretch, dot(rotation, panel.reciprocal));
        CArray<double> crystal = CArray<double>::ensure(dot(rt, sample));
        py::array_t<double> vectors({n, Index(3)}), angles({n, Index(3)}),
            other({n, Index(3)});
        py::array_t<double> omegas(n);
        bool infeasible = false;
        {
          py::gil_scoped_release release;
          for (Index i = 0; i < n; ++i) {
            double x = crystal.data()[i], y = crystal.data()[n + i],
                   z = crystal.data()[2 * n + i];
            double norm = std::sqrt((x * x + y * y) + z * z);
            if (norm <= 10. * eps)
              norm = 1.;
            vectors.mutable_data()[3 * i] = x / norm;
            vectors.mutable_data()[3 * i + 1] = y / norm;
            vectors.mutable_data()[3 * i + 2] = z / norm;
          }
          OscillationGeometry geometry(panel.chi, wavelength, p + 6,
                                       beam.data(), eta.data());
          geometry.predict(n, panel.oscillation_reciprocal.data(), rc.data(),
                           angles.mutable_data(), other.mutable_data(), nullptr,
                           nullptr, nullptr, true);
          for (Index i = 0; i < n; ++i) {
            for (int j = 0; j < 3; ++j)
              infeasible |= std::isnan(angles.data()[3 * i + j]);
            double measured = panel.measured.data()[3 * i + 2];
            double a = angles.data()[3 * i + 2], b = other.data()[3 * i + 2];
            double da = std::abs(a - measured), db = std::abs(b - measured);
            da = std::min(da, 2. * pi - da);
            db = std::min(db, 2. * pi - db);
            omegas.mutable_data()[i] =
                (db < da || (std::isnan(da) && !std::isnan(db))) ? b : a;
          }
        }
        if (infeasible)
          return py::none();
        auto rotations = makeOscillRotMat(
            panel.chi, Eigen::Map<const Eigen::VectorXd>(omegas.data(), n));
        py::array_t<double> rs({n, Index(3), Index(3)});
        for (Index i = 0; i < n; ++i)
          for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
              rs.mutable_data()[9 * i + 3 * j + k] = rotations(3 * i + j, k);
        geometry->panels.push_back({vectors, rs, omegas});
      }
      geometry->rc = rc;
      for (Index i = cache.size() - 1; i > 0; --i)
        cache[i] = cache[i - 1];
      cache[0] = geometry;
    }
    py::array_t<double> tc(3), result(3 * total);
    std::copy(p + 3, p + 6, tc.mutable_data());
    Index first = 0, index = 0;
    for (const auto &panel : panels) {
      Index n = panel.measured.shape(0);
      const auto &stored = geometry->panels[index++];
      const auto &vectors = stored.vectors, &rs = stored.rs,
                 &omegas = stored.omegas;
      const auto &rc = geometry->rc;
      bool infeasible = false;
      auto xy = projectGvecs<false>(vectors, panel.rd, rs, rc, panel.td,
                                    panel.ts, tc, beam);
      {
        py::gil_scoped_release release;
        for (Index i = 0; i < n; ++i) {
          if (std::isnan(xy.data()[2 * i]) || std::isnan(xy.data()[2 * i + 1]))
            infeasible = true;
          result.mutable_data()[3 * (first + i)] =
              xy.data()[2 * i] - panel.measured.data()[3 * i];
          result.mutable_data()[3 * (first + i) + 1] =
              xy.data()[2 * i + 1] - panel.measured.data()[3 * i + 1];
          double diff =
              std::abs(panel.measured.data()[3 * i + 2] - omegas.data()[i]);
          result.mutable_data()[3 * (first + i) + 2] =
              std::min(diff, 2. * pi - diff);
        }
      }
      if (infeasible)
        return py::none();
      first += n;
    }
    return result;
  }
};

static void bind(py::module_ &m) {
  py::class_<Objective>(m, "_GrainObjective")
      .def(py::init<py::list, CArray<double>, CArray<double>, CArray<double>>())
      .def("residual", &Objective::residual);
}
} // namespace
} // namespace grain_objective
