// Immutable histogram geometry; preserve the reference block/prefix arithmetic.
#pragma once

class HistogramPlan {
  struct Block { std::vector<Index> pixels, cuts; };
  std::vector<std::vector<Block>> rings;
  Index rows, cols, bins, largest = 0;
  py::object sparse = py::none();
  bool sparse_unique = true;

  template<class T> py::array_t<double> evaluate(const py::array &frame,
                                                double threshold, double scale) const {
    const char *raw = static_cast<const char *>(frame.data());
    py::array_t<double> output({Index(rings.size()), bins});
    auto out = output.mutable_data();
    py::gil_scoped_release release;
    std::vector<double> cumulative(bins + 1);
    for (Index ring = 0; ring < Index(rings.size()); ++ring) {
      std::fill(cumulative.begin(), cumulative.end(), 0.);
      for (const auto &block : rings[ring]) {
        double sum = 0.;
        Index next = 0;
        for (Index edge = 0; edge <= bins; ++edge) {
          for (; next < block.cuts[edge]; ++next) {
            T item;
            // Stream buffers can be unaligned. memcpy also permits those views.
            std::memcpy(&item, raw + block.pixels[next] * sizeof(T), sizeof(T));
            double value = double(item);
            if (scale != 1.) value *= scale;
            if (value < threshold) value = 0.;
            if (value != 0.) sum += value;
          }
          cumulative[edge] += sum;
        }
      }
      for (Index bin = 0; bin < bins; ++bin)
        out[ring*bins + bin] = cumulative[bin + 1] - cumulative[bin];
    }
    return output;
  }

public:
  HistogramPlan(py::list input, std::pair<Index, Index> shape, Index bins)
      : rows(shape.first), cols(shape.second), bins(bins) {
    if (rows < 1 || cols < 1 || rows > PY_SSIZE_T_MAX / cols || bins < 1)
      throw py::value_error("invalid histogram dimensions");
    for (auto ring : input) {
      std::vector<Block> blocks;
      Index count = 0;
      for (auto item : py::cast<py::list>(ring)) {
        auto pair = py::cast<py::tuple>(item);
        if (pair.size() != 2) throw py::value_error("invalid histogram block");
        CArray<Index> pixels(pair[0]), cuts(pair[1]);
        checkShape(pixels, {-1}); checkShape(cuts, {bins + 1});
        for (Index i = 0; i < pixels.size(); ++i)
          if (pixels.data()[i] < 0 || pixels.data()[i] >= rows*cols)
            throw py::value_error("histogram pixel outside image");
        Index last = 0;
        for (Index i = 0; i < cuts.size(); ++i) {
          Index stop = cuts.data()[i];
          if (stop < last || stop > pixels.size())
            throw py::value_error("invalid histogram cuts");
          last = stop;
        }
        blocks.push_back({{pixels.data(), pixels.data()+pixels.size()},
                          {cuts.data(), cuts.data()+cuts.size()}});
        if (pixels.size() > PY_SSIZE_T_MAX - count)
          throw py::value_error("histogram plan too large");
        count += pixels.size();
      }
      largest = std::max(largest, count);
      rings.push_back(std::move(blocks));
    }
  }

  Index maxPixels() const { return largest; }

  py::array_t<double> accumulate(py::array frame, double threshold, double scale) const {
    if (!(scale > 0.) || !std::isfinite(scale))
      throw py::value_error("histogram scale must be positive and finite");
    checkShape(frame, {rows, cols});
    frame = py::array::ensure(frame, py::array::c_style);
    if (!frame) throw py::type_error("expected a numeric image");
#define HISTOGRAM_TYPE(T)                                                      \
    if (frame.dtype().equal(py::dtype::of<T>())) return evaluate<T>(frame, threshold, scale)
    HISTOGRAM_TYPE(bool); HISTOGRAM_TYPE(uint8_t); HISTOGRAM_TYPE(int8_t);
    HISTOGRAM_TYPE(uint16_t); HISTOGRAM_TYPE(int16_t);
    HISTOGRAM_TYPE(uint32_t); HISTOGRAM_TYPE(int32_t);
    HISTOGRAM_TYPE(uint64_t); HISTOGRAM_TYPE(int64_t);
    HISTOGRAM_TYPE(float); HISTOGRAM_TYPE(double);
#undef HISTOGRAM_TYPE
    // Non-native byte order and uncommon numeric types retain NumPy conversion.
    CArray<double> converted(frame);
    return evaluate<double>(converted, threshold, scale);
  }

  py::object sparseBins() {
    if (!sparse.is_none()) return sparse;
    py::array_t<int64_t> masks({(Index(rings.size())+62)/63, rows*cols});
    py::array_t<int32_t> eta(rows*cols);
    std::fill(masks.mutable_data(), masks.mutable_data()+masks.size(), 0);
    std::fill(eta.mutable_data(), eta.mutable_data()+eta.size(), -1);
    for (Index ring = 0; ring < Index(rings.size()); ++ring)
      for (const auto &block : rings[ring]) {
        if (block.cuts[0] != 0) sparse_unique = false;
        for (Index bin = 0; bin < bins; ++bin)
          for (Index p = block.cuts[bin]; p < block.cuts[bin+1]; ++p) {
            Index pixel = block.pixels[p];
            if (eta.data()[pixel] >= 0 && eta.data()[pixel] != bin)
              throw py::value_error("inconsistent eta bins in histogram plan");
            eta.mutable_data()[pixel] = bin;
            auto &mask = masks.mutable_data()[(ring/63)*rows*cols+pixel];
            int64_t bit = int64_t(1) << (ring%63);
            if (mask & bit) sparse_unique = false;
            mask |= bit;
          }
      }
    masks.attr("setflags")(false); eta.attr("setflags")(false);
    sparse = py::make_tuple(eta, masks);
    return sparse;
  }

  py::object accumulateSparse(CArray<uint32_t> pixels, CArray<uint32_t> values,
                              double threshold, double scale) {
    checkShape(pixels, {-1}); checkShape(values, {pixels.size()});
    if (!std::isfinite(threshold) || (scale != 1. && scale != .5))
      return py::none();
    py::tuple geometry;
    try { geometry = py::cast<py::tuple>(sparseBins()); }
    catch (const py::value_error &) { return py::none(); }
    if (!sparse_unique) return py::none();
    CArray<int32_t> eta(geometry[0]); CArray<int64_t> masks(geometry[1]);
    std::vector<uint64_t> sums(rings.size()*bins, 0), mass(rings.size(), 0);
    const uint64_t limit = uint64_t(1) << 53;
    bool exact = true;
    {
      py::gil_scoped_release release;
      for (Index i = 0; i < pixels.size() && exact; ++i) {
        Index pixel = pixels.data()[i];
        if (pixel >= rows*cols || (i && pixels.data()[i-1] >= pixels.data()[i]))
          throw std::out_of_range("expected unique sorted image pixel IDs");
        uint64_t value = values.data()[i];
        if (double(value)*scale < threshold) continue;
        for (Index word = 0; word < masks.shape(0) && exact; ++word) {
          uint64_t mask = uint64_t(masks.data()[word*rows*cols+pixel]);
          for (Index ring = word*63; mask; ++ring, mask >>= 1) if (mask & 1) {
            if (mass[ring] > limit-value) { exact = false; break; }
            mass[ring] += value;
            sums[ring*bins+eta.data()[pixel]] += value;
          }
        }
      }
    }
    // Every reference prefix and subtraction is exact in this nonnegative range.
    // Other numeric cases retain the dense reference path.
    if (!exact) return py::none();
    py::array_t<double> result({Index(rings.size()), bins});
    for (Index i = 0; i < Index(sums.size()); ++i)
      result.mutable_data()[i] = double(sums[i])*scale;
    return result;
  }
};
