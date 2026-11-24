// Compact decoded images without full-image coordinate or conversion arrays.
#pragma once

template<class T, bool filtered> static py::object denseCsrTyped(
    const py::array &input, double threshold) {
  if (reinterpret_cast<uintptr_t>(input.data()) % alignof(T)) return py::none();
  const T *raw = static_cast<const T *>(input.data());
  const Index rows = input.shape(0), cols = input.shape(1);
  py::array_t<int32_t> offsets(rows + 1);
  auto starts = offsets.mutable_data();
  T cut = threshold < 0. ? T(0) : threshold >= double(std::numeric_limits<T>::max())
      ? std::numeric_limits<T>::max() : T(threshold);
  if (double(cut) > threshold && threshold >= 0.)
    cut = std::nextafter(cut, -std::numeric_limits<T>::infinity());
  auto keep = [threshold](T value) {
    return !filtered ? value != T(0) : !(value >= T(0) && double(value) <= threshold);
  };
  auto count = [&]() {
    py::gil_scoped_release release;
    return std::count_if(raw, raw + input.size(), keep);
  };
  auto allocate = [](Index n) {
    return std::make_pair(py::array_t<int32_t>(n), py::array_t<T>(n));
  };
  auto buffers = [&]() {
    if (filtered) return allocate(std::min(input.size(), Index(65536)));
    try {
      return allocate(input.size());
    } catch (py::error_already_set &error) {
      if (!error.matches(PyExc_MemoryError)) throw;
      return allocate(count());
    }
  }();
  auto &indices = buffers.first;
  auto &data = buffers.second;
  auto columns = indices.mutable_data();
  auto values = data.mutable_data();
  Index cursor = 0;
  {
    py::gil_scoped_release release;
    using Batch = xsimd::batch<T>;
    constexpr Index lanes = Batch::size;
    starts[0] = 0;
    for (Index row = 0; row < rows; ++row) {
      Index col = 0;
      auto append = [&](Index c) {
        T value = raw[row * cols + c];
        if (keep(value)) {
          if (filtered && cursor == indices.size()) {
            py::gil_scoped_acquire acquire;
            Index capacity = indices.size() + std::min(indices.size(), input.size()-indices.size());
            indices.resize({capacity}, false); data.resize({capacity}, false);
            columns = indices.mutable_data(); values = data.mutable_data();
          }
          columns[cursor] = int32_t(c);
          values[cursor++] = value;
        }
      };
      for (; col + lanes <= cols; col += lanes) {
        auto packed = Batch::load_unaligned(raw + row * cols + col);
        if (xsimd::any(!filtered ? packed != Batch(T(0)) :
                       !((packed >= Batch(T(0))) & (packed <= Batch(cut)))))
          for (Index i = 0; i < lanes; ++i) append(col + i);
      }
      for (; col < cols; ++col) append(col);
      starts[row + 1] = int32_t(cursor);
    }
  }
  indices.resize({cursor}, false);
  data.resize({cursor}, false);
  return py::make_tuple(data, indices, offsets);
}

static py::object denseCsr(py::array input, double threshold) {
  if (!std::isfinite(threshold)) throw py::value_error("CSR threshold must be finite");
  if (input.ndim() != 2 || !(input.flags() & py::array::c_style) ||
      input.size() > INT32_MAX) return py::none();
  if (threshold >= 0. && input.itemsize() == 8 &&
      (input.dtype().kind() == 'i' || input.dtype().kind() == 'u')) return py::none();
#define DENSE_CSR_TYPE(T)                                                       \
  if (input.dtype().equal(py::dtype::of<T>())) return threshold < 0. \
      ? denseCsrTyped<T, false>(input, threshold) : denseCsrTyped<T, true>(input, threshold)
  DENSE_CSR_TYPE(uint8_t); DENSE_CSR_TYPE(int8_t);
  DENSE_CSR_TYPE(uint16_t); DENSE_CSR_TYPE(int16_t);
  DENSE_CSR_TYPE(uint32_t); DENSE_CSR_TYPE(int32_t);
  DENSE_CSR_TYPE(uint64_t); DENSE_CSR_TYPE(int64_t);
  DENSE_CSR_TYPE(float); DENSE_CSR_TYPE(double);
#undef DENSE_CSR_TYPE
  return py::none(); // SciPy retains uncommon dtypes and array-layout semantics.
}
