// Native execution of a bounded, already planned set of exact spot windows.
#pragma once
namespace spot_extraction {
namespace {

struct WindowCoordinates {
  py::array owner;
  int itemsize;
  explicit WindowCoordinates(py::array values)
      : owner(values), itemsize(values.itemsize()) {
    checkShape(owner, {-1});
    if (!(owner.flags() & py::array::c_style) ||
        !(owner.flags() & py::detail::npy_api::NPY_ARRAY_ALIGNED_) ||
        !(sameDtype(owner.dtype(), py::dtype::of<int16_t>()) ||
          sameDtype(owner.dtype(), py::dtype::of<int32_t>()) ||
          sameDtype(owner.dtype(), py::dtype::of<int64_t>())))
      throw py::value_error("expected compact signed window coordinates");
  }
  int64_t at(int64_t index) const {
    if (itemsize == 2)
      return static_cast<const int16_t *>(owner.data())[index];
    if (itemsize == 4)
      return static_cast<const int32_t *>(owner.data())[index];
    return static_cast<const int64_t *>(owner.data())[index];
  }
};

struct MeasuredWindow {
  WindowCoordinates rows, cols;
  CArray<int64_t> shape;
  py::tuple destinations;
  py::array pixels;
  double threshold;
  bool dense, allocated = false;
  int64_t depth, remaining;
  std::vector<bool> seen;
  explicit MeasuredWindow(py::tuple request)
      : rows(py::cast<py::array>(request[0])),
        cols(py::cast<py::array>(request[1])),
        shape(py::cast<CArray<int64_t>>(request[2])),
        destinations(py::cast<py::tuple>(request[4])),
        threshold(py::cast<double>(request[3])),
        dense(py::cast<bool>(request[5])) {
    checkShape(shape, {1, 3});
    depth = remaining = shape.data()[0];
    if (rows.owner.size() != cols.owner.size() || depth <= 0 ||
        shape.data()[1] <= 0 || shape.data()[2] <= 0 ||
        shape.data()[1] > INT64_MAX / shape.data()[2] ||
        shape.data()[1] * shape.data()[2] != rows.owner.size() ||
        rows.owner.size() > INT64_MAX / depth || destinations.size() != 4)
      throw py::value_error("invalid planned window shape");
    seen.resize(size_t(depth));
  }
};

template <typename T>
static void normalizeWindowZeros(void *data, int64_t count) {
  auto values = static_cast<T *>(data);
  for (int64_t i = 0; i < count; ++i)
    if (values[i] == T(0))
      values[i] = T(0);
}

static py::object measureWindows(py::list requests, CArray<int64_t> events,
                                 py::dtype dtype, py::function sample,
                                 py::function cancel, py::function tick) {
  checkShape(events, {-1, 3});
  // Only raw numeric buffers supported by spotStats may be copied this way.
  // In particular, object arrays require reference-counted assignment.
  if (!(sameDtype(dtype, py::dtype::of<uint8_t>()) ||
        sameDtype(dtype, py::dtype::of<int16_t>()) ||
        sameDtype(dtype, py::dtype::of<uint16_t>()) ||
        sameDtype(dtype, py::dtype::of<int32_t>()) ||
        sameDtype(dtype, py::dtype::of<uint32_t>()) ||
        sameDtype(dtype, py::dtype::of<float>()) ||
        sameDtype(dtype, py::dtype::of<double>())))
    return py::none();
  const bool float32 = dtype.kind() == 'f' && dtype.itemsize() == 4;
  const bool float64 = dtype.kind() == 'f' && dtype.itemsize() == 8;
  std::vector<MeasuredWindow> windows;
  windows.reserve(requests.size());
  for (auto request : requests)
    windows.emplace_back(py::cast<py::tuple>(request));
  const auto data = events.data();
  const int64_t count = events.shape(0), itemsize = dtype.itemsize();
  // Validate all events before reading pixels or changing statistics.
  for (int64_t i = 0; i < count; ++i) {
    const int64_t patch = data[3 * i + 1], omega = data[3 * i + 2];
    if (patch < 0 || patch >= int64_t(windows.size()) || omega < 0 ||
        omega >= windows[patch].depth || windows[patch].seen[omega])
      throw py::value_error("invalid window frame event");
    windows[patch].seen[omega] = true;
  }
  for (auto &window : windows)
    if (std::find(window.seen.begin(), window.seen.end(), false) !=
        window.seen.end())
      throw py::value_error("incomplete window frame events");
  int64_t visits = 0;
  for (int64_t first = 0; first < count;) {
    cancel();
    int64_t last = first + 1;
    while (last < count && data[3 * last] == data[3 * first])
      ++last;
    struct Slice {
      int64_t event, offset, size;
    };
    int64_t event = first, offset = 0, completed = 0;
    while (event < last) {
      cancel();
      std::vector<Slice> slices;
      int64_t size = 0;
      while (event < last && size < 65536) {
        const auto &window = windows[data[3 * event + 1]];
        const int64_t n =
            std::min(window.rows.owner.size() - offset, int64_t(65536) - size);
        slices.push_back({event, offset, n});
        size += n;
        offset += n;
        if (offset == window.rows.owner.size()) {
          ++event;
          offset = 0;
        }
      }
      py::array_t<int64_t> rows(size), cols(size);
      auto row_data = rows.mutable_data(), col_data = cols.mutable_data();
      {
        py::gil_scoped_release release;
        int64_t cursor = 0;
        for (const auto &slice : slices) {
          const auto &window = windows[data[3 * slice.event + 1]];
          for (int64_t p = slice.offset; p < slice.offset + slice.size;
               ++p, ++cursor) {
            row_data[cursor] = window.rows.at(p);
            col_data[cursor] = window.cols.at(p);
          }
        }
      }
      py::array values =
          py::cast<py::array>(sample(data[3 * first], rows, cols));
      checkShape(values, {size});
      if (!sameDtype(values.dtype(), dtype) ||
          !(values.flags() & py::array::c_style))
        throw py::value_error(
            "sampled pixels do not match planned image dtype");
      const auto input = static_cast<const char *>(values.data());
      int64_t cursor = 0;
      for (const auto &slice : slices) {
        auto &window = windows[data[3 * slice.event + 1]];
        const int64_t n = window.rows.owner.size(),
                      omega = data[3 * slice.event + 2];
        if (!window.allocated) {
          window.pixels =
              py::array(dtype, std::vector<py::ssize_t>{n * window.depth});
          window.allocated = true;
        }
        auto output = static_cast<char *>(window.pixels.mutable_data()) +
                      (omega * n + slice.offset) * itemsize;
        {
          py::gil_scoped_release release;
          std::memcpy(output, input + cursor * itemsize,
                      size_t(slice.size * itemsize));
          if (!window.dense) {
            if (float64)
              normalizeWindowZeros<double>(output, slice.size);
            if (float32)
              normalizeWindowZeros<float>(output, slice.size);
          }
        }
        cursor += slice.size;
        if (slice.offset + slice.size == n && !--window.remaining) {
          cancel();
          CArray<int64_t> offsets(Index(2));
          offsets.mutable_data()[0] = 0;
          offsets.mutable_data()[1] = window.pixels.size();
          auto stats =
              spotStats(window.pixels, offsets, window.shape, window.threshold);
          for (int j = 0; j < 4; ++j) {
            auto source = py::cast<py::array>(stats[j]);
            auto destination = py::cast<py::array>(window.destinations[j]);
            if (PyArray_CopyInto(
                    reinterpret_cast<PyArrayObject *>(destination.ptr()),
                    reinterpret_cast<PyArrayObject *>(source.ptr())))
              throw py::error_already_set();
          }
          window.pixels = py::array();
          ++completed;
        }
      }
    }
    tick(completed);
    ++visits;
    first = last;
  }
  return py::int_(visits);
}

} // namespace
} // namespace spot_extraction
