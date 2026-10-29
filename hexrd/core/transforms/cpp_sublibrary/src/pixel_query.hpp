#pragma once
namespace spot_extraction {
namespace {

// Python retains the byte-bounded frame catalogue. All per-pixel request work
// (packing, sorting, deduplication, lookup and scatter) is native and zero-copy
// at the boundary for compatible input buffers.
class PixelQuery {
  struct Entry {
    uint64_t key;
    int64_t index;
  };
  py::dtype dtype;
  py::array keys, missingKeys, uniqueValues, missingRows, missingCols;
  CArray<int64_t> inverse;
  std::vector<int64_t> missing;
  int64_t count, itemsize;
  bool wide;

  uint64_t key(int64_t index) const {
    return wide ? static_cast<const uint64_t *>(keys.data())[index]
                : static_cast<const uint32_t *>(keys.data())[index];
  }

public:
  int64_t uniqueCount = 0, missingCount = 0;
  PixelQuery(CArray<int64_t> rows, CArray<int64_t> cols, int64_t height,
             int64_t width, py::dtype dtype, py::object old, uint64_t maximum)
      : dtype(dtype), inverse(rows.size()), count(rows.size()),
        itemsize(dtype.itemsize()), wide(maximum > UINT32_MAX) {
    std::vector<Entry> entries(static_cast<size_t>(count));
    {
      py::gil_scoped_release release;
      for (int64_t i = 0; i < count; ++i) {
        if (rows.data()[i] < 0 || rows.data()[i] >= height ||
            cols.data()[i] < 0 || cols.data()[i] >= width)
          throw std::invalid_argument("pixel coordinates outside source");
        entries[i] = {uint64_t(rows.data()[i]) * uint64_t(width) +
                          uint64_t(cols.data()[i]),
                      i};
      }
      std::sort(entries.begin(), entries.end(),
                [](const Entry &a, const Entry &b) { return a.key < b.key; });
      for (int64_t i = 0; i < count; ++i)
        uniqueCount += !i || entries[i].key != entries[i - 1].key;
    }
    auto keydtype =
        wide ? py::dtype::of<uint64_t>() : py::dtype::of<uint32_t>();
    keys = py::array(keydtype, std::vector<py::ssize_t>{uniqueCount});
    uniqueValues = py::array(dtype, std::vector<py::ssize_t>{uniqueCount});
    py::array storedKeys, storedValues;
    int64_t stored = 0;
    bool storedWide = false;
    if (!old.is_none()) {
      auto item = py::cast<py::tuple>(old);
      storedKeys = py::cast<py::array>(item[0]);
      storedValues = py::cast<py::array>(item[1]);
      stored = storedKeys.size();
      storedWide = sameDtype(storedKeys.dtype(), py::dtype::of<uint64_t>());
      if ((!storedWide &&
           !sameDtype(storedKeys.dtype(), py::dtype::of<uint32_t>())) ||
          !sameDtype(storedValues.dtype(), dtype) || storedKeys.ndim() != 1 ||
          storedValues.ndim() != 1 || storedValues.size() != stored ||
          !(storedKeys.flags() & py::array::c_style) ||
          !(storedValues.flags() & py::array::c_style) ||
          !(storedKeys.flags() & py::detail::npy_api::NPY_ARRAY_ALIGNED_))
        throw py::value_error("invalid source-pixel catalogue");
    }
    missing.reserve(size_t(uniqueCount));
    {
      py::gil_scoped_release release;
      auto storedKey = [&](int64_t i) {
        return storedWide ? static_cast<const uint64_t *>(storedKeys.data())[i]
                          : static_cast<const uint32_t *>(storedKeys.data())[i];
      };
      int64_t index = -1;
      for (int64_t i = 0; i < count; ++i) {
        if (!i || entries[i].key != entries[i - 1].key) {
          ++index;
          if (wide)
            static_cast<uint64_t *>(keys.mutable_data())[index] =
                entries[i].key;
          else
            static_cast<uint32_t *>(keys.mutable_data())[index] =
                uint32_t(entries[i].key);
          int64_t lo = 0, hi = stored;
          while (lo < hi) {
            const int64_t mid = lo + (hi - lo) / 2;
            if (storedKey(mid) < entries[i].key)
              lo = mid + 1;
            else
              hi = mid;
          }
          if (lo < stored && storedKey(lo) == entries[i].key)
            std::memcpy(static_cast<char *>(uniqueValues.mutable_data()) +
                            index * itemsize,
                        static_cast<const char *>(storedValues.data()) +
                            lo * itemsize,
                        size_t(itemsize));
          else
            missing.push_back(index);
        }
        inverse.mutable_data()[entries[i].index] = index;
      }
    }
    missingCount = int64_t(missing.size());
    missingKeys = py::array(keydtype, std::vector<py::ssize_t>{missingCount});
    missingRows = py::array_t<int64_t>(missingCount);
    missingCols = py::array_t<int64_t>(missingCount);
    {
      py::gil_scoped_release release;
      for (int64_t i = 0; i < missingCount; ++i) {
        const uint64_t value = key(missing[i]);
        if (wide)
          static_cast<uint64_t *>(missingKeys.mutable_data())[i] = value;
        else
          static_cast<uint32_t *>(missingKeys.mutable_data())[i] =
              uint32_t(value);
        static_cast<int64_t *>(missingRows.mutable_data())[i] =
            int64_t(value / uint64_t(width));
        static_cast<int64_t *>(missingCols.mutable_data())[i] =
            int64_t(value % uint64_t(width));
      }
    }
  }
  py::array finish(py::object input) {
    py::array values;
    if (missingCount) {
      values = py::cast<py::array>(input);
      checkShape(values, {missingCount});
      if (!sameDtype(values.dtype(), dtype) ||
          !(values.flags() & py::array::c_style))
        throw py::value_error("missing pixels do not match source dtype");
    }
    py::array output(dtype, std::vector<py::ssize_t>{count});
    {
      py::gil_scoped_release release;
      for (int64_t i = 0; i < missingCount; ++i)
        std::memcpy(static_cast<char *>(uniqueValues.mutable_data()) +
                        missing[i] * itemsize,
                    static_cast<const char *>(values.data()) + i * itemsize,
                    size_t(itemsize));
      for (int64_t i = 0; i < count; ++i)
        std::memcpy(static_cast<char *>(output.mutable_data()) + i * itemsize,
                    static_cast<const char *>(uniqueValues.data()) +
                        inverse.data()[i] * itemsize,
                    size_t(itemsize));
    }
    return output;
  }
  py::array missingCoordinates(bool rows) const {
    return rows ? missingRows : missingCols;
  }
  py::array missingKeysArray() const { return missingKeys; }
};

static std::unique_ptr<PixelQuery>
makePixelQuery(py::array rows, py::array cols, int64_t height, int64_t width,
               py::dtype dtype, py::object old) {
  const auto numericInteger = [](py::array a) {
    return a.dtype().kind() == 'i' || a.dtype().kind() == 'u';
  };
  if (height <= 0 || width <= 0 || rows.size() > 65536 || rows.ndim() != 1 ||
      cols.ndim() != 1 || rows.size() != cols.size() || !numericInteger(rows) ||
      !numericInteger(cols) ||
      (dtype.kind() != 'i' && dtype.kind() != 'u' && dtype.kind() != 'f'))
    return nullptr;
  if (uint64_t(height - 1) > UINT64_MAX / uint64_t(width))
    return nullptr;
  const uint64_t base = uint64_t(height - 1) * uint64_t(width);
  if (uint64_t(width - 1) > UINT64_MAX - base)
    return nullptr;
  try {
    return std::unique_ptr<PixelQuery>(new PixelQuery(
        py::cast<CArray<int64_t>>(rows), py::cast<CArray<int64_t>>(cols),
        height, width, dtype, old, base + uint64_t(width - 1)));
  } catch (const std::invalid_argument &) {
    return nullptr;
  }
}

static void bindPixelQuery(py::module_ &m) {
  py::class_<PixelQuery>(m, "_PixelQuery")
      .def_readonly("unique_count", &PixelQuery::uniqueCount)
      .def_readonly("missing_count", &PixelQuery::missingCount)
      .def_property_readonly(
          "rows",
          [](const PixelQuery &q) { return q.missingCoordinates(true); })
      .def_property_readonly(
          "cols",
          [](const PixelQuery &q) { return q.missingCoordinates(false); })
      .def_property_readonly("missing_keys", &PixelQuery::missingKeysArray)
      .def("finish", &PixelQuery::finish, py::arg("values") = py::none());
  m.def("_makePixelQuery", &makePixelQuery);
}

} // namespace
} // namespace spot_extraction
