#pragma once
namespace spot_extraction {
namespace {

struct CatalogueKeys {
  py::array array;
  bool wide;
  explicit CatalogueKeys(py::array array)
      : array(array),
        wide(sameDtype(array.dtype(), py::dtype::of<uint64_t>())) {}
  uint64_t at(int64_t i) const {
    return wide ? static_cast<const uint64_t *>(array.data())[i]
                : static_cast<const uint32_t *>(array.data())[i];
  }
};

static py::object mergePixelEntries(py::array oldKeys, py::array oldValues,
                                    py::array newKeys, py::array newValues) {
  const auto validKey = [](const py::array &array) {
    return array.ndim() == 1 && (array.flags() & py::array::c_style) &&
           (array.flags() & py::detail::npy_api::NPY_ARRAY_ALIGNED_) &&
           (sameDtype(array.dtype(), py::dtype::of<uint32_t>()) ||
            sameDtype(array.dtype(), py::dtype::of<uint64_t>()));
  };
  if (!validKey(oldKeys) || !validKey(newKeys) || oldValues.ndim() != 1 ||
      newValues.ndim() != 1 || oldKeys.size() != oldValues.size() ||
      newKeys.size() != newValues.size() ||
      !sameDtype(oldValues.dtype(), newValues.dtype()) ||
      !(oldValues.flags() & py::array::c_style) ||
      !(newValues.flags() & py::array::c_style) ||
      (oldValues.dtype().kind() != 'i' && oldValues.dtype().kind() != 'u' &&
       oldValues.dtype().kind() != 'f'))
    return py::none();
  CatalogueKeys old(oldKeys), fresh(newKeys);
  const int64_t n = oldKeys.size(), m = newKeys.size(),
                itemsize = oldValues.itemsize();
  int64_t count = n;
  {
    py::gil_scoped_release release;
    for (int64_t i = 1; i < n; ++i)
      if (old.at(i) <= old.at(i - 1))
        throw std::invalid_argument("catalogue keys must be unique and sorted");
    for (int64_t j = 1; j < m; ++j)
      if (fresh.at(j) <= fresh.at(j - 1))
        throw std::invalid_argument("catalogue keys must be unique and sorted");
    int64_t i = 0;
    for (int64_t j = 0; j < m; ++j) {
      while (i < n && old.at(i) < fresh.at(j))
        ++i;
      count += i == n || old.at(i) != fresh.at(j);
    }
  }
  const bool wide = old.wide || fresh.wide;
  py::array keys(wide ? py::dtype::of<uint64_t>() : py::dtype::of<uint32_t>(),
                 std::vector<py::ssize_t>{count});
  py::array values(oldValues.dtype(), std::vector<py::ssize_t>{count});
  auto output = static_cast<char *>(values.mutable_data());
  auto oldData = static_cast<const char *>(oldValues.data()),
       newData = static_cast<const char *>(newValues.data());
  {
    py::gil_scoped_release release;
    int64_t i = 0, j = 0, k = 0;
    while (i < n || j < m) {
      uint64_t key;
      const char *source;
      if (j == m || (i < n && old.at(i) < fresh.at(j))) {
        key = old.at(i);
        source = oldData + itemsize * i++;
      } else {
        key = fresh.at(j);
        source = newData + itemsize * j++;
        if (i < n && old.at(i) == key)
          ++i;
      }
      if (wide)
        static_cast<uint64_t *>(keys.mutable_data())[k] = key;
      else
        static_cast<uint32_t *>(keys.mutable_data())[k] = uint32_t(key);
      std::memcpy(output + itemsize * k++, source, size_t(itemsize));
    }
  }
  return py::make_tuple(keys, values);
}

} // namespace
} // namespace spot_extraction
