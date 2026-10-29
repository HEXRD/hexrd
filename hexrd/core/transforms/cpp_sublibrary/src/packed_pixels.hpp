// Lossless, RAM-only catalogue entries. Encoding never changes numeric values.
#pragma once
namespace spot_extraction {
namespace {

class PackedPixels {
  py::object library;
  BloscReader::Validate validate;
  BloscReader::Decompress decompress;
  py::dtype keytype, valuetype;
  int64_t count, keybytes, valuebytes;
  std::vector<unsigned char> packedKeys, packedValues;

public:
  PackedPixels(const BloscReader &codec, py::array keys, py::array values,
               const std::string &algorithm)
      : library(codec.library), validate(codec.validate),
        decompress(codec.decompress), keytype(keys.dtype()),
        valuetype(values.dtype()), count(keys.size()), keybytes(keys.nbytes()),
        valuebytes(values.nbytes()) {
    const auto keydata = keys.data(), valuedata = values.data();
    const auto keysize = keys.itemsize(), valuesize = values.itemsize();
    py::gil_scoped_release release;
    auto encode = [&](const void *source, int64_t bytes, int64_t itemsize) {
      std::vector<unsigned char> workspace(size_t(bytes + 16));
      const int written = codec.compress(
          1, 1, size_t(itemsize), size_t(bytes), source, workspace.data(),
          workspace.size(), algorithm.c_str(), 128 * 1024, 1);
      if (written <= 0)
        return std::vector<unsigned char>();
      // Shrinking size alone would retain the uncompressed allocation.
      return std::vector<unsigned char>(workspace.begin(),
                                        workspace.begin() + written);
    };
    packedKeys = encode(keydata, keybytes, keysize);
    packedValues = encode(valuedata, valuebytes, valuesize);
  }
  bool useful() const {
    return !packedKeys.empty() && !packedValues.empty() &&
           packedKeys.capacity() + packedValues.capacity() <
               size_t(keybytes + valuebytes);
  }
  int64_t storageBytes() const {
    return 1024 + int64_t(packedKeys.capacity()) +
           int64_t(packedValues.capacity());
  }
  int64_t decodedBytes() const { return keybytes + valuebytes; }
  py::tuple restore() const {
    py::array keys(keytype, std::vector<py::ssize_t>{count});
    py::array values(valuetype, std::vector<py::ssize_t>{count});
    auto keydata = keys.mutable_data(), valuedata = values.mutable_data();
    {
      py::gil_scoped_release release;
      auto decode = [&](const std::vector<unsigned char> &source, void *target,
                        int64_t bytes) {
        size_t expected = 0;
        if (validate(source.data(), source.size(), &expected) ||
            expected != size_t(bytes) ||
            decompress(source.data(), target, size_t(bytes), 1) != bytes)
          throw std::runtime_error("invalid packed source-pixel catalogue");
      };
      decode(packedKeys, keydata, keybytes);
      decode(packedValues, valuedata, valuebytes);
    }
    PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject *>(keys.ptr()),
                       NPY_ARRAY_WRITEABLE);
    PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject *>(values.ptr()),
                       NPY_ARRAY_WRITEABLE);
    return py::make_tuple(keys, values);
  }
};

py::object BloscReader::packPixels(py::array keys, py::array values,
                                   const std::string &algorithm) const {
  if (algorithm != "zstd" && algorithm != "lz4")
    throw py::value_error("unsupported catalogue compressor");
  checkShape(keys, {-1});
  checkShape(values, {keys.size()});
  const char kind = values.dtype().kind();
  if (!compress || !keys.size() ||
      keys.nbytes() + values.nbytes() > 1024 * 1024 ||
      !(keys.flags() & py::array::c_style) ||
      !(values.flags() & py::array::c_style) ||
      !(sameDtype(keys.dtype(), py::dtype::of<uint32_t>()) ||
        sameDtype(keys.dtype(), py::dtype::of<uint64_t>())) ||
      (kind != 'i' && kind != 'u' && kind != 'f'))
    return py::none();
  auto frame = std::make_shared<PackedPixels>(*this, keys, values, algorithm);
  return frame->useful() ? py::cast(frame) : py::object(py::none());
}

static void bindPackedPixels(py::module_ &m) {
  py::class_<PackedPixels, std::shared_ptr<PackedPixels>>(m, "_PackedPixels")
      .def_property_readonly("storage_bytes", &PackedPixels::storageBytes)
      .def_property_readonly("decoded_bytes", &PackedPixels::decodedBytes)
      .def("restore", &PackedPixels::restore);
}

} // namespace
} // namespace spot_extraction
