// Bounded, exact support extraction from little-endian uint32 Eiger bitshuffle.
#pragma once
#ifndef _WIN32
namespace eiger_decoder {
namespace {

class Decoder {
  using Pool = spot_extraction::NativeChunkExecutor;
  using Lz4 = int (*)(const char *, char *, int, int);
  struct Block {
    Index source, target;
    int packed, raw;
  };
  struct Jobs {
    std::atomic<bool> cancelled{false};
    std::vector<std::future<int>> pending;
    ~Jobs() {
      cancelled.store(true, std::memory_order_relaxed);
      for (auto &job : pending)
        if (job.valid())
          job.wait();
    }
  };
  py::object library;
  Lz4 decompress;
  std::unique_ptr<Pool> pool;

  static uint64_t bigEndian(const uint8_t *p, int n) {
    uint64_t v = 0;
    for (int i = 0; i < n; ++i)
      v = (v << 8) | p[i];
    return v;
  }
  static void select(const uint8_t *source, Index bytes, uint32_t threshold,
                     uint32_t first,
                     std::vector<std::pair<uint32_t, uint32_t>> &keep) {
    const Index n = bytes / 4, stride = n / 8;
    auto value = [&](Index pixel) {
      uint32_t result = 0;
      for (unsigned bit = 0; bit < 32; ++bit)
        result |=
            uint32_t((source[bit * stride + pixel / 8] >> (pixel % 8)) & 1u)
            << bit;
      return result;
    };
    Index pixel = 0;
    for (; pixel + 64 <= n; pixel += 64) {
      uint64_t greater = 0;
      // A more significant bit overrides the comparison of all lower bits.
      for (int bit = 0; bit < 32; ++bit) {
        uint64_t plane;
        std::memcpy(&plane, source + bit * stride + pixel / 8, 8);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        plane = __builtin_bswap64(plane);
#endif
        if ((threshold >> bit) & 1u)
          greater &= plane;
        else
          greater |= plane;
      }
      while (greater) {
        unsigned lane = __builtin_ctzll(greater);
        uint32_t offset = uint32_t(pixel + lane);
        keep.emplace_back(first + offset, value(offset));
        greater &= greater - 1;
      }
    }
    for (; pixel < n; ++pixel) {
      auto v = value(pixel);
      if (v > threshold)
        keep.emplace_back(first + uint32_t(pixel), v);
    }
  }

public:
  Decoder(py::object lib, int workers) : library(lib) {
    if (workers < 1 || workers > 8)
      throw py::value_error("Eiger block workers must be between 1 and 8");
    auto ctypes = py::module_::import("ctypes");
    decompress =
        reinterpret_cast<Lz4>(ctypes
                                  .attr("cast")(lib.attr("LZ4_decompress_safe"),
                                                ctypes.attr("c_void_p"))
                                  .attr("value")
                                  .cast<uintptr_t>());
    pool = std::make_unique<Pool>(workers);
  }

  py::tuple support(CArray<uint8_t> input, double threshold,
                    py::object check) const {
    if (!check.is_none())
      check();
    checkShape(input, {-1});
    if (!std::isfinite(threshold) || threshold < 0. || threshold > UINT32_MAX)
      throw py::value_error("expected a finite uint32 support threshold");
    const auto *source = input.data();
    const Index length = input.size();
    if (length < 12)
      throw py::value_error("truncated bitshuffle header");
    uint64_t size = bigEndian(source, 8), bytes = bigEndian(source + 8, 4);
    if (!bytes)
      bytes = 8192;
    if (!size || size > 256 * 1024 * 1024 || size % 4 ||
        bytes > Pool::maxChunk || bytes % 32)
      throw py::value_error("invalid uint32 bitshuffle dimensions");
    const Index elements = Index(size / 4), tail = (elements % 8) * 4;
    Index packed = 12, raw = 0;
    std::vector<Block> blocks;
    while (raw < Index(size) - tail) {
      Index count = std::min(Index(bytes / 4), (Index(size) - raw) / 4);
      count -= count % 8;
      if (length - packed < 4)
        throw py::value_error("truncated bitshuffle block");
      uint64_t n = bigEndian(source + packed, 4);
      packed += 4;
      if (n > uint64_t(length - packed) || n > INT32_MAX)
        throw py::value_error("invalid compressed block length");
      blocks.push_back({packed, raw, int(n), int(count * 4)});
      packed += Index(n);
      raw += count * 4;
    }
    if (length - packed != tail)
      throw py::value_error("invalid bitshuffle tail");
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> selected(
        (blocks.size() + 255) / 256);
    for (auto &chunk : selected) chunk.reserve(4096);
    const auto cut = uint32_t(threshold);
    {
      py::gil_scoped_release release;
      Jobs jobs; // Drain submitted jobs before their input storage can be
                 // released.
      for (size_t first = 0; first < blocks.size(); first += 256) {
        const auto stop = std::min(first + 256, blocks.size());
        jobs.pending.push_back(pool->submit([&, first,
                                             stop](Pool::Buffer &buffer) {
          for (auto i = first; i < stop; ++i) {
            if (jobs.cancelled.load(std::memory_order_relaxed))
              return 0;
            const auto &b = blocks[i];
            auto temp = buffer.get();
            if (decompress(reinterpret_cast<const char *>(source + b.source),
                           reinterpret_cast<char *>(temp), b.packed,
                           b.raw) != b.raw)
              return -1;
            select(temp, b.raw, cut, uint32_t(b.target / 4),
                   selected[first / 256]);
          }
          return 0;
        }));
      }
      bool invalid = false;
      for (auto &job : jobs.pending) {
        while (job.wait_for(std::chrono::milliseconds(10)) !=
               std::future_status::ready) {
          py::gil_scoped_acquire acquire;
          if (PyErr_CheckSignals() < 0)
            throw py::error_already_set();
          if (!check.is_none())
            check();
        }
        invalid |= job.get() != 0;
      }
      if (invalid)
        throw std::runtime_error("invalid bitshuffle compressed block");
    }
    // The final fewer-than-eight elements are stored without bitshuffling.
    std::vector<std::pair<uint32_t, uint32_t>> remainder;
    for (Index i = 0; i < tail; i += 4) {
      uint32_t value;
      std::memcpy(&value, source + packed + i, 4);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      value = __builtin_bswap32(value);
#endif
      if (value > cut)
        remainder.emplace_back(uint32_t((raw + i) / 4), value);
    }
    selected.push_back(std::move(remainder));
    Index count = 0;
    for (const auto &chunk : selected)
      count += chunk.size();
    py::array_t<uint32_t> values(count), pixels(count);
    auto data = values.mutable_data(), ids = pixels.mutable_data();
    {
      py::gil_scoped_release release;
      Index cursor = 0;
      for (const auto &chunk : selected)
        for (const auto &item : chunk) {
          ids[cursor] = item.first;
          data[cursor++] = item.second;
        }
    }
    return py::make_tuple(values, pixels);
  }
};

static void bind(py::module_ &m) {
  py::class_<Decoder>(m, "_EigerSupport")
      .def(py::init<py::object, int>(), py::arg("library"),
           py::arg("workers") = 1)
      .def("support", &Decoder::support, py::arg("payload").noconvert(),
           py::arg("threshold"), py::arg("check") = py::none());
}
} // namespace
} // namespace eiger_decoder
#endif
