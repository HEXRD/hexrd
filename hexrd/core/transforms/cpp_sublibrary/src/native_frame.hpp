// Included after spot_extraction.hpp: this is the implementation of its private
// frame reader, not a separate extension or a new image representation.
#pragma once
#include <map>

namespace spot_extraction {
namespace {

struct FrameDirectory {
  py::array addresses, sizes, masks;
  int64_t chunk, itemsize, count, stride;
  const int64_t *positions;
  const uint32_t *lengths, *filters;

  explicit FrameDirectory(py::tuple spec) {
    if (spec.size() != 5)
      throw py::value_error("invalid native chunk directory");
    auto shape = py::cast<py::tuple>(spec[0]);
    if (shape.size() != 2 || py::cast<int64_t>(shape[0]) <= 0 ||
        py::cast<int64_t>(shape[1]) != 1)
      throw py::value_error("invalid native chunk shape");
    auto dtype = py::cast<py::dtype>(spec[1]);
    chunk = py::cast<int64_t>(shape[0]);
    itemsize = dtype.itemsize();
    if (itemsize <= 0 ||
        chunk > int64_t(NativeChunkExecutor::maxChunk) / itemsize)
      throw py::value_error("native chunk exceeds decoder workspace");
    addresses = py::cast<py::array>(spec[2]);
    sizes = py::cast<py::array>(spec[3]);
    masks = py::cast<py::array>(spec[4]);
    auto valid = [](const py::array &array, const py::dtype &dtype) {
      return array.ndim() == 2 && sameDtype(array.dtype(), dtype) &&
             (array.flags() & py::array::c_style) &&
             (array.flags() & py::detail::npy_api::NPY_ARRAY_ALIGNED_);
    };
    if (!valid(addresses, py::dtype::of<int64_t>()) ||
        !valid(sizes, py::dtype::of<uint32_t>()) ||
        !valid(masks, py::dtype::of<uint32_t>()))
      throw py::value_error("invalid native chunk directory arrays");
    count = addresses.shape(0);
    stride = addresses.shape(1);
    checkShape(sizes, {count, stride});
    checkShape(masks, {count, stride});
    if (stride <= 0 || count > INT64_MAX / chunk)
      throw py::value_error("invalid native directory dimensions");
    positions = static_cast<const int64_t *>(addresses.data());
    lengths = static_cast<const uint32_t *>(sizes.data());
    filters = static_cast<const uint32_t *>(masks.data());
  }
};

class FrameSampler {
  struct Query {
    int64_t row, col, index, position = -1;
  };
  struct Span {
    int64_t first, last;
  };
  struct Job {
    int64_t index, first, last;
    std::unique_ptr<unsigned char[]> data;
  };
  using Jobs = std::map<int64_t, std::shared_ptr<Job>>;
  using Store = void (*)(void *, int64_t, const unsigned char *, int64_t);

  BloscReader &reader;
  const FrameDirectory &coordinates, &values;
  const py::function &check;
  int fd;
  bool direct;
  std::atomic<bool> stopped{false};
  std::vector<std::future<int>> pending;
  std::chrono::steady_clock::time_point checked =
      std::chrono::steady_clock::now();

  void checkpoint(bool force = false) {
    const auto now = std::chrono::steady_clock::now();
    if (force || now - checked >= std::chrono::milliseconds(25)) {
      py::gil_scoped_acquire acquire;
      check();
      checked = now;
    }
  }
  void drain() noexcept {
    stopped.store(true, std::memory_order_relaxed);
    for (auto &future : pending)
      if (future.valid())
        future.wait();
    pending.clear();
  }
  bool allocated(const FrameDirectory &directory, int column, int64_t index) {
    if (index < 0 || index >= directory.count || column >= directory.stride)
      return false;
    const int64_t slot = index * directory.stride + column;
    return directory.positions[slot] >= 0 && directory.lengths[slot] > 0 &&
           directory.filters[slot] <= 1;
  }
  int load(const FrameDirectory &directory, int column, const Jobs &jobs) {
    pending.clear();
    pending.reserve(jobs.size());
    // Metadata and jobs are coordinator-owned. Workers never touch HDF5,
    // NumPy, reference counts, the UI or any Python callback.
    for (const auto &entry : jobs) {
      const auto job = entry.second;
      const int64_t slot = job->index * directory.stride + column;
      const int64_t offset = directory.positions[slot],
                    size = directory.lengths[slot];
      const int mask = directory.filters[slot];
      const int64_t expected = directory.chunk * directory.itemsize;
      if (size > expected + 16)
        return -2;
      pending.push_back(reader.executor->submit(
          [this, job, offset, size, mask, expected,
           &directory](NativeChunkExecutor::Buffer &buffer) {
            if (stopped.load(std::memory_order_relaxed))
              return -4;
            job->data.reset(new unsigned char[size_t(expected)]);
            const int64_t begin = direct ? offset / 4096 * 4096 : offset;
            const int64_t skip = offset - begin;
            const int64_t count =
                direct ? (skip + size + 4095) / 4096 * 4096 : size;
            return reader.readData(fd, begin, count, skip, size, buffer.get(),
                                   job->data.get(), expected,
                                   directory.itemsize, mask, job->first,
                                   job->last, &stopped);
          }));
    }
    for (auto &future : pending) {
      while (future.wait_for(std::chrono::milliseconds(25)) !=
             std::future_status::ready)
        checkpoint(true);
      checkpoint();
      const int result = future.get();
      if (result < 0)
        return result;
      decodedBytes += result;
      ++chunkReads;
    }
    pending.clear();
    return 1;
  }
  static std::shared_ptr<Job> job(int64_t index, Span span) {
    auto output = std::make_shared<Job>();
    output->index = index;
    output->first = span.first;
    output->last = span.last;
    return output;
  }
  template <typename T>
  static void storeValue(void *output, int64_t index, const unsigned char *data,
                         int64_t offset) {
    // Match CSR's zero += stored_value, including NaNs and stored -0.0.
    static_cast<T *>(output)[index] =
        addValue(T(0), reinterpret_cast<const T *>(data)[offset]);
  }

public:
  int64_t decodedBytes = 0, chunkReads = 0;
  FrameSampler(BloscReader &reader, const FrameDirectory &coordinates,
               const FrameDirectory &values, const py::function &check, int fd,
               bool direct)
      : reader(reader), coordinates(coordinates), values(values), check(check),
        fd(fd), direct(direct) {}
  ~FrameSampler() { drain(); }
  void finish() noexcept { drain(); }

  static Store store(const py::dtype &dtype) {
#define FRAME_VALUE(T)                                                         \
  if (sameDtype(dtype, py::dtype::of<T>()))                                    \
  return &storeValue<T>
    FRAME_VALUE(uint8_t);
    FRAME_VALUE(int8_t);
    FRAME_VALUE(uint16_t);
    FRAME_VALUE(int16_t);
    FRAME_VALUE(uint32_t);
    FRAME_VALUE(int32_t);
    FRAME_VALUE(uint64_t);
    FRAME_VALUE(int64_t);
    FRAME_VALUE(float);
    FRAME_VALUE(double);
#undef FRAME_VALUE
    return nullptr;
  }

  int run(const int64_t *ptr, int64_t height, int64_t start, int64_t stop,
          const int64_t *rows, const int64_t *cols, int64_t n, int64_t width,
          int64_t workspace, void *output, Store store) {
    checkpoint(true);
    if (start < 0 || stop < start ||
        stop > coordinates.count * coordinates.chunk ||
        stop > values.count * values.chunk || ptr[0] != 0 ||
        ptr[height] != stop - start)
      return 0;
    for (int64_t row = 0; row < height; ++row)
      if (ptr[row] < 0 || ptr[row + 1] < ptr[row] ||
          ptr[row + 1] > stop - start)
        return 0;
    std::vector<Query> queries;
    queries.reserve(size_t(n));
    for (int64_t i = 0; i < n; ++i) {
      if (rows[i] < 0 || rows[i] >= height)
        return 0;
      queries.push_back({rows[i], cols[i], i, -1});
    }
    std::sort(queries.begin(), queries.end(),
              [](const Query &a, const Query &b) {
                return a.row != b.row ? a.row < b.row : a.col < b.col;
              });
    std::map<int64_t, Span> spans;
    int64_t previous = -1;
    for (const auto &query : queries) {
      if (query.row == previous)
        continue;
      previous = query.row;
      for (int64_t first = start + ptr[query.row],
                   last = start + ptr[query.row + 1];
           first < last;) {
        const int64_t index = first / coordinates.chunk;
        const int64_t end = std::min(last, (index + 1) * coordinates.chunk);
        const Span incoming{first % coordinates.chunk,
                            first % coordinates.chunk + end - first};
        auto inserted = spans.emplace(index, incoming);
        auto &span = inserted.first->second;
        span.first = std::min(span.first, incoming.first);
        span.last = std::max(span.last, incoming.last);
        if (spans.size() > 128 ||
            int64_t(spans.size()) * coordinates.chunk * 2 +
                    n * int64_t(sizeof(Query)) + 65536 >
                workspace)
          return 2;
        first = end;
      }
    }
    Jobs columns;
    for (const auto &span : spans) {
      if (!allocated(coordinates, 1, span.first))
        return 2; // HDF5 supplies missing chunks / unusual layouts.
      columns.emplace(span.first, job(span.first, span.second));
    }
    int status = load(coordinates, 1, columns);
    if (status != 1)
      return status;
    auto column = [&](int64_t position) {
      return reinterpret_cast<const uint16_t *>(
          columns.at(position / coordinates.chunk)
              ->data.get())[position % coordinates.chunk];
    };
    previous = -1;
    for (auto &query : queries) {
      checkpoint();
      const int64_t first = start + ptr[query.row],
                    last = start + ptr[query.row + 1];
      if (query.row != previous) {
        uint16_t before = 0;
        for (int64_t pos = first; pos < last;) {
          const int64_t offset = pos % coordinates.chunk;
          const int64_t count =
              std::min(last - pos, coordinates.chunk - offset);
          const auto begin =
              reinterpret_cast<const uint16_t *>(
                  columns.at(pos / coordinates.chunk)->data.get()) +
              offset;
          const auto end = begin + count;
          if (uint64_t(end[-1]) >= uint64_t(width) ||
              (pos != first && *begin <= before) ||
              std::adjacent_find(begin, end, [](uint16_t a, uint16_t b) {
                return a >= b;
              }) != end)
            return 0; // Preserve the original CSR coalescing fallback.
          before = end[-1];
          pos += count;
        }
        previous = query.row;
      }
      if (first == last || query.col < 0 || query.col >= width)
        continue;
      if (first / coordinates.chunk == (last - 1) / coordinates.chunk) {
        const auto begin =
            reinterpret_cast<const uint16_t *>(
                columns.at(first / coordinates.chunk)->data.get()) +
            first % coordinates.chunk;
        const auto end = begin + last - first;
        const auto found = std::lower_bound(begin, end, query.col);
        if (found != end && *found == query.col)
          query.position = first + found - begin;
      } else {
        int64_t lo = first, hi = last;
        while (lo < hi) {
          const int64_t mid = lo + (hi - lo) / 2;
          if (column(mid) < query.col)
            lo = mid + 1;
          else
            hi = mid;
        }
        if (lo < last && column(lo) == query.col)
          query.position = lo;
      }
    }
    columns.clear();
    spans.clear();
    for (const auto &query : queries) {
      if (query.position < 0)
        continue;
      const int64_t index = query.position / values.chunk,
                    offset = query.position % values.chunk;
      auto inserted = spans.emplace(index, Span{offset, offset + 1});
      auto &span = inserted.first->second;
      span.first = std::min(span.first, offset);
      span.last = std::max(span.last, offset + 1);
      if (spans.size() > 128 || !allocated(values, 0, index))
        return 2;
    }
    // Intensity chunks are consumed in fixed-size batches; decoded output
    // storage is never proportional to the number of requested disk chunks.
    auto span = spans.begin();
    size_t position = 0;
    while (span != spans.end()) {
      Jobs batch;
      for (int i = 0; i < reader.workers && span != spans.end(); ++i, ++span)
        batch.emplace(span->first, job(span->first, span->second));
      status = load(values, 0, batch);
      if (status != 1)
        return status;
      const int64_t last = batch.rbegin()->first;
      while (position < queries.size()) {
        const auto &query = queries[position];
        if (query.position >= 0) {
          const int64_t index = query.position / values.chunk;
          if (index > last)
            break;
          store(output, query.index, batch.at(index)->data.get(),
                query.position % values.chunk);
        }
        ++position;
      }
    }
    return 1;
  }
};

py::tuple BloscReader::readPixels(int fd, py::tuple coordinateSpec,
                                  py::tuple valueSpec, CArray<int64_t> offsets,
                                  int64_t start, int64_t stop,
                                  CArray<int64_t> rows, CArray<int64_t> cols,
                                  int64_t width, bool direct, int64_t workspace,
                                  py::function check) {
  checkShape(offsets, {-1});
  checkShape(rows, {-1});
  checkShape(cols, {rows.size()});
  if (offsets.size() < 1 || width <= 0 || workspace < 0 ||
      workspace > 16 * 1024 * 1024)
    throw py::value_error("invalid native frame dimensions or workspace");
  FrameDirectory coordinates(coordinateSpec), values(valueSpec);
  const auto dtype = py::cast<py::dtype>(valueSpec[1]);
  auto store = FrameSampler::store(dtype);
  if (rows.size() > 65536 || offsets.size() > 65537 || !store ||
      !sameDtype(py::cast<py::dtype>(coordinateSpec[1]),
                 py::dtype::of<uint16_t>()))
    return py::make_tuple(py::none(), 2, 0, 0);
  std::unique_lock<std::mutex> lock(frameMutex, std::try_to_lock);
  if (!lock)
    return py::make_tuple(py::none(), 2, 0, 0);
  if (!executor)
    executor.reset(new NativeChunkExecutor(workers));
  py::array output(dtype, std::vector<py::ssize_t>{rows.size()});
  std::memset(output.mutable_data(), 0, size_t(output.nbytes()));
  FrameSampler sampler(*this, coordinates, values, check, fd, direct);
  int status;
  {
    py::gil_scoped_release release;
    // Destruction/draining happens before the caller may close descriptors.
    try {
      status = sampler.run(offsets.data(), offsets.size() - 1, start, stop,
                           rows.data(), cols.data(), rows.size(), width,
                           workspace, output.mutable_data(), store);
    } catch (...) {
      sampler.finish();
      throw;
    }
    sampler.finish();
  }
  return py::make_tuple(status == 1 ? py::object(output)
                                    : py::object(py::none()),
                        status, sampler.decodedBytes, sampler.chunkReads);
}

} // namespace
} // namespace spot_extraction
