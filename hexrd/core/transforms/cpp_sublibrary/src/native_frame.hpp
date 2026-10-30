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
  struct Read {
    int64_t offset, size;
    std::function<int(const unsigned char *, NativeChunkExecutor::Buffer &)>
        consume;
  };
  // Merge nearby chunks in physical order without reducing the available
  // batches below the decoder-worker count. Input and output buffers belong
  // to each worker and remain inside its existing byte reservation.
  int runReads(std::vector<Read> work) {
    std::sort(work.begin(), work.end(),
              [](const Read &a, const Read &b) { return a.offset < b.offset; });
    auto reads = std::make_shared<std::vector<Read>>(std::move(work));
    pending.clear();
    pending.reserve(reads->size());
    const size_t limit =
        std::max(size_t(1), reads->size() / size_t(reader.workers));
    for (size_t first = 0; first < reads->size();) {
      size_t last = first + 1;
      const int64_t offset = (*reads)[first].offset;
      int64_t end = offset + (*reads)[first].size;
      while (last < reads->size() && last - first < limit) {
        const auto &next = (*reads)[last];
        const int64_t stop = std::max(end, next.offset + next.size);
        if (next.offset - end > 4096 ||
            stop - offset > int64_t(NativeChunkExecutor::maxChunk))
          break;
        end = stop;
        ++last;
      }
      const int64_t begin = direct ? offset / 4096 * 4096 : offset;
      const int64_t count =
          direct ? (end - begin + 4095) / 4096 * 4096 : end - begin;
      pending.push_back(
          reader.executor->submit([this, reads, first, last, begin, count,
                                   end](NativeChunkExecutor::Buffer &buffer) {
            int status = reader.readSource(fd, begin, count, end - begin,
                                           buffer.get(), &stopped);
            if (status < 0)
              return status;
            int total = 0;
            for (size_t i = first; i < last; ++i) {
              if (stopped.load(std::memory_order_relaxed))
                return -4;
              status = (*reads)[i].consume(
                  buffer.get() + ((*reads)[i].offset - begin), buffer);
              if (status < 0)
                return status;
              total += status;
            }
            return total;
          }));
      first = last;
    }
    for (auto &future : pending) {
      while (future.wait_for(std::chrono::milliseconds(25)) !=
             std::future_status::ready)
        checkpoint(true);
      checkpoint();
      const int status = future.get();
      if (status < 0)
        return status;
      decodedBytes += status;
      ++chunkReads;
    }
    pending.clear();
    return 1;
  }
  int load(const FrameDirectory &directory, int column, const Jobs &jobs) {
    std::vector<Read> reads;
    for (const auto &entry : jobs) {
      const auto job = entry.second;
      const int64_t slot = job->index * directory.stride + column;
      const int64_t size = directory.lengths[slot],
                    expected = directory.chunk * directory.itemsize;
      const int mask = directory.filters[slot];
      const int64_t offset = directory.positions[slot];
      if (size > expected + 16 || offset > INT64_MAX - size)
        return -2;
      reads.push_back(
          {offset, size,
           [this, job, size, expected, mask, &directory](
               const unsigned char *source, NativeChunkExecutor::Buffer &) {
             job->data.reset(new unsigned char[size_t(expected)]);
             return reader.decodeData(source, size, job->data.get(), expected,
                                      directory.itemsize, mask, job->first,
                                      job->last);
           }});
    }
    return runReads(std::move(reads));
  }
  int loadValues(std::vector<Query> queries,
                 const std::map<int64_t, Span> &spans, void *output,
                 Store store) {
    // Queue all lightweight descriptors, allocating decoded storage only when
    // a worker starts a job. Slow chunks cannot strand the other readers at a
    // batch barrier. Each job scatters to disjoint original sample slots.
    auto requests = std::make_shared<std::vector<Query>>(std::move(queries));
    std::vector<Read> reads;
    size_t cursor = 0;
    for (const auto &entry : spans) {
      const int64_t index = entry.first;
      const Span span = entry.second;
      const size_t first = cursor;
      while (cursor < requests->size()) {
        const auto &query = (*requests)[cursor];
        if (query.position >= 0 && query.position / values.chunk > index)
          break;
        ++cursor;
      }
      const size_t last = cursor;
      const int64_t slot = index * values.stride;
      const int64_t offset = values.positions[slot],
                    size = values.lengths[slot],
                    expected = values.chunk * values.itemsize;
      const int mask = values.filters[slot];
      if (size > expected + 16)
        return -2;
      if (offset > INT64_MAX - size)
        return -2;
      reads.push_back({offset, size,
                       [this, requests, first, last, span, output, store, size,
                        expected, mask](const unsigned char *source,
                                        NativeChunkExecutor::Buffer &buffer) {
                         auto decoded = buffer.output();
                         const int result = reader.decodeData(
                             source, size, decoded, expected, values.itemsize,
                             mask, span.first, span.last);
                         if (result < 0)
                           return result;
                         for (size_t i = first; i < last; ++i) {
                           const auto &query = (*requests)[i];
                           if (query.position >= 0)
                             store(output, query.index, decoded,
                                   query.position % values.chunk);
                         }
                         return result;
                       }});
    }
    return runReads(std::move(reads));
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
    auto ordered = [](const Query &a, const Query &b) {
      return a.row != b.row ? a.row < b.row : a.col < b.col;
    };
    if (!std::is_sorted(queries.begin(), queries.end(), ordered))
      std::sort(queries.begin(), queries.end(), ordered);
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
    const uint16_t *rowBegin = nullptr, *search = nullptr, *rowEnd = nullptr;
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
        rowBegin = nullptr;
        if (first < last &&
            first / coordinates.chunk == (last - 1) / coordinates.chunk) {
          rowBegin = reinterpret_cast<const uint16_t *>(
                         columns.at(first / coordinates.chunk)->data.get()) +
                     first % coordinates.chunk;
          search = rowBegin;
          rowEnd = rowBegin + (last - first);
        }
        previous = query.row;
      }
      if (first == last || query.col < 0 || query.col >= width)
        continue;
      if (rowBegin) {
        // Sorted queries usually visit adjacent columns. Reuse the row's
        // decoded pointer and advance locally before searching a larger gap.
        for (int step = 0; step < 8 && search < rowEnd && *search < query.col;
             ++step)
          ++search;
        if (search < rowEnd && *search < query.col)
          search = std::lower_bound(search, rowEnd, query.col);
        if (search < rowEnd && *search == query.col)
          query.position = first + (search - rowBegin);
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
    return loadValues(std::move(queries), spans, output, store);
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

// Row and intensity work share the same native executor and source buffers.
int BloscReader::scanRows(int fd, const FrameDirectory &directory,
                          int64_t start, int64_t stop, Index height,
                          bool direct, std::vector<int32_t> &offsets,
                          NativeChunkExecutor::Buffer &source,
                          std::atomic<bool> &cancelled, int64_t &total) {
  offsets.assign(size_t(height + 1), int32_t(stop - start));
  auto decoded = reinterpret_cast<uint16_t *>(source.output());
  int status = 1, previous = -1;
  Index next = 0;
  for (int64_t first = start; first < stop;) {
    if (cancelled.load(std::memory_order_relaxed))
      return -4;
    const int64_t index = first / directory.chunk,
                  slot = index * directory.stride,
                  offset = directory.positions[slot],
                  size = directory.lengths[slot],
                  expected = directory.chunk * 2;
    if (offset < 0 || size <= 0 || directory.filters[slot] > 1) {
      status = 2; // HDF5 handles missing chunks and other filter layouts.
      break;
    }
    if (size > expected + 16) {
      status = -2;
      break;
    }
    const int64_t begin = direct ? offset / 4096 * 4096 : offset,
                  skip = offset - begin,
                  count = direct ? (skip + size + 4095) / 4096 * 4096 : size;
    const int got =
        readData(fd, begin, count, skip, size, source.get(),
                 reinterpret_cast<unsigned char *>(decoded), expected, 2,
                 directory.filters[slot], -1, -1, &cancelled);
    if (got < 0) {
      status = got;
      break;
    }
    total += got;
    const int64_t end = std::min(stop, (index + 1) * directory.chunk);
    if (!updateRowsRaw(decoded, first % directory.chunk,
                       first % directory.chunk + end - first, offsets.data(),
                       height, first - start, previous, next)) {
      status = 0;
      break;
    }
    first = end;
  }
  return status;
}

class RowIndexes {
  struct Data {
    std::atomic<bool> cancelled{false};
    std::vector<int32_t> offsets;
    int64_t decoded = 0;
  };
  struct Job {
    int64_t frame;
    std::shared_ptr<Data> data;
    std::future<int> future;
  };
  BloscReader &reader;
  std::shared_ptr<NativeChunkExecutor> executor;
  py::object owner;
  FrameDirectory directory;
  CArray<uint64_t> bounds;
  CArray<int64_t> frames;
  int fd;
  Index height, cursor = 0;
  bool direct;
  std::deque<Job> pending;

  void submit() {
    if (cursor == frames.size())
      return;
    const int64_t frame = frames.data()[cursor++];
    const uint64_t first = bounds.data()[2 * frame],
                   last = bounds.data()[2 * frame + 1];
    auto data = std::make_shared<Data>();
    pending.push_back({frame, data, std::future<int>()});
    pending.back().future = executor->submit(
        [this, data, first, last](NativeChunkExecutor::Buffer &source) {
          return reader.scanRows(fd, directory, int64_t(first), int64_t(last),
                                 height, direct, data->offsets, source,
                                 data->cancelled, data->decoded);
        });
  }

public:
  RowIndexes(BloscReader &reader, int fd, py::tuple spec,
             CArray<uint64_t> bounds, CArray<int64_t> frames, Index height,
             bool direct)
      : reader(reader),
        owner(py::cast(&reader, py::return_value_policy::reference)),
        directory(spec), bounds(bounds), frames(frames), fd(fd), height(height),
        direct(direct) {
    checkShape(bounds, {-1, 2});
    checkShape(frames, {-1});
    if (height < 0 || height > 32767)
      throw py::value_error("invalid row-index dimensions");
    for (Index i = 0; i < frames.size(); ++i) {
      int64_t frame = frames.data()[i];
      if (frame < 0 || frame >= bounds.shape(0) ||
          bounds.data()[2 * frame] > bounds.data()[2 * frame + 1] ||
          bounds.data()[2 * frame + 1] >
              uint64_t(directory.count * directory.chunk) ||
          bounds.data()[2 * frame + 1] - bounds.data()[2 * frame] > INT32_MAX)
        throw py::value_error("invalid row-index dimensions");
    }
    {
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> lock(reader.frameMutex);
      if (!reader.executor)
        reader.executor.reset(new NativeChunkExecutor(reader.workers));
      executor = reader.executor;
    }
    try {
      for (int i = 0; i < reader.workers; ++i)
        submit();
    } catch (...) {
      py::gil_scoped_release release;
      close();
      throw;
    }
  }
  void close() noexcept {
    for (auto &job : pending)
      job.data->cancelled.store(true, std::memory_order_relaxed);
    for (auto &job : pending)
      if (job.future.valid())
        job.future.wait();
    pending.clear();
  }
  ~RowIndexes() {
    py::gil_scoped_release release;
    close();
  }
  py::tuple get(int64_t frame, py::function check) {
    // A cached index may have been evicted after look-ahead was planned.
    // Let the ordinary reader rebuild frames absent from this sequence.
    if (pending.empty())
      return py::make_tuple(2, py::none(), 0);
    Index first = cursor - Index(pending.size());
    if (std::find(frames.data() + first, frames.data() + frames.size(),
                  frame) == frames.data() + frames.size())
      return py::make_tuple(2, py::none(), 0);
    try {
      while (!pending.empty()) {
        auto &job = pending.front();
        if (job.frame != frame)
          job.data->cancelled.store(true, std::memory_order_relaxed);
        check();
        {
          py::gil_scoped_release release;
          while (job.future.wait_for(std::chrono::milliseconds(25)) !=
                 std::future_status::ready) {
            py::gil_scoped_acquire acquire;
            check();
          }
        }
        check();
        int status = job.future.get();
        bool selected = job.frame == frame;
        py::array_t<int32_t> output(selected && status == 1 ? height + 1 : 0);
        if (selected && status == 1)
          std::copy(job.data->offsets.begin(), job.data->offsets.end(),
                    output.mutable_data());
        int64_t decoded = job.data->decoded;
        pending.pop_front();
        submit();
        if (selected)
          return py::make_tuple(
              status, status == 1 ? py::object(output) : py::object(py::none()),
              decoded);
      }
    } catch (...) {
      py::gil_scoped_release release;
      close();
      throw;
    }
    return py::make_tuple(2, py::none(), 0);
  }
};

py::object BloscReader::prefetchRows(int fd, py::tuple spec,
                                     CArray<uint64_t> bounds,
                                     CArray<int64_t> frames, Index height,
                                     bool direct) {
  if (!sameDtype(py::cast<py::dtype>(spec[1]), py::dtype::of<uint16_t>()) ||
      height > 32767)
    return py::none();
  return py::cast(std::make_shared<RowIndexes>(*this, fd, spec, bounds, frames,
                                               height, direct));
}

py::tuple BloscReader::readRows(int fd, py::tuple spec, int64_t start,
                                int64_t stop, Index height, bool direct,
                                py::function check) {
  if (start < 0 || stop < start)
    throw py::value_error("invalid row-index dimensions");
  if (!sameDtype(py::cast<py::dtype>(spec[1]), py::dtype::of<uint16_t>()) ||
      stop - start > INT32_MAX || height > 32767)
    return py::make_tuple(2, py::none(), 0);
  CArray<uint64_t> bounds({Index(1), Index(2)});
  bounds.mutable_data()[0] = uint64_t(start);
  bounds.mutable_data()[1] = uint64_t(stop);
  CArray<int64_t> frames(1);
  frames.mutable_data()[0] = 0;
  RowIndexes queue(*this, fd, spec, bounds, frames, height, direct);
  return queue.get(0, check);
}

static void bindRowIndexes(py::module_ &m) {
  py::class_<RowIndexes, std::shared_ptr<RowIndexes>>(m, "_RowIndexes")
      .def("get", &RowIndexes::get)
      .def("close", [](RowIndexes &q) {
        py::gil_scoped_release release;
        q.close();
      });
}

} // namespace
} // namespace spot_extraction
