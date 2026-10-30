#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace spot_extraction {
namespace {

#ifndef _WIN32
class NativeChunkExecutor {
public:
  static constexpr size_t maxChunk = 1024 * 1024;
  struct Buffer {
    unsigned char *source = nullptr;
    std::unique_ptr<unsigned char[]> decoded;
    unsigned char *output() {
      if (!decoded)
        decoded.reset(new unsigned char[maxChunk]);
      return decoded.get();
    }
    unsigned char *get() {
      if (!source) {
        void *raw = nullptr;
        if (posix_memalign(&raw, 4096, maxChunk + 8192))
          throw std::bad_alloc();
        source = static_cast<unsigned char *>(raw);
      }
      return source;
    }
    ~Buffer() { std::free(source); }
  };

private:
  std::mutex mutex;
  std::condition_variable ready;
  std::deque<std::function<void(Buffer &)>> pending;
  std::vector<std::thread> threads;
  bool stopped = false;

public:
  explicit NativeChunkExecutor(int count) {
    try {
      for (int i = 0; i < count; ++i)
        threads.emplace_back([this] {
          Buffer buffer;
          for (;;) {
            std::function<void(Buffer &)> task;
            {
              std::unique_lock<std::mutex> lock(mutex);
              ready.wait(lock, [this] { return stopped || !pending.empty(); });
              if (pending.empty())
                return;
              task = std::move(pending.front());
              pending.pop_front();
            }
            task(buffer);
          }
        });
    } catch (...) {
      close();
      throw;
    }
  }
  std::future<int> submit(std::function<int(Buffer &)> function) {
    auto task = std::make_shared<std::packaged_task<int(Buffer &)>>(
        std::move(function));
    auto future = task->get_future();
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (stopped)
        throw std::runtime_error("native chunk executor is closed");
      pending.emplace_back([task](Buffer &buffer) { (*task)(buffer); });
    }
    ready.notify_one();
    return future;
  }
  void close() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopped = true;
      ready.notify_all();
    }
    for (auto &thread : threads)
      if (thread.joinable())
        thread.join();
    threads.clear();
  }
  ~NativeChunkExecutor() { close(); }
};
#endif

} // namespace
} // namespace spot_extraction
