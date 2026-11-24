#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

// DBSCAN with one required sample is the connected components of the closed
// radius graph. Index three coordinates and test the full Euclidean distance.
// Return false for inputs better suited to the general neighbor search.
template <class Check>
bool radiusComponents(const double *points, int n, int dims, double eps,
                      int *labels, Check check) {
  const double squared = eps * eps;
  if (!n || (dims != 3 && dims != 4) || !(eps > 0.) || !std::isnormal(squared))
    return false;
  const double width = eps * (1. + 1e-6);
  std::array<int, 3> lo{1000000, 1000000, 1000000},
      hi{-1000000, -1000000, -1000000};
  unsigned comparisons = 0;
  for (int i = 0; i < n; ++i) {
    if (i % 16384 == 0)
      check();
    const auto *p = points + size_t(i) * dims;
    if (!std::isfinite(p[0]))
      return false;
    for (int k = 0; k < 3; ++k) {
      double cell = std::floor(p[dims - 3 + k] / width);
      // At this scale the padding in width exceeds coordinate rounding.
      if (!(std::abs(cell) <= 1000000))
        return false;
      lo[k] = std::min(lo[k], int(cell));
      hi[k] = std::max(hi[k], int(cell));
    }
  }
  std::array<int, 3> size;
  int cells = 1, limit = int(std::min(32LL * n, 16LL * 1024 * 1024));
  for (int k = 0; k < 3; ++k) {
    size[k] = hi[k] - lo[k] + 3;
    if (cells > limit / size[k])
      return false;
    cells *= size[k];
    --lo[k];
  }
  std::vector<int> head(cells, -1), next(n);
  std::iota(labels, labels + n, 0);
  auto root = [&](int i) {
    while (i != labels[i]) {
      labels[i] = labels[labels[i]];
      i = labels[i];
    }
    return i;
  };
  for (int i = 0; i < n; ++i) {
    if (i % 1024 == 0)
      check();
    const auto *p = points + size_t(i) * dims;
    int key = 0;
    for (int k = 0; k < 3; ++k)
      key = key * size[k] + int(std::floor(p[dims - 3 + k] / width)) - lo[k];
    if (head[key] >= 0 &&
        std::equal(p, p + dims, points + size_t(head[key]) * dims)) {
      labels[i] = root(head[key]);
      continue;
    }
    int left = i;
    for (int x = -1; x <= 1; ++x)
      for (int y = -1; y <= 1; ++y)
        for (int z = -1; z <= 1; ++z)
          for (int j = head[key + (x * size[1] + y) * size[2] + z]; j >= 0;
               j = next[j]) {
            if (++comparisons % 262144 == 0)
              check();
            const auto *q = points + size_t(j) * dims;
            double distance = 0.;
            for (int k = 0; k < dims; ++k)
              distance += (p[k] - q[k]) * (p[k] - q[k]);
            if (distance <= squared) {
              int right = root(j);
              labels[std::max(left, right)] = std::min(left, right);
              left = std::min(left, right);
            }
          }
    next[i] = head[key];
    head[key] = i;
  }
  for (int i = 0; i < n; ++i)
    labels[i] = root(i);
  int clusters = 0;
  for (int i = 0; i < n; ++i)
    labels[i] = labels[i] == i ? clusters++ : labels[labels[i]];
  return true;
}
