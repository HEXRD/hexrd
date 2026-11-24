#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <vector>

// Enumerate the exact Prim visitation used by scipy. The implicit graph
// contains edges no longer than cutoff. Any edge missing from its frontier
// is longer than every queued edge. On an empty frontier enlarge the radius
// and enumerate all newly reachable edges crossing the current cut.
template <class Distance, class Emit, class Check>
bool sparsePrim(const double *q, const double *images,
                const std::vector<double> &scalar, int n, int sym,
                Distance distance, Emit emit, Check check) {
  if (!sym || n < 2)
    return false;
  auto unit = [&](const double *p, int64_t size) {
    for (int64_t j = 0; j < size; j++, p += 4) {
      if ((j & 65535) == 0)
        check();
      double n2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + p[3] * p[3];
      if (!(std::abs(n2 - 1.) < 1e-14))
        return false;
    }
    return true;
  };
  if (!unit(q, n) || !unit(images, int64_t(n) * sym))
    return false;
  std::vector<uint8_t> used(n);
  std::vector<int> link(n), head;
  std::vector<size_t> seen(n);
  std::vector<double> nearest(n, INFINITY);
  std::set<std::pair<double, int>> queue;
  std::array<int, 3> lo, hi, dim;
  double cutoff = std::min(3., 1.4 / std::cbrt(n)), width;
  auto configure = [&] {
    // For unit quaternions the smaller chord to +/- an image is 2 sin(angle/4).
    // The allowance exceeds the norm and rounding errors of the four-term
    // products, including the conjugate image reconstructed below.
    // A larger cell width only adds candidates. Bound the table even for
    // widely separated input orientations or very large candidate counts.
    double side = std::cbrt(std::min(8000000., 8. * n));
    width = std::max(std::sqrt(4 * std::pow(std::sin(cutoff * .25), 2) + 1e-12),
                     2.01 / std::max(1., side - 5.));
    lo.fill(1000000);
    hi.fill(-1000000);
    for (int i = 0; i < n; i++)
      for (int k = 0; k < 3; k++) {
        int val = std::floor(q[size_t(4) * i + 1 + k] / width);
        lo[k] = std::min(lo[k], val);
        hi[k] = std::max(hi[k], val);
      }
    for (int k = 0; k < 3; k++)
      dim[k] = hi[k] - lo[k] + 5;
    head.assign(size_t(dim[0]) * dim[1] * dim[2], -1);
    for (int i = 0; i < n; i++) {
      int key = 0;
      for (int k = 0; k < 3; k++)
        key = key * dim[k] + int(std::floor(q[size_t(4) * i + 1 + k] / width)) -
              lo[k] + 2;
      link[i] = head[key];
      head[key] = i;
    }
  };
  configure();
  size_t stamp = 0;
  auto visit = [&](int x) {
    ++stamp;
    auto p = q + size_t(4) * x;
    for (int s = 0; s < sym; s++) {
      auto im = images + 4 * (int64_t(x) * sym + s);
      // For raw target i<x look around x*s. For i>x look around x*conj(s),
      // since <x,i*s> = <x*conj(s),i>. Exact weights always call distance().
      for (int side = 0; side < 2; side++)
        for (int sign : {1, -1}) {
          int key = 0;
          bool inside = true;
          for (int k = 0; k < 3; k++) {
            double v = side ? 2 * scalar[s] * p[k + 1] - im[k + 1] : im[k + 1];
            int c = std::floor(sign * v / width);
            if (c < lo[k] - 1 || c > hi[k] + 1) {
              inside = false;
              break;
            }
            key = key * dim[k] + c - lo[k] + 2;
          }
          if (!inside)
            continue;
          for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
              for (int k = -1; k <= 1; k++) {
                for (int y = head[key + (i * dim[1] + j) * dim[2] + k]; y >= 0;
                     y = link[y]) {
                  if (used[y] == used[x] || seen[y] == stamp ||
                      ((y > x) != bool(side)))
                    continue;
                  seen[y] = stamp;
                  const int target = used[x] ? y : x;
                  double cost = distance(std::min(x, y), std::max(x, y));
                  if (cost <= cutoff && cost < nearest[target]) {
                    if (std::isfinite(nearest[target]))
                      queue.erase({nearest[target], target});
                    nearest[target] = cost;
                    queue.insert({cost, target});
                  }
                }
              }
        }
    }
  };
  int x = 0;
  for (int count = 1; count < n; count++) {
    if ((count & 63) == 1)
      check();
    used[x] = true;
    visit(x);
    while (queue.empty()) {
      cutoff = std::min(std::acos(-1.), cutoff * 2.);
      configure();
      bool smaller = count < n - count;
      for (int j = 0; j < n; j++)
        if (used[j] == smaller) {
          if ((j & 255) == 0)
            check();
          visit(j);
        }
      if (queue.empty() && cutoff >= std::acos(-1.))
        return false;
    }
    auto edge = *queue.begin();
    queue.erase(queue.begin());
    emit(x, edge.second, edge.first);
    x = edge.second;
  }
  return true;
}
