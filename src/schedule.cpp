#include "hlbl/schedule.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace hlbl {

namespace {

bool is_diagonal_separation(const Vec4& d, int stride) {
  for (int i = 0; i < 4; ++i) {
    if (std::abs(d[i]) != stride) return false;
  }
  return true;
}

bool contains(const std::vector<Vec4>& v, const Vec4& p) {
  for (const auto& q : v) {
    if (q == p) return true;
  }
  return false;
}

std::pair<Vec4, Vec4> canonical_pair(const Vec4& a, const Vec4& b) {
  return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

}  // namespace

std::array<Vec4, 16> sign_vectors() {
  std::array<Vec4, 16> g{};
  int idx = 0;
  for (int a : {-1, 1}) {
    for (int b : {-1, 1}) {
      for (int c : {-1, 1}) {
        for (int d : {-1, 1}) {
          g[idx++] = {a, b, c, d};
        }
      }
    }
  }
  return g;
}

std::vector<Vec4> window_offsets(const Vec4& h) {
  std::vector<Vec4> offsets = {Vec4{0, 0, 0, 0}};
  for (const auto& g : sign_vectors()) {
    if (g != h) offsets.push_back(g);
  }
  return offsets;
}

Schedule build_schedule(const Vec4& h, int stride, int n_steps,
                         const Vec4& origin) {
  auto offsets = window_offsets(h);

  Schedule steps;
  std::vector<Vec4> previous_window;
  for (int k = 0; k <= n_steps; ++k) {
    std::vector<Vec4> window;
    window.reserve(offsets.size());
    for (const auto& a : offsets) {
      window.push_back(add(origin, scale(stride, add(scale(k, h), a))));
    }

    std::vector<Vec4> available = previous_window;
    Step step;
    step.index = k;
    for (const auto& p : window) {
      if (contains(available, p)) continue;  // shared with window k-1
      Source src;
      src.position = p;
      for (const auto& q : available) {
        if (is_diagonal_separation(sub(p, q), stride)) src.partners.push_back(q);
      }
      step.sources.push_back(std::move(src));
      available.push_back(p);
    }
    steps.push_back(std::move(step));
    previous_window = std::move(window);
  }
  return steps;
}

std::vector<Vec4> all_positions(const Schedule& steps) {
  std::vector<Vec4> positions;
  for (const auto& step : steps) {
    for (const auto& src : step.sources) positions.push_back(src.position);
  }
  return positions;
}

std::vector<std::pair<Vec4, Vec4>> all_pairs(const Schedule& steps) {
  std::vector<std::pair<Vec4, Vec4>> pairs;
  for (const auto& step : steps) {
    for (const auto& src : step.sources) {
      for (const auto& q : src.partners) pairs.emplace_back(q, src.position);
    }
  }
  return pairs;
}

void check_complete(const Schedule& steps, int stride) {
  auto positions = all_positions(steps);
  std::set<Vec4> pos_set(positions.begin(), positions.end());
  if (pos_set.size() != positions.size()) {
    throw std::runtime_error("check_complete: a propagator is scheduled twice");
  }

  std::set<std::pair<Vec4, Vec4>> expected;
  for (std::size_t i = 0; i < positions.size(); ++i) {
    for (std::size_t j = i + 1; j < positions.size(); ++j) {
      if (is_diagonal_separation(sub(positions[i], positions[j]), stride)) {
        expected.insert(canonical_pair(positions[i], positions[j]));
      }
    }
  }

  std::set<std::pair<Vec4, Vec4>> scheduled;
  for (const auto& [q, p] : all_pairs(steps)) {
    if (!scheduled.insert(canonical_pair(q, p)).second) {
      throw std::runtime_error("check_complete: a pair is scheduled twice");
    }
  }

  if (scheduled != expected) {
    throw std::runtime_error(
        "check_complete: scheduled pairs do not match every |y|=2n pair "
        "among the scheduled sources");
  }
}

int max_steps_without_wrap(const Vec4& h, int stride, const Vec4& lattice,
                            const Vec4& origin, int cap) {
  auto offsets = window_offsets(h);
  std::set<Vec4> seen;
  for (int k = 0; k < cap; ++k) {
    std::set<Vec4> window;
    for (const auto& a : offsets) {
      Vec4 p = add(origin, scale(stride, add(scale(k, h), a)));
      window.insert(mod_pos(p, lattice));
    }
    std::set<Vec4> new_sites;
    if (k == 0) {
      new_sites = window;
    } else {
      Vec4 old_center = mod_pos(add(origin, scale(stride * (k - 1), h)), lattice);
      for (const auto& s : window) {
        if (s != old_center) new_sites.insert(s);
      }
    }
    for (const auto& s : new_sites) {
      if (seen.count(s)) return k - 1;
    }
    seen.insert(new_sites.begin(), new_sites.end());
  }
  return cap;
}

ScheduleStats simulate(const Schedule& steps, int n_flavors, Mode mode) {
  std::map<Vec4, int> pending;
  for (const auto& [older, newer] : all_pairs(steps)) {
    pending[older]++;
    pending[newer]++;
  }

  std::set<Vec4> cache;
  int peak = 0;
  ScheduleStats stats;

  auto invert = [&](const Vec4& p) {
    cache.insert(p);
    peak = std::max(peak, static_cast<int>(cache.size()));
    stats.n_propagators++;
    stats.n_solves += 12 * n_flavors;
  };
  auto contract = [&](const Vec4& older, const Vec4& newer) {
    stats.n_kernel_sums += n_flavors;
    stats.n_pairs++;
    for (const auto& pos : {older, newer}) {
      if (--pending[pos] == 0) cache.erase(pos);
    }
  };

  for (std::size_t k = 0; k < steps.size(); ++k) {
    const Step& step = steps[k];
    if (mode == Mode::kSerial || k == 0) {
      for (const auto& src : step.sources) invert(src.position);
    }
    if (mode == Mode::kPipelined && k + 1 < steps.size()) {
      for (const auto& src : steps[k + 1].sources) invert(src.position);
    }
    for (const auto& src : step.sources) {
      for (const auto& q : src.partners) contract(q, src.position);
    }
  }

  if (!cache.empty()) {
    throw std::runtime_error(
        "simulate: propagators left in the cache after the last pair");
  }
  stats.peak_cache = peak;
  return stats;
}

}  // namespace hlbl
