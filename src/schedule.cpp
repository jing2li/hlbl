#include "hlbl/schedule.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <tuple>

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
  // A reflection keeps the pair separations, so the schedule of h mirrors that
  // of (1,1,1,1), peak 16 included. G \ {h} in sign_vectors() order would
  // peak at 17 to 25.
  std::vector<Vec4> offsets = {Vec4{0, 0, 0, 0}};
  for (const auto& g : sign_vectors()) {
    if (g == Vec4{1, 1, 1, 1}) continue;
    offsets.push_back({h[0] * g[0], h[1] * g[1], h[2] * g[2], h[3] * g[3]});
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

ScheduleStats simulate(const Schedule& steps, Mode mode) {
  constexpr int n_flavors = 2;
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

  for (const Step& step : steps) {
    if (mode == Mode::kInterleaved) {
      for (const auto& src : step.sources) {
        invert(src.position);
        for (const auto& q : src.partners) contract(q, src.position);
      }
      continue;
    }
    for (const auto& src : step.sources) invert(src.position);
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

std::vector<std::string> overlap_report(const std::vector<Schedule>& chains,
                                        const Vec4& lattice) {
  using Pair = std::pair<Vec4, Vec4>;
  std::size_t const n = chains.size();
  std::vector<std::map<Vec4, int>> sources(n);  // position -> window
  std::vector<std::set<Pair>> pairs(n);
  std::map<Vec4, int> source_count;
  std::map<Pair, int> pair_count;
  for (std::size_t c = 0; c < n; ++c) {
    for (const Step& step : chains[c]) {
      for (const Source& src : step.sources) {
        Vec4 const p = mod_pos(src.position, lattice);
        sources[c].emplace(p, step.index);
        for (const Vec4& q : src.partners) {
          pairs[c].insert(canonical_pair(p, mod_pos(q, lattice)));
        }
      }
    }
    for (const auto& [p, window] : sources[c]) source_count[p]++;
    for (const auto& p : pairs[c]) pair_count[p]++;
  }
  auto repeated = [](const auto& m) {
    return static_cast<long>(std::count_if(
        m.begin(), m.end(), [](const auto& kv) { return kv.second > 1; }));
  };
  long const s = repeated(source_count), p = repeated(pair_count);
  if (s == 0 && p == 0) return {};

  std::vector<std::string> lines;
  char buf[256];
  auto add = [&](auto... args) {
    std::snprintf(buf, sizeof buf, args...);
    lines.emplace_back(buf);
  };
  add("WARNING: %ld source positions are in more than one chain (solved once "
      "per chain) and %ld pairs are computed in more than one chain", s, p);
  constexpr int kMaxLines = 20;     // pairs of chains listed
  constexpr long kMaxSources = 10;  // shared sources listed per pair of chains
  int shown = 0, more = 0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      long ns = 0;
      std::vector<std::tuple<Vec4, int, int>> shared;  // position, windows
      for (const auto& [pos, window_i] : sources[i]) {
        auto const it = sources[j].find(pos);
        if (it == sources[j].end()) continue;
        if (ns++ < kMaxSources) shared.emplace_back(pos, window_i, it->second);
      }
      long np = 0;
      for (const auto& x : pairs[i]) np += static_cast<long>(pairs[j].count(x));
      if (ns == 0 && np == 0) continue;
      if (shown >= kMaxLines) {
        ++more;
        continue;
      }
      ++shown;
      add("chains %zu and %zu share %ld sources and %ld pairs", i, j, ns, np);
      for (const auto& [pos, window_i, window_j] : shared) {
        add("  source (%d,%d,%d,%d) in window %d of chain %zu and window %d of "
            "chain %zu", pos[0], pos[1], pos[2], pos[3], window_i, i, window_j, j);
      }
      if (ns > kMaxSources) add("  ... and %ld more sources", ns - kMaxSources);
    }
  }
  if (more > 0) add("... and %d more pairs of chains", more);
  return lines;
}

Plan2p2 plan_2p2(const std::vector<Schedule>& chains,
                 const std::vector<Vec4>& sources, const Vec4& lattice) {
  std::map<Vec4, int> index;  // wrapped 2+2 source -> its index
  for (std::size_t i = 0; i < sources.size(); ++i) {
    index.emplace(mod_pos(sources[i], lattice), static_cast<int>(i));
  }
  Plan2p2 plan;
  plan.at.resize(chains.size());
  std::set<Vec4> solved;  // wrapped sites solved so far
  for (std::size_t c = 0; c < chains.size(); ++c) {
    for (const Step& step : chains[c]) {
      for (const Source& src : step.sources) {
        Vec4 const p = mod_pos(src.position, lattice);
        if (!solved.insert(p).second) continue;  // not the first solve
        auto const it = index.find(p);
        if (it != index.end()) {
          plan.at[c].emplace(src.position, it->second);
        } else {
          plan.not_sources.emplace_back(p, static_cast<int>(c), step.index);
        }
      }
    }
  }
  for (std::size_t i = 0; i < sources.size(); ++i) {
    if (!solved.count(mod_pos(sources[i], lattice))) {
      plan.unsolved.push_back(static_cast<int>(i));
    }
  }
  return plan;
}

std::vector<std::string> report_2p2(const Plan2p2& plan,
                                    const std::vector<Vec4>& sources) {
  std::vector<std::string> lines;
  if (sources.empty()) return lines;
  char buf[256];
  auto add = [&](auto... args) {
    std::snprintf(buf, sizeof buf, args...);
    lines.emplace_back(buf);
  };
  constexpr long kMaxSites = 10;  // sites listed per warning
  long const n_not = static_cast<long>(plan.not_sources.size());
  if (n_not > 0) {
    add("WARNING: %ld solved source positions are not 2+2 sources (no 2+2 "
        "computed for them)", n_not);
    for (long i = 0; i < std::min(n_not, kMaxSites); ++i) {
      const auto& [p, chain, window] = plan.not_sources[i];
      add("  (%d,%d,%d,%d) in window %d of chain %d", p[0], p[1], p[2], p[3],
          window, chain);
    }
    if (n_not > kMaxSites) add("  ... and %ld more", n_not - kMaxSites);
  }
  long const n_unsolved = static_cast<long>(plan.unsolved.size());
  if (n_unsolved > 0) {
    add("WARNING: %ld 2+2 sources are not solved by any chain (no 2+2 "
        "computed for them)", n_unsolved);
    for (long i = 0; i < std::min(n_unsolved, kMaxSites); ++i) {
      const Vec4& p = sources[plan.unsolved[i]];
      add("  (%d,%d,%d,%d)", p[0], p[1], p[2], p[3]);
    }
    if (n_unsolved > kMaxSites) add("  ... and %ld more", n_unsolved - kMaxSites);
  }
  return lines;
}

}  // namespace hlbl
