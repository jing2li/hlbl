#include "hlbl/driver.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace hlbl {

std::vector<PairResult> run_schedule(const Lattice& lattice,
                                      const Schedule& schedule,
                                      const RunOptions& options,
                                      PropagatorProvider& provider,
                                      Contractor& contractor,
                                      ResultWriter* writer, TimingLog* timing,
                                      RunStats* stats) {
  int const n_windows = static_cast<int>(schedule.size());
  int const begin = options.window_begin;
  int const end = options.window_end < 0
                      ? n_windows
                      : std::min(options.window_end, n_windows);
  if (begin < 0 || begin >= end) {
    throw std::runtime_error("run_schedule: empty window range");
  }
  Vec4 const global = lattice.global();
  auto wrap = [&](const Vec4& p) { return mod_pos(p, global); };

  // Pairs per position within this range, and the positions from before the
  // range that its pairs need (solved again first)
  std::map<Vec4, int> pending;
  std::set<Vec4> produced;
  for (int k = begin; k < end; ++k) {
    for (const Source& src : schedule[k].sources) {
      produced.insert(src.position);
      for (const Vec4& q : src.partners) {
        pending[q]++;
        pending[src.position]++;
      }
    }
  }
  std::vector<Vec4> preload;
  std::set<Vec4> preload_set;
  for (int k = begin; k < end; ++k) {
    for (const Source& src : schedule[k].sources) {
      for (const Vec4& q : src.partners) {
        if (!produced.count(q) && preload_set.insert(q).second) {
          preload.push_back(q);
        }
      }
    }
  }

  PropagatorCache cache(options.residency, block_length(lattice), pending);
  RunStats local_stats;
  int rank = 0;
  MPI_Comm_rank(lattice.cart_comm(), &rank);

  auto solve = [&](int window, const Vec4& position) {
    Vec4 const site = wrap(position);
    DeviceBuffer<double>& dest = cache.begin_insert(position);
    double const t0 = MPI_Wtime();
    int const iterations = provider.compute(site, dest);
    double const dt = MPI_Wtime() - t0;
    cache.end_insert(position);
    local_stats.positions_solved++;
    local_stats.iterations += iterations;
    local_stats.seconds_solve += dt;
    if (timing) timing->solve(window, site, dt, iterations);
  };

  std::vector<PairResult> results;
  for (const Vec4& position : preload) solve(begin - 1, position);

  for (int k = begin; k < end; ++k) {
    std::vector<PairResult> rows;
    // Contract each source right after its solve, so propagators are freed
    // early and the cache stays small.
    for (const Source& src : schedule[k].sources) {
      solve(k, src.position);
      if (src.partners.empty()) continue;
      Vec4 const gy = wrap(src.position);
      // the new source is the pair's y; it stays staged for all its partners
      const double* d_y = cache.device_ptr(src.position, 0);
      for (const Vec4& q : src.partners) {
        const double* d_x = cache.device_ptr(q, 1);
        Vec4 const gx = wrap(q);
        Vec4 const yv = lattice.site_map_diff(src.position, q);

        double const t0 = MPI_Wtime();
        auto ks = contractor.kernel_sum(d_x, d_y, gx, yv, options.xunit);
        double const dt = MPI_Wtime() - t0;

        PairResult r;
        r.x = gx;
        r.y = gy;
        r.yv = yv;
        r.window = k;
        r.kernel_sum = ks;
        rows.push_back(r);
        local_stats.pairs++;
        local_stats.seconds_kernel_sum += dt;
        if (timing) timing->kernel_sum(k, gx, gy, dt);

        cache.pair_done(q, src.position);
      }
    }
    if (writer) writer->append(rows);
    results.insert(results.end(), rows.begin(), rows.end());
    if (options.verbose && rank == 0) {
      std::printf("# [hlbl] window %d: %zu pairs, cache %d (peak %d)\n", k,
                  rows.size(), cache.size(), cache.peak());
      std::fflush(stdout);
    }
  }

  if (cache.size() != 0) {
    throw std::logic_error("run_schedule: propagators left in the cache");
  }
  local_stats.cache_peak = cache.peak();
  if (stats) *stats = local_stats;
  return results;
}

namespace {

// Free memory of this node in bytes (0 if /proc/meminfo cannot be read)
std::size_t node_memory_available() {
  std::ifstream in("/proc/meminfo");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("MemAvailable:", 0) == 0) {
      std::istringstream s(line.substr(13));
      std::size_t kb = 0;
      s >> kb;
      return kb * 1024;
    }
  }
  return 0;
}

}  // namespace

MemoryEstimate estimate_memory(const Lattice& lattice, const Schedule& schedule,
                               PropagatorCache::Residency residency) {
  MemoryEstimate e;
  e.bytes_per_position = block_length(lattice) * sizeof(double);
  e.peak_positions = simulate(schedule, 2, Mode::kInterleaved).peak_cache;

  std::size_t free_dev = 0, total_dev = 0;
  check_cuda(cudaMemGetInfo(&free_dev, &total_dev), "cudaMemGetInfo");
  e.device_available = free_dev;

  MPI_Comm node = MPI_COMM_NULL;
  MPI_Comm_split_type(lattice.cart_comm(), MPI_COMM_TYPE_SHARED, 0,
                      MPI_INFO_NULL, &node);
  int ranks_on_node = 1;
  MPI_Comm_size(node, &ranks_on_node);
  MPI_Comm_free(&node);
  e.host_available = node_memory_available() / std::max(1, ranks_on_node);

  if (residency == PropagatorCache::Residency::kDevice) {
    e.device_needed = e.peak_positions * e.bytes_per_position;
    e.host_needed = 0;
  } else {
    e.device_needed = 3 * e.bytes_per_position;
    e.host_needed = e.peak_positions * e.bytes_per_position;
  }
  return e;
}

}  // namespace hlbl
