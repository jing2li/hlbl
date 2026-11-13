#include "hlbl/driver.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace hlbl {

namespace {

// One chain, using `cache`, which must be empty. Appends the results to
// `results` (and the 2+2 results to `results_2p2` if not null) and returns
// the chain's own statistics. `at_2p2`: the chain's solves followed by 2+2
// (Plan2p2::at), position -> index into options.sources_2p2.
RunStats run_chain(int chain, int n_chains, const Schedule& schedule,
                   const Lattice& lattice, const RunOptions& options,
                   PropagatorProvider& provider, Contractor& contractor,
                   ResultWriter* writer, TimingLog* timing,
                   PropagatorCache& cache, std::vector<PairResult>& results,
                   const std::map<Vec4, int>& at_2p2,
                   std::vector<Result2p2>* results_2p2) {
  int const n_windows = static_cast<int>(schedule.size());
  Vec4 const global = lattice.global();
  auto wrap = [&](const Vec4& p) { return mod_pos(p, global); };

  std::map<Vec4, int> pending;  // pairs per position
  for (const auto& [older, newer] : all_pairs(schedule)) {
    pending[older]++;
    pending[newer]++;
  }
  cache.reset(std::move(pending));

  RunStats local_stats;
  int rank = 0;
  MPI_Comm_rank(lattice.cart_comm(), &rank);
  if (rank == 0) {
    std::printf("# [hlbl] chain %d of %d: %d windows\n", chain, n_chains,
                n_windows);
    std::fflush(stdout);
  }
  double const chain_t0 = MPI_Wtime();
  bool const with_2p2 = !options.sources_2p2.empty();
  std::vector<Result2p2> rows_2p2;  // of the current window
  auto progress = [&](int w, const Vec4& p, const std::string& s) {  // stderr: unbuffered
    if (rank == 0) std::fprintf(stderr, "# [hlbl] chain %d window %d "
        "at (%d,%d,%d,%d): %s\n", chain, w, p[0], p[1], p[2], p[3], s.c_str());
  };

  // the 2+2 pieces of `site` from its block `d_fwd`
  auto pieces_2p2 = [&](int window, const Vec4& site, const double* d_fwd,
                        const Source2p2& source) {
    double const t0 = MPI_Wtime();
    Pieces2p2 pieces =
        contractor.pieces_2p2(d_fwd, site, source.targets, options.xunit);
    double const dt = MPI_Wtime() - t0;
    Result2p2 r;
    r.w = site;
    r.chain = chain;
    r.window = window;
    r.p1 = std::move(pieces.p1);
    r.y = source.targets;
    for (const Vec4& y : source.targets) r.yv.push_back(lattice.site_map_diff(y, site));
    r.p23 = std::move(pieces.p23);
    rows_2p2.push_back(std::move(r));
    local_stats.pieces_2p2++;
    local_stats.targets_2p2 += static_cast<long long>(source.targets.size());
    local_stats.seconds_2p2 += dt;
    if (timing) timing->pieces_2p2(chain, window, site, dt);
    progress(window, site, "2+2 done, " + std::to_string(source.targets.size()) + " targets");
  };

  auto solve = [&](int window, const Vec4& position) {
    Vec4 const site = wrap(position);
    DeviceBuffer<double>& dest = cache.begin_insert(position);
    double const t0 = MPI_Wtime();
    int const iterations = provider.compute(site, dest);
    double const dt = MPI_Wtime() - t0;
    local_stats.positions_solved++;
    local_stats.iterations += iterations;
    local_stats.seconds_solve += dt;
    if (timing) timing->solve(chain, window, site, dt, iterations);
    progress(window, site, "inversion done");
    // 2+2 before the block is stored: `dest` is the solve target (the cache
    // entry, or the host cache's scratch block), so no staging copy
    auto const it = at_2p2.find(position);
    if (it != at_2p2.end()) {
      pieces_2p2(window, site, dest.get(), options.sources_2p2[it->second]);
    }
    cache.end_insert(position);
  };

  for (int k = 0; k < n_windows; ++k) {
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
        r.chain = chain;
        r.window = k;
        r.kernel_sum = ks;
        rows.push_back(r);
        local_stats.pairs++;
        local_stats.seconds_kernel_sum += dt;
        if (timing) timing->kernel_sum(chain, k, gx, gy, dt);

        cache.pair_done(q, src.position);
      }
      progress(k, gy, "4pt done, " + std::to_string(src.partners.size()) + " pairs");
    }
    if (writer) {
      writer->append(rows);
      writer->append_2p2(rows_2p2);
    }
    results.insert(results.end(), rows.begin(), rows.end());
    if (rank == 0) {
      std::printf("# [hlbl] chain %d window %d: %zu pairs, ", chain, k, rows.size());
      if (with_2p2) std::printf("2+2 at %zu sources, ", rows_2p2.size());
      std::printf("cache %d (peak %d)\n", cache.size(), cache.peak());
      std::fflush(stdout);
    }
    if (results_2p2) {
      results_2p2->insert(results_2p2->end(), std::make_move_iterator(rows_2p2.begin()),
                          std::make_move_iterator(rows_2p2.end()));
    }
    rows_2p2.clear();
  }

  if (cache.size() != 0) {
    throw std::logic_error("run_chains: propagators left in the cache");
  }
  local_stats.cache_peak = cache.peak();
  if (rank == 0) {
    std::printf("# [hlbl] chain %d done: %d pairs, %d positions solved, ", chain,
                local_stats.pairs, local_stats.positions_solved);
    if (with_2p2) std::printf("2+2 at %d sources, ", local_stats.pieces_2p2);
    std::printf("%.1f s\n", MPI_Wtime() - chain_t0);
    std::fflush(stdout);
  }
  return local_stats;
}

}  // namespace

std::vector<PairResult> run_chains(const Lattice& lattice,
                                    const std::vector<Schedule>& chains,
                                    const RunOptions& options,
                                    PropagatorProvider& provider,
                                    Contractor& contractor,
                                    ResultWriter* writer, TimingLog* timing,
                                    RunStats* stats,
                                    std::vector<Result2p2>* results_2p2) {
  if (chains.empty()) throw std::runtime_error("run_chains: no chains");
  if (!(options.xunit > 0)) {
    throw std::runtime_error("run_chains: RunOptions::xunit (m_mu a) is not set");
  }
  // one cache for all chains: its pinned and device buffers are reused
  PropagatorCache cache(options.residency, block_length(lattice));
  // which solves are followed by 2+2: the first solve of each 2+2 source
  std::vector<Vec4> sources;
  for (const Source2p2& g : options.sources_2p2) sources.push_back(g.source);
  Plan2p2 const plan = plan_2p2(chains, sources, lattice.global());
  if (results_2p2) results_2p2->clear();

  RunStats total;
  std::vector<PairResult> results;
  int const n_chains = static_cast<int>(chains.size());
  for (int c = 0; c < n_chains; ++c) {
    RunStats s = run_chain(c, n_chains, chains[c], lattice, options, provider,
                           contractor, writer, timing, cache, results,
                           plan.at[c], results_2p2);
    total.positions_solved += s.positions_solved;
    total.iterations += s.iterations;
    total.pairs += s.pairs;
    total.cache_peak = std::max(total.cache_peak, s.cache_peak);
    total.seconds_solve += s.seconds_solve;
    total.seconds_kernel_sum += s.seconds_kernel_sum;
    total.pieces_2p2 += s.pieces_2p2;
    total.targets_2p2 += s.targets_2p2;
    total.seconds_2p2 += s.seconds_2p2;
  }
  if (stats) *stats = total;
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

MemoryEstimate estimate_memory(const Lattice& lattice, int peak_positions,
                               PropagatorCache::Residency residency,
                               int max_targets_2p2) {
  MemoryEstimate e;
  e.bytes_per_position = block_length(lattice) * sizeof(double);
  e.peak_positions = peak_positions;
  if (max_targets_2p2 > 0) {
    // compute_2p2_gpu: Pi (16 doubles per site), P1 (64 Lmax), P23 (6 x 64
    // per target: both copies) and the target list
    const Vec4& g = lattice.global();
    std::size_t const lmax = *std::max_element(g.begin(), g.end());
    std::size_t const n = static_cast<std::size_t>(max_targets_2p2);
    e.bytes_2p2 = (16 * lattice.volume() + 64 * lmax + 6 * 64 * n) * sizeof(double) +
                  4 * n * sizeof(int);
  }

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
  e.device_needed += e.bytes_2p2;
  return e;
}

}  // namespace hlbl
