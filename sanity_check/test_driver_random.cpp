// Runs the schedule loop (run_schedule) on random propagators. Needs one GPU
// but no solver or gauge field. Checks that:
//  - the pair count and cache peak agree with simulate();
//  - results are the same with a device cache and a host cache;
//  - a pair equals a direct Contractor call on regenerated fields;
//  - two window ranges together reproduce the whole run;
//  - the HDF5 file holds exactly the computed rows.

#include "hlbl/driver.hpp"

#include <hdf5.h>
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

#include "check.hpp"
#include "hlbl/contract.hpp"
#include "hlbl/lattice.hpp"
#include "hlbl/provider.hpp"
#include "hlbl/schedule.hpp"

using namespace hlbl;

namespace {

// the order of the atomicAdd sums in the kernels changes from run to run
constexpr double kTol = 1e-10;

bool close(double a, double b) {
  return std::fabs(a - b) <= kTol * std::max(1.0, std::fabs(b));
}

using Key = std::pair<Vec4, Vec4>;

std::map<Key, PairResult> by_pair(const std::vector<PairResult>& rows) {
  std::map<Key, PairResult> m;
  for (const auto& r : rows) {
    HLBL_CHECK(m.emplace(Key{r.x, r.y}, r).second);  // no pair twice
  }
  return m;
}

void check_same(const std::vector<PairResult>& a,
                const std::vector<PairResult>& b) {
  auto ma = by_pair(a), mb = by_pair(b);
  HLBL_CHECK(ma.size() == mb.size());
  for (const auto& [key, ra] : ma) {
    auto it = mb.find(key);
    HLBL_CHECK(it != mb.end());
    HLBL_CHECK(it->second.window == ra.window);
    HLBL_CHECK(it->second.yv == ra.yv);
    for (int f = 0; f < 2; ++f) {
      HLBL_CHECK(close(ra.kernel_sum[f], it->second.kernel_sum[f]));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  HLBL_CHECK(size == 1);  // single-rank test

  Lattice lat(MPI_COMM_WORLD, Vec4{1, 1, 1, 1}, Vec4{6, 6, 6, 6});
  Vec4 const h{1, 1, 1, 1};
  int const stride = 1, steps = 2;
  HLBL_CHECK(max_steps_without_wrap(h, stride, lat.global()) >= steps);
  Schedule const schedule = build_schedule(h, stride, steps);
  check_complete(schedule, stride);
  auto const expect = simulate(schedule, 2, Mode::kInterleaved);

  QED_kernel_temps kqed_t;
  HLBL_CHECK(initialise(&kqed_t) == 0);
  Contractor contractor(lat, kqed_t);
  RandomProvider provider(lat, 7);

  // ---- whole run, device-resident cache, written to HDF5
  RunOptions opt;
  RunStats stats;
  std::vector<PairResult> dev;
  {
    RunInfo info;
    info.lattice = lat.global();
    info.procs = lat.procs();
    info.direction = h;
    info.stride = stride;
    info.window_end = steps + 1;
    ResultWriter writer("test_driver_random.h5", info);
    TimingLog timing("test_driver_random.csv");
    dev = run_schedule(lat, schedule, opt, provider, contractor, &writer,
                       &timing, &stats);
    HLBL_CHECK(writer.rows() == static_cast<long long>(dev.size()));
  }
  HLBL_CHECK(dev.size() == static_cast<std::size_t>(expect.n_pairs));
  HLBL_CHECK(stats.pairs == expect.n_pairs);
  HLBL_CHECK(stats.positions_solved == expect.n_propagators);
  HLBL_CHECK(stats.cache_peak == expect.peak_cache);
  for (const auto& r : dev) {
    HLBL_CHECK(std::isfinite(r.kernel_sum[0]) && std::isfinite(r.kernel_sum[1]));
    // separation of every pair is (+-n,+-n,+-n,+-n)
    for (int k = 0; k < 4; ++k) HLBL_CHECK(std::abs(r.yv[k]) == stride);
  }

  // ---- host-resident cache, staged: same numbers, same peak
  {
    RunOptions o = opt;
    o.residency = PropagatorCache::Residency::kHost;
    RunStats s;
    auto host = run_schedule(lat, schedule, o, provider, contractor, nullptr,
                             nullptr, &s);
    check_same(dev, host);
    HLBL_CHECK(s.cache_peak == expect.peak_cache);
  }

  // ---- a few pairs recomputed directly from freshly generated fields
  {
    std::size_t const n = block_length(lat);
    for (std::size_t i : {std::size_t{0}, dev.size() / 2, dev.size() - 1}) {
      const PairResult& r = dev[i];
      DeviceBuffer<double> d_x(n), d_y(n);
      provider.compute(r.x, d_x);
      provider.compute(r.y, d_y);
      auto direct = contractor.kernel_sum(d_x.get(), d_y.get(), r.x, r.yv,
                                          opt.xunit);
      for (int f = 0; f < 2; ++f) HLBL_CHECK(close(direct[f], r.kernel_sum[f]));
    }
  }

  // ---- window ranges: [0,1) then [1,3) reproduce the whole run
  {
    RunOptions a = opt, b = opt;
    a.window_begin = 0;
    a.window_end = 1;
    b.window_begin = 1;
    b.window_end = -1;
    RunStats sb;
    auto ra = run_schedule(lat, schedule, a, provider, contractor, nullptr,
                           nullptr);
    auto rb = run_schedule(lat, schedule, b, provider, contractor, nullptr,
                           nullptr, &sb);
    std::vector<PairResult> both = ra;
    both.insert(both.end(), rb.begin(), rb.end());
    check_same(dev, both);
    // the second range solved again the window 0 sources it needs
    HLBL_CHECK(sb.positions_solved > schedule[1].n_new() + schedule[2].n_new());
  }

  // ---- the HDF5 file holds what was computed
  {
    hid_t f = H5Fopen("test_driver_random.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    HLBL_CHECK(f >= 0);
    hid_t ds = H5Dopen2(f, "kernel_sum", H5P_DEFAULT);
    HLBL_CHECK(ds >= 0);
    hid_t sp = H5Dget_space(ds);
    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(sp, dims, nullptr);
    HLBL_CHECK(dims[0] == dev.size() && dims[1] == 2);
    std::vector<double> ks(2 * dev.size());
    HLBL_CHECK(H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       ks.data()) >= 0);
    for (std::size_t i = 0; i < dev.size(); ++i) {
      HLBL_CHECK(ks[2 * i] == dev[i].kernel_sum[0]);
      HLBL_CHECK(ks[2 * i + 1] == dev[i].kernel_sum[1]);
    }
    H5Sclose(sp);
    H5Dclose(ds);
    hid_t at = H5Aopen(f, "stride", H5P_DEFAULT);
    int stride_read = 0;
    H5Aread(at, H5T_NATIVE_INT, &stride_read);
    HLBL_CHECK(stride_read == stride);
    H5Aclose(at);
    H5Fclose(f);
  }

  free_QED_temps(&kqed_t);
  std::printf("test_driver_random: all checks passed (%zu pairs)\n", dev.size());
  MPI_Finalize();
  return 0;
}
