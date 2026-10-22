// The schedule loop on random propagators (one GPU, no solver). Checks pair
// count and cache peak against simulate(), device vs host cache, pairs
// against direct Contractor calls, two chains against independent runs, and
// the HDF5 files.

#include "hlbl/driver.hpp"

#include <hdf5.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <tuple>
#include <vector>

#include "check.hpp"
#include "hlbl/contract.hpp"
#include "hlbl/lattice.hpp"
#include "hlbl/physics.hpp"
#include "hlbl/provider.hpp"
#include "hlbl/schedule.hpp"

using namespace hlbl;

namespace {

// the order of the atomicAdd sums in the kernels changes from run to run
constexpr double kTol = 1e-10;

bool close(double a, double b) {
  return std::fabs(a - b) <= kTol * std::max(1.0, std::fabs(b));
}

using Key = std::tuple<int, Vec4, Vec4>;  // chain, x, y

std::map<Key, PairResult> by_pair(const std::vector<PairResult>& rows) {
  std::map<Key, PairResult> m;
  for (const auto& r : rows) {
    HLBL_CHECK(m.emplace(Key{r.chain, r.x, r.y}, r).second);  // no pair twice
  }
  return m;
}

Chain make_chain(const Vec4& start, const Vec4& dir, int stride, int steps) {
  Chain c;
  c.start = start;
  c.direction = dir;
  c.stride = stride;
  c.steps = steps;
  return c;
}

// all values of an int dataset, and its dimensions
std::vector<int> read_ints(hid_t file, const char* name,
                           std::vector<hsize_t>& dims) {
  hid_t ds = H5Dopen2(file, name, H5P_DEFAULT);
  HLBL_CHECK(ds >= 0);
  hid_t sp = H5Dget_space(ds);
  dims.assign(H5Sget_simple_extent_ndims(sp), 0);
  H5Sget_simple_extent_dims(sp, dims.data(), nullptr);
  hsize_t n = 1;
  for (hsize_t d : dims) n *= d;
  std::vector<int> v(n);
  HLBL_CHECK(H5Dread(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     v.data()) >= 0);
  H5Sclose(sp);
  H5Dclose(ds);
  return v;
}

void check_same(const std::vector<PairResult>& a,
                const std::vector<PairResult>& b) {
  auto ma = by_pair(a), mb = by_pair(b);
  HLBL_CHECK(ma.size() == mb.size());
  for (const auto& [key, ra] : ma) {
    auto it = mb.find(key);
    HLBL_CHECK(it != mb.end());
    HLBL_CHECK(it->second.chain == ra.chain);
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
  auto const expect = simulate(schedule, Mode::kInterleaved);

  QED_kernel_temps kqed_t;
  HLBL_CHECK(initialise(&kqed_t) == 0);
  Contractor contractor(lat, kqed_t);
  RandomProvider provider(lat, 7);

  // ---- whole run, device-resident cache, written to HDF5
  RunOptions opt;
  opt.xunit = xunit(105.6583745, 0.07957);
  RunStats stats;
  std::vector<PairResult> dev;
  {
    RunInfo info;
    info.lattice = lat.global();
    info.procs = lat.procs();
    info.chains = {make_chain(Vec4{0, 0, 0, 0}, h, stride, steps)};
    info.muon_mass = 105.6583745;
    info.lattice_spacing = 0.07957;
    info.xunit = opt.xunit;
    ResultWriter writer("test_driver_random.h5", info);
    TimingLog timing("test_driver_random.csv");
    dev = run_chains(lat, {schedule}, opt, provider, contractor, &writer,
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
    auto host = run_chains(lat, {schedule}, o, provider, contractor, nullptr,
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

  // ---- two chains: the same pairs as two independent runs, with either
  //      cache, and the right HDF5 file
  {
    Vec4 const h2{1, -1, 1, 1}, start2{3, 0, 0, 0};
    Schedule const schedule2 = build_schedule(h2, stride, 2, start2);
    check_complete(schedule2, stride);
    std::vector<Schedule> const chains = {schedule, schedule2};

    RunStats s1;
    auto second = run_chains(lat, {schedule2}, opt, provider, contractor,
                             nullptr, nullptr, &s1);
    for (auto& r : second) r.chain = 1;
    std::vector<PairResult> expected = dev;
    expected.insert(expected.end(), second.begin(), second.end());

    std::vector<PairResult> rows;
    for (auto residency : {PropagatorCache::Residency::kDevice,
                           PropagatorCache::Residency::kHost}) {
      RunOptions o = opt;
      o.residency = residency;
      RunStats s;
      RunInfo info;
      info.lattice = lat.global();
      info.procs = lat.procs();
      info.chains = {make_chain(Vec4{0, 0, 0, 0}, h, stride, steps),
                     make_chain(start2, h2, stride, 2)};
      {
        ResultWriter writer("test_driver_random_multi.h5", info);
        rows = run_chains(lat, chains, o, provider, contractor, &writer, nullptr,
                          &s);
      }
      check_same(expected, rows);
      HLBL_CHECK(s.pairs == stats.pairs + s1.pairs);
      HLBL_CHECK(s.positions_solved == stats.positions_solved + s1.positions_solved);
      HLBL_CHECK(s.cache_peak == std::max(stats.cache_peak, s1.cache_peak));
    }

    // the file written last: chain column, chains table, no single-chain attributes
    hid_t f = H5Fopen("test_driver_random_multi.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    HLBL_CHECK(f >= 0);
    std::vector<hsize_t> dims;
    auto chain_col = read_ints(f, "chain", dims);
    HLBL_CHECK(dims.size() == 1 && dims[0] == rows.size());
    std::size_t n_second = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
      HLBL_CHECK(chain_col[i] == rows[i].chain);
      n_second += chain_col[i] == 1;
    }
    HLBL_CHECK(n_second == second.size());
    auto table = read_ints(f, "chains", dims);
    HLBL_CHECK(dims.size() == 2 && dims[0] == 2 && dims[1] == 10);
    int const row0[10] = {0, 0, 0, 0, 1, 1, 1, 1, stride, steps};
    int const row1[10] = {3, 0, 0, 0, 1, -1, 1, 1, stride, 2};
    for (int k = 0; k < 10; ++k) {
      HLBL_CHECK(table[k] == row0[k]);
      HLBL_CHECK(table[10 + k] == row1[k]);
    }
    HLBL_CHECK(H5Aexists(f, "direction") == 0);
    H5Fclose(f);
  }

  // ---- the HDF5 file holds what was computed
  {
    hid_t f = H5Fopen("test_driver_random.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    HLBL_CHECK(f >= 0);
    {
      // one chain: all rows are chain 0, and the attributes of old files are there
      std::vector<hsize_t> cdims;
      auto chain_col = read_ints(f, "chain", cdims);
      HLBL_CHECK(cdims.size() == 1 && cdims[0] == dev.size());
      for (int c : chain_col) HLBL_CHECK(c == 0);
      auto table = read_ints(f, "chains", cdims);
      HLBL_CHECK(cdims.size() == 2 && cdims[0] == 1 && cdims[1] == 10);
      HLBL_CHECK(table[8] == stride && table[9] == steps);
      HLBL_CHECK(H5Aexists(f, "direction") > 0 && H5Aexists(f, "origin") > 0);
    }
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
    // the physics inputs of the run
    at = H5Aopen(f, "xunit", H5P_DEFAULT);
    double xunit_read = 0;
    H5Aread(at, H5T_NATIVE_DOUBLE, &xunit_read);
    HLBL_CHECK(xunit_read == opt.xunit);
    H5Aclose(at);
    HLBL_CHECK(H5Aexists(f, "muon_mass_MeV") > 0 && H5Aexists(f, "lattice_spacing_fm") > 0);
    H5Fclose(f);
  }

  free_QED_temps(&kqed_t);
  std::printf("test_driver_random: all checks passed (%zu pairs)\n", dev.size());
  MPI_Finalize();
  return 0;
}
