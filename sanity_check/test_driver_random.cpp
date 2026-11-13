// The schedule loop on random propagators (one GPU, no solver). Checks pair
// count and cache peak against simulate(), device vs host cache, pairs and
// 2+2 pieces against direct Contractor calls, 2+2 only at the first solve of
// a site, two chains against independent runs, and the HDF5 files.

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

// all values of a double dataset, and its dimensions
std::vector<double> read_doubles(hid_t file, const char* name,
                                 std::vector<hsize_t>& dims) {
  hid_t ds = H5Dopen2(file, name, H5P_DEFAULT);
  HLBL_CHECK(ds >= 0);
  hid_t sp = H5Dget_space(ds);
  dims.assign(H5Sget_simple_extent_ndims(sp), 0);
  H5Sget_simple_extent_dims(sp, dims.data(), nullptr);
  hsize_t n = 1;
  for (hsize_t d : dims) n *= d;
  std::vector<double> v(n);
  HLBL_CHECK(H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     v.data()) >= 0);
  H5Sclose(sp);
  H5Dclose(ds);
  return v;
}

void check_close(const std::vector<double>& a, const std::vector<double>& b) {
  HLBL_CHECK(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) HLBL_CHECK(close(a[i], b[i]));
}

// the same 2+2 rows, in the same order
void check_same_2p2(const std::vector<Result2p2>& a,
                    const std::vector<Result2p2>& b) {
  HLBL_CHECK(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    HLBL_CHECK(a[i].w == b[i].w && a[i].chain == b[i].chain &&
               a[i].window == b[i].window);
    HLBL_CHECK(a[i].y == b[i].y && a[i].yv == b[i].yv);
    check_close(a[i].p1, b[i].p1);
    check_close(a[i].p23, b[i].p23);
  }
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

  // 2+2 sources: the centres of windows 0, 1 and 2, and (3,0,0,0), which this
  // chain does not solve (the start of the second chain below)
  std::vector<Source2p2> const sources_2p2 = {
      {Vec4{0, 0, 0, 0}, {Vec4{3, 3, 3, 3}, Vec4{0, 5, 1, 2}}, 1},
      {Vec4{1, 1, 1, 1}, {Vec4{0, 0, 0, 0}}, 2},
      {Vec4{2, 2, 2, 2}, {Vec4{5, 5, 5, 5}, Vec4{1, 1, 1, 1}, Vec4{3, 0, 4, 1}}, 3},
      {Vec4{3, 0, 0, 0}, {Vec4{0, 0, 0, 0}}, 4}};

  QED_kernel_temps kqed_t;
  HLBL_CHECK(initialise(&kqed_t) == 0);
  Contractor contractor(lat, kqed_t);
  RandomProvider provider(lat, 7);

  // ---- whole run, device-resident cache, written to HDF5
  RunOptions opt;
  opt.xunit = xunit(105.6583745, 0.07957);
  opt.sources_2p2 = sources_2p2;
  RunStats stats;
  std::vector<PairResult> dev;
  std::vector<Result2p2> dev_2p2;
  {
    RunInfo info;
    info.lattice = lat.global();
    info.procs = lat.procs();
    info.chains = {make_chain(Vec4{0, 0, 0, 0}, h, stride, steps)};
    info.muon_mass = 105.6583745;
    info.lattice_spacing = 0.07957;
    info.xunit = opt.xunit;
    info.with_2p2 = true;
    ResultWriter writer("test_driver_random.h5", info);
    TimingLog timing("test_driver_random.csv");
    dev = run_chains(lat, {schedule}, opt, provider, contractor, &writer,
                     &timing, &stats, &dev_2p2);
    HLBL_CHECK(writer.rows() == static_cast<long long>(dev.size()));
    HLBL_CHECK(writer.rows_2p2() == static_cast<long long>(dev_2p2.size()));
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

  // ---- 2+2 at the first solve of the three sources this chain solves (the
  //      centre of window k is new in window k)
  HLBL_CHECK(dev_2p2.size() == 3);
  HLBL_CHECK(stats.pieces_2p2 == 3 && stats.targets_2p2 == 6);
  for (int i = 0; i < 3; ++i) {
    const Result2p2& r = dev_2p2[i];
    HLBL_CHECK(r.w == sources_2p2[i].source && r.chain == 0 && r.window == i);
    HLBL_CHECK(r.y == sources_2p2[i].targets && r.yv.size() == r.y.size());
    for (std::size_t t = 0; t < r.y.size(); ++t) {
      HLBL_CHECK(r.yv[t] == lat.site_map_diff(r.y[t], r.w));
    }
    HLBL_CHECK(r.p1.size() == 64u * 6 && r.p23.size() == r.y.size() * 6 * 64);
  }
  // the same pieces directly from freshly generated fields
  for (const Result2p2& r : dev_2p2) {
    DeviceBuffer<double> d(block_length(lat));
    provider.compute(r.w, d);
    Pieces2p2 const direct = contractor.pieces_2p2(d.get(), r.w, r.y, opt.xunit);
    check_close(direct.p1, r.p1);
    check_close(direct.p23, r.p23);
  }

  // ---- host-resident cache, staged: same numbers, same peak
  {
    RunOptions o = opt;
    o.residency = PropagatorCache::Residency::kHost;
    RunStats s;
    std::vector<Result2p2> host_2p2;
    auto host = run_chains(lat, {schedule}, o, provider, contractor, nullptr,
                           nullptr, &s, &host_2p2);
    check_same(dev, host);
    check_same_2p2(dev_2p2, host_2p2);  // 2+2 from the host cache's scratch block
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
    std::vector<Result2p2> second_2p2;
    auto second = run_chains(lat, {schedule2}, opt, provider, contractor,
                             nullptr, nullptr, &s1, &second_2p2);
    for (auto& r : second) r.chain = 1;
    std::vector<PairResult> expected = dev;
    expected.insert(expected.end(), second.begin(), second.end());
    // the chains share no site: chain 1 alone solves (3,0,0,0)
    HLBL_CHECK(second_2p2.size() == 1 && (second_2p2[0].w == Vec4{3, 0, 0, 0}));
    for (auto& r : second_2p2) r.chain = 1;
    std::vector<Result2p2> expected_2p2 = dev_2p2;
    expected_2p2.insert(expected_2p2.end(), second_2p2.begin(), second_2p2.end());

    std::vector<PairResult> rows;
    std::vector<Result2p2> rows_2p2;
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
      info.with_2p2 = true;
      {
        ResultWriter writer("test_driver_random_multi.h5", info);
        rows = run_chains(lat, chains, o, provider, contractor, &writer, nullptr,
                          &s, &rows_2p2);
      }
      check_same(expected, rows);
      check_same_2p2(expected_2p2, rows_2p2);
      HLBL_CHECK(s.pieces_2p2 == stats.pieces_2p2 + s1.pieces_2p2);
      HLBL_CHECK(s.pairs == stats.pairs + s1.pairs);
      HLBL_CHECK(s.positions_solved == stats.positions_solved + s1.positions_solved);
      HLBL_CHECK(s.cache_peak == std::max(stats.cache_peak, s1.cache_peak));
    }

    // the file written last: chain column, chains table, no single-chain attributes
    hid_t f = H5Fopen("test_driver_random_multi.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    HLBL_CHECK(f >= 0);
    std::vector<hsize_t> dims;
    auto chain_col = read_ints(f, "kernel_sum/chain", dims);
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
    // the 2+2 rows of both chains
    auto chain_2p2 = read_ints(f, "2p2/chain", dims);
    HLBL_CHECK(dims.size() == 1 && dims[0] == rows_2p2.size());
    for (std::size_t i = 0; i < rows_2p2.size(); ++i) {
      HLBL_CHECK(chain_2p2[i] == rows_2p2[i].chain);
    }
    H5Fclose(f);
  }

  // ---- overlapping chains: a second chain from the centre of window 1 (the
  //      README's resume) solves only sites the first chain solved, so it
  //      computes no 2+2; the run's 2+2 is that of the first chain alone
  {
    Schedule const resumed = build_schedule(h, stride, 1, Vec4{1, 1, 1, 1});
    RunStats s;
    std::vector<Result2p2> both_2p2;
    run_chains(lat, {schedule, resumed}, opt, provider, contractor, nullptr,
               nullptr, &s, &both_2p2);
    check_same_2p2(dev_2p2, both_2p2);
    HLBL_CHECK(s.pieces_2p2 == stats.pieces_2p2);
  }

  // ---- the HDF5 file holds what was computed
  {
    hid_t f = H5Fopen("test_driver_random.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    HLBL_CHECK(f >= 0);
    {
      // one chain: all rows are chain 0, and the attributes of old files are there
      std::vector<hsize_t> cdims;
      auto chain_col = read_ints(f, "kernel_sum/chain", cdims);
      HLBL_CHECK(cdims.size() == 1 && cdims[0] == dev.size());
      for (int c : chain_col) HLBL_CHECK(c == 0);
      auto table = read_ints(f, "chains", cdims);
      HLBL_CHECK(cdims.size() == 2 && cdims[0] == 1 && cdims[1] == 10);
      HLBL_CHECK(table[8] == stride && table[9] == steps);
      HLBL_CHECK(H5Aexists(f, "direction") > 0 && H5Aexists(f, "origin") > 0);
    }
    hid_t ds = H5Dopen2(f, "kernel_sum/kernel_sum", H5P_DEFAULT);
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

    // the 2+2 group: one row per source, one per (source, target)
    std::vector<hsize_t> d;
    auto w = read_ints(f, "2p2/w", d);
    HLBL_CHECK(d.size() == 2 && d[0] == dev_2p2.size() && d[1] == 4);
    auto window_2p2 = read_ints(f, "2p2/window", d);
    auto p1 = read_doubles(f, "2p2/P1", d);
    HLBL_CHECK(d.size() == 5 && d[0] == dev_2p2.size() && d[1] == 4 && d[4] == 6);
    auto pair_w = read_ints(f, "2p2/pair_w", d);
    auto pair_y = read_ints(f, "2p2/pair_y", d);
    auto pair_yv = read_ints(f, "2p2/pair_yv", d);
    HLBL_CHECK(d.size() == 2 && d[0] == static_cast<hsize_t>(stats.targets_2p2));
    auto p23 = read_doubles(f, "2p2/P23", d);
    HLBL_CHECK(d.size() == 5 && d[1] == 6 && d[4] == 4);
    std::size_t row = 0;
    for (std::size_t i = 0; i < dev_2p2.size(); ++i) {
      const Result2p2& r = dev_2p2[i];
      HLBL_CHECK(window_2p2[i] == r.window);
      for (int k = 0; k < 4; ++k) HLBL_CHECK(w[4 * i + k] == r.w[k]);
      for (std::size_t j = 0; j < r.p1.size(); ++j) {
        HLBL_CHECK(p1[i * r.p1.size() + j] == r.p1[j]);
      }
      for (std::size_t t = 0; t < r.y.size(); ++t, ++row) {
        for (int k = 0; k < 4; ++k) {
          HLBL_CHECK(pair_w[4 * row + k] == r.w[k]);
          HLBL_CHECK(pair_y[4 * row + k] == r.y[t][k]);
          HLBL_CHECK(pair_yv[4 * row + k] == r.yv[t][k]);
        }
        for (int j = 0; j < 6 * 64; ++j) {
          HLBL_CHECK(p23[row * 6 * 64 + j] == r.p23[t * 6 * 64 + j]);
        }
      }
    }
    H5Fclose(f);
  }

  free_QED_temps(&kqed_t);
  std::printf("test_driver_random: all checks passed (%zu pairs, %zu 2+2 sources)\n",
              dev.size(), dev_2p2.size());
  MPI_Finalize();
  return 0;
}
