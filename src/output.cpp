#include "hlbl/output.hpp"

#include <hdf5.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace hlbl {

namespace {

void check(herr_t status, const char* what) {
  if (status < 0) {
    throw std::runtime_error(std::string("hlbl::ResultWriter: HDF5 ") + what +
                              " failed");
  }
}
hid_t check_id(hid_t id, const char* what) {
  if (id < 0) check(-1, what);
  return id;
}

// [N][row...], unlimited along N; row = {} gives one value per row. Chunks
// of at most 256 rows and about 1 MiB.
hid_t make_dataset(hid_t loc, const char* name, hid_t type,
                   const std::vector<hsize_t>& row) {
  int const rank = 1 + static_cast<int>(row.size());
  hsize_t row_bytes = H5Tget_size(type);
  for (hsize_t d : row) row_bytes *= d;
  hsize_t const chunk_rows =
      std::clamp<hsize_t>((hsize_t{1} << 20) / row_bytes, 1, 256);
  std::vector<hsize_t> dims{0}, maxdims{H5S_UNLIMITED}, chunk{chunk_rows};
  for (hsize_t d : row) {
    dims.push_back(d);
    maxdims.push_back(d);
    chunk.push_back(d);
  }
  hid_t space = check_id(H5Screate_simple(rank, dims.data(), maxdims.data()),
                         "create space");
  hid_t dcpl = check_id(H5Pcreate(H5P_DATASET_CREATE), "create dcpl");
  check(H5Pset_chunk(dcpl, rank, chunk.data()), "set chunk");
  hid_t ds = check_id(
      H5Dcreate2(loc, name, type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT),
      "create dataset");
  H5Pclose(dcpl);
  H5Sclose(space);
  return ds;
}

// appends n rows of shape `row` (as in make_dataset)
void append_rows(hid_t ds, hid_t memtype, const void* data, hsize_t first,
                 hsize_t n, const std::vector<hsize_t>& row) {
  int const rank = 1 + static_cast<int>(row.size());
  std::vector<hsize_t> total{first + n}, start(rank, 0), count{n};
  start[0] = first;
  for (hsize_t d : row) {
    total.push_back(d);
    count.push_back(d);
  }
  check(H5Dset_extent(ds, total.data()), "extend dataset");
  hid_t fspace = check_id(H5Dget_space(ds), "get space");
  check(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start.data(), nullptr,
                            count.data(), nullptr),
        "select rows");
  hid_t mspace = check_id(H5Screate_simple(rank, count.data(), nullptr),
                          "mem space");
  check(H5Dwrite(ds, memtype, mspace, fspace, H5P_DEFAULT, data), "write rows");
  H5Sclose(mspace);
  H5Sclose(fspace);
}

hid_t make_group(hid_t file, const char* name) {
  return check_id(H5Gcreate2(file, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                  "create group");
}

void attr_ints(hid_t file, const char* name, const int* v, hsize_t n) {
  hid_t space = check_id(H5Screate_simple(1, &n, nullptr), "attr space");
  hid_t a = check_id(
      H5Acreate2(file, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT),
      "create attribute");
  check(H5Awrite(a, H5T_NATIVE_INT, v), "write attribute");
  H5Aclose(a);
  H5Sclose(space);
}
void attr_int(hid_t file, const char* name, int v) { attr_ints(file, name, &v, 1); }

void attr_double(hid_t file, const char* name, double v) {
  hid_t space = check_id(H5Screate(H5S_SCALAR), "attr space");
  hid_t a = check_id(
      H5Acreate2(file, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT),
      "create attribute");
  check(H5Awrite(a, H5T_NATIVE_DOUBLE, &v), "write attribute");
  H5Aclose(a);
  H5Sclose(space);
}

// a small dataset [rows][cols] of ints, written at once
void int_table(hid_t file, const char* name, const std::vector<int>& data,
               hsize_t rows, hsize_t cols) {
  hsize_t dims[2] = {rows, cols};
  hid_t space = check_id(H5Screate_simple(2, dims, nullptr), "table space");
  hid_t ds = check_id(H5Dcreate2(file, name, H5T_NATIVE_INT, space, H5P_DEFAULT,
                                 H5P_DEFAULT, H5P_DEFAULT),
                      "create table");
  check(H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()),
        "write table");
  H5Dclose(ds);
  H5Sclose(space);
}

void attr_string(hid_t file, const char* name, const std::string& s) {
  hid_t type = H5Tcopy(H5T_C_S1);
  H5Tset_size(type, s.size() + 1);
  hid_t space = check_id(H5Screate(H5S_SCALAR), "attr space");
  hid_t a = check_id(
      H5Acreate2(file, name, type, space, H5P_DEFAULT, H5P_DEFAULT),
      "create attribute");
  check(H5Awrite(a, type, s.c_str()), "write attribute");
  H5Aclose(a);
  H5Sclose(space);
  H5Tclose(type);
}

}  // namespace

ResultWriter::ResultWriter(const std::string& path, const RunInfo& info) {
  if (info.chains.empty()) throw std::runtime_error("hlbl::ResultWriter: no chains");
  hid_t file = check_id(
      H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
      "create file");
  file_ = file;
  hid_t const ks = make_group(file, "kernel_sum");
  group_kernel_sum_ = ks;
  ds_x_ = make_dataset(ks, "x", H5T_NATIVE_INT, {4});
  ds_y_ = make_dataset(ks, "y", H5T_NATIVE_INT, {4});
  ds_yv_ = make_dataset(ks, "yv", H5T_NATIVE_INT, {4});
  ds_chain_ = make_dataset(ks, "chain", H5T_NATIVE_INT, {});
  ds_window_ = make_dataset(ks, "window", H5T_NATIVE_INT, {});
  ds_kernel_ = make_dataset(ks, "kernel_sum", H5T_NATIVE_DOUBLE, {2});

  lmax_ = *std::max_element(info.lattice.begin(), info.lattice.end());
  if (info.with_2p2) {
    hid_t const g = make_group(file, "2p2");
    group_2p2_ = g;
    hsize_t const lmax = static_cast<hsize_t>(lmax_);
    ds_w_ = make_dataset(g, "w", H5T_NATIVE_INT, {4});
    ds_chain_2p2_ = make_dataset(g, "chain", H5T_NATIVE_INT, {});
    ds_window_2p2_ = make_dataset(g, "window", H5T_NATIVE_INT, {});
    ds_p1_ = make_dataset(g, "P1", H5T_NATIVE_DOUBLE, {4, 4, 4, lmax});
    ds_pair_w_ = make_dataset(g, "pair_w", H5T_NATIVE_INT, {4});
    ds_pair_y_ = make_dataset(g, "pair_y", H5T_NATIVE_INT, {4});
    ds_pair_yv_ = make_dataset(g, "pair_yv", H5T_NATIVE_INT, {4});
    ds_p23_ = make_dataset(g, "P23", H5T_NATIVE_DOUBLE, {3, 4, 4, 4});
    attr_string(g, "rows",
                "w, chain, window, P1: one row per 2+2 source, at its first solve; "
                "pair_w, pair_y, pair_yv, P23: one row per (source, target)");
    attr_string(g, "P1_index",
                "P1[row][rho][sigma][nu][zeta]: Pi_{sigma nu}(z) summed over the "
                "sites z with (z - w)_rho mod L_rho = zeta; 0 for zeta >= L_rho");
    attr_string(g, "P23_index",
                "P23[row][geometry][rho][sigma][nu], geometry P2_0, P2_1, P3; "
                "only rho < sigma is filled");
    attr_string(g, "flavour",
                "operator 0 only (iflavor 0 of compute_2p2_gpu, as cvc): the "
                "two-point function is the same for both flavours");
  }

  constexpr hsize_t kChainCols = 10;
  std::vector<int> table;
  for (const Chain& c : info.chains) {
    table.insert(table.end(), c.start.begin(), c.start.end());
    table.insert(table.end(), c.direction.begin(), c.direction.end());
    table.push_back(c.stride);
    table.push_back(c.steps);
  }
  int_table(file, "chains", table, info.chains.size(), kChainCols);
  attr_string(file, "chains_columns",
              "start0..3, dir0..3, stride, steps; row = chain index");

  attr_ints(file, "lattice", info.lattice.data(), 4);
  attr_ints(file, "procs", info.procs.data(), 4);
  attr_int(file, "nconf", info.nconf);
  attr_double(file, "muon_mass_MeV", info.muon_mass);
  attr_double(file, "lattice_spacing_fm", info.lattice_spacing);
  attr_double(file, "xunit", info.xunit);
  if (info.chains.size() == 1) {
    // as in files from before chain lists
    const Chain& c = info.chains[0];
    attr_ints(file, "direction", c.direction.data(), 4);
    attr_ints(file, "origin", c.start.data(), 4);
    attr_int(file, "stride", c.stride);
  }
  attr_string(file, "kernel", info.kernel);
  attr_string(file, "flavour_order",
              "kernel_sum/kernel_sum[:,0] = operator 0, [:,1] = operator 1");
  attr_string(file, "coordinate_ranges",
              "x, y, w, pair_w, pair_y: [0, L); yv, pair_yv: minimum image of "
              "y - x, pair_y - pair_w in [-L/2, L/2)");
  check(H5Fflush(file, H5F_SCOPE_GLOBAL), "flush");
}

ResultWriter::~ResultWriter() {
  for (long long id : {ds_x_, ds_y_, ds_yv_, ds_chain_, ds_window_, ds_kernel_,
                       ds_w_, ds_chain_2p2_, ds_window_2p2_, ds_p1_, ds_pair_w_,
                       ds_pair_y_, ds_pair_yv_, ds_p23_}) {
    if (id >= 0) H5Dclose(static_cast<hid_t>(id));
  }
  for (long long id : {group_kernel_sum_, group_2p2_}) {
    if (id >= 0) H5Gclose(static_cast<hid_t>(id));
  }
  if (file_ >= 0) H5Fclose(static_cast<hid_t>(file_));
}

void ResultWriter::append(const std::vector<PairResult>& rows) {
  if (rows.empty()) return;
  std::size_t const n = rows.size();
  std::vector<int> x(4 * n), y(4 * n), yv(4 * n), chain(n), window(n);
  std::vector<double> kernel(2 * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (int k = 0; k < 4; ++k) {
      x[4 * i + k] = rows[i].x[k];
      y[4 * i + k] = rows[i].y[k];
      yv[4 * i + k] = rows[i].yv[k];
    }
    chain[i] = rows[i].chain;
    window[i] = rows[i].window;
    kernel[2 * i] = rows[i].kernel_sum[0];
    kernel[2 * i + 1] = rows[i].kernel_sum[1];
  }
  hsize_t const first = static_cast<hsize_t>(count_);
  append_rows(static_cast<hid_t>(ds_x_), H5T_NATIVE_INT, x.data(), first, n, {4});
  append_rows(static_cast<hid_t>(ds_y_), H5T_NATIVE_INT, y.data(), first, n, {4});
  append_rows(static_cast<hid_t>(ds_yv_), H5T_NATIVE_INT, yv.data(), first, n, {4});
  append_rows(static_cast<hid_t>(ds_chain_), H5T_NATIVE_INT, chain.data(), first,
              n, {});
  append_rows(static_cast<hid_t>(ds_window_), H5T_NATIVE_INT, window.data(),
              first, n, {});
  append_rows(static_cast<hid_t>(ds_kernel_), H5T_NATIVE_DOUBLE, kernel.data(),
              first, n, {2});
  count_ += static_cast<long long>(n);
  check(H5Fflush(static_cast<hid_t>(file_), H5F_SCOPE_GLOBAL), "flush");
}

void ResultWriter::append_2p2(const std::vector<Result2p2>& rows) {
  if (rows.empty()) return;
  if (group_2p2_ < 0) {
    throw std::logic_error("hlbl::ResultWriter: 2+2 rows, but RunInfo::with_2p2 is off");
  }
  hsize_t const lmax = static_cast<hsize_t>(lmax_);
  std::size_t const n = rows.size();
  std::vector<int> w, chain, window, pair_w, pair_y, pair_yv;
  std::vector<double> p1, p23;
  for (const Result2p2& r : rows) {
    if (r.p1.size() != 64 * lmax || r.y.size() != r.yv.size() ||
        r.p23.size() != r.y.size() * 3 * 64) {
      throw std::logic_error("hlbl::ResultWriter: 2+2 row of the wrong size");
    }
    w.insert(w.end(), r.w.begin(), r.w.end());
    chain.push_back(r.chain);
    window.push_back(r.window);
    p1.insert(p1.end(), r.p1.begin(), r.p1.end());
    for (std::size_t t = 0; t < r.y.size(); ++t) {
      pair_w.insert(pair_w.end(), r.w.begin(), r.w.end());
      pair_y.insert(pair_y.end(), r.y[t].begin(), r.y[t].end());
      pair_yv.insert(pair_yv.end(), r.yv[t].begin(), r.yv[t].end());
    }
    p23.insert(p23.end(), r.p23.begin(), r.p23.end());
  }
  hsize_t const first = static_cast<hsize_t>(count_2p2_);
  append_rows(static_cast<hid_t>(ds_w_), H5T_NATIVE_INT, w.data(), first, n, {4});
  append_rows(static_cast<hid_t>(ds_chain_2p2_), H5T_NATIVE_INT, chain.data(),
              first, n, {});
  append_rows(static_cast<hid_t>(ds_window_2p2_), H5T_NATIVE_INT, window.data(),
              first, n, {});
  append_rows(static_cast<hid_t>(ds_p1_), H5T_NATIVE_DOUBLE, p1.data(), first, n,
              {4, 4, 4, lmax});
  count_2p2_ += static_cast<long long>(n);

  hsize_t const n_pairs = pair_w.size() / 4;
  if (n_pairs > 0) {
    hsize_t const first_pair = static_cast<hsize_t>(count_pairs_2p2_);
    append_rows(static_cast<hid_t>(ds_pair_w_), H5T_NATIVE_INT, pair_w.data(),
                first_pair, n_pairs, {4});
    append_rows(static_cast<hid_t>(ds_pair_y_), H5T_NATIVE_INT, pair_y.data(),
                first_pair, n_pairs, {4});
    append_rows(static_cast<hid_t>(ds_pair_yv_), H5T_NATIVE_INT, pair_yv.data(),
                first_pair, n_pairs, {4});
    append_rows(static_cast<hid_t>(ds_p23_), H5T_NATIVE_DOUBLE, p23.data(),
                first_pair, n_pairs, {3, 4, 4, 4});
    count_pairs_2p2_ += static_cast<long long>(n_pairs);
  }
  check(H5Fflush(static_cast<hid_t>(file_), H5F_SCOPE_GLOBAL), "flush");
}

TimingLog::TimingLog(const std::string& path) {
  file_ = std::fopen(path.c_str(), "w");
  if (!file_) throw std::runtime_error("cannot open timing log " + path);
  std::fprintf(file_,
               "kind,chain,window,x0,x1,x2,x3,y0,y1,y2,y3,seconds,iterations\n");
  std::fflush(file_);
}

TimingLog::~TimingLog() {
  if (file_) std::fclose(file_);
}

void TimingLog::solve(int chain, int window, const Vec4& p, double seconds,
                      int iterations) {
  std::fprintf(file_, "solve,%d,%d,%d,%d,%d,%d,,,,,%.6f,%d\n", chain, window,
               p[0], p[1], p[2], p[3], seconds, iterations);
  std::fflush(file_);
}

void TimingLog::kernel_sum(int chain, int window, const Vec4& x, const Vec4& y,
                           double seconds) {
  std::fprintf(file_, "kernel_sum,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,\n", chain,
               window, x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3], seconds);
  std::fflush(file_);
}

void TimingLog::pieces_2p2(int chain, int window, const Vec4& w, double seconds) {
  std::fprintf(file_, "2p2,%d,%d,%d,%d,%d,%d,,,,,%.6f,\n", chain, window, w[0],
               w[1], w[2], w[3], seconds);
  std::fflush(file_);
}

}  // namespace hlbl
