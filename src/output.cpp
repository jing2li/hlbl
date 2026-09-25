#include "hlbl/output.hpp"

#include <hdf5.h>

#include <cstring>
#include <stdexcept>

#ifndef HLBL_GIT_REVISION
#define HLBL_GIT_REVISION "unknown"
#endif

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

// rank 1: [N] (cols == 0), rank 2: [N][cols]; unlimited along N
hid_t make_dataset(hid_t file, const char* name, hid_t type, hsize_t cols) {
  int const rank = cols == 0 ? 1 : 2;
  hsize_t dims[2] = {0, cols};
  hsize_t maxdims[2] = {H5S_UNLIMITED, cols};
  hsize_t chunk[2] = {256, cols};
  hid_t space = check_id(H5Screate_simple(rank, dims, maxdims), "create space");
  hid_t dcpl = check_id(H5Pcreate(H5P_DATASET_CREATE), "create dcpl");
  check(H5Pset_chunk(dcpl, rank, chunk), "set chunk");
  hid_t ds = check_id(
      H5Dcreate2(file, name, type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT),
      "create dataset");
  H5Pclose(dcpl);
  H5Sclose(space);
  return ds;
}

// appends n rows of `cols` values (cols == 0: one value per row)
void append_rows(hid_t ds, hid_t memtype, const void* data, hsize_t first,
                 hsize_t n, hsize_t cols) {
  int const rank = cols == 0 ? 1 : 2;
  hsize_t total[2] = {first + n, cols};
  check(H5Dset_extent(ds, total), "extend dataset");
  hid_t fspace = check_id(H5Dget_space(ds), "get space");
  hsize_t start[2] = {first, 0};
  hsize_t count[2] = {n, cols};
  check(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, nullptr, count,
                            nullptr),
        "select rows");
  hid_t mspace = check_id(H5Screate_simple(rank, count, nullptr), "mem space");
  check(H5Dwrite(ds, memtype, mspace, fspace, H5P_DEFAULT, data), "write rows");
  H5Sclose(mspace);
  H5Sclose(fspace);
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
  hid_t file = check_id(
      H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
      "create file");
  file_ = file;
  ds_x_ = make_dataset(file, "x", H5T_NATIVE_INT, 4);
  ds_y_ = make_dataset(file, "y", H5T_NATIVE_INT, 4);
  ds_yv_ = make_dataset(file, "yv", H5T_NATIVE_INT, 4);
  ds_window_ = make_dataset(file, "window", H5T_NATIVE_INT, 0);
  ds_kernel_ = make_dataset(file, "kernel_sum", H5T_NATIVE_DOUBLE, 2);

  attr_ints(file, "lattice", info.lattice.data(), 4);
  attr_ints(file, "procs", info.procs.data(), 4);
  attr_ints(file, "direction", info.direction.data(), 4);
  attr_ints(file, "origin", info.origin.data(), 4);
  attr_int(file, "stride", info.stride);
  attr_int(file, "nconf", info.nconf);
  attr_int(file, "window_begin", info.window_begin);
  attr_int(file, "window_end", info.window_end);
  attr_string(file, "kernel", info.kernel);
  attr_string(file, "hlbl_revision", HLBL_GIT_REVISION);
  attr_string(file, "flavour_order", "kernel_sum[:,0] = operator 0, [:,1] = operator 1");
  check(H5Fflush(file, H5F_SCOPE_GLOBAL), "flush");
}

ResultWriter::~ResultWriter() {
  for (long long id : {ds_x_, ds_y_, ds_yv_, ds_window_, ds_kernel_}) {
    if (id >= 0) H5Dclose(static_cast<hid_t>(id));
  }
  if (file_ >= 0) H5Fclose(static_cast<hid_t>(file_));
}

void ResultWriter::append(const std::vector<PairResult>& rows) {
  if (rows.empty()) return;
  std::size_t const n = rows.size();
  std::vector<int> x(4 * n), y(4 * n), yv(4 * n), window(n);
  std::vector<double> kernel(2 * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (int k = 0; k < 4; ++k) {
      x[4 * i + k] = rows[i].x[k];
      y[4 * i + k] = rows[i].y[k];
      yv[4 * i + k] = rows[i].yv[k];
    }
    window[i] = rows[i].window;
    kernel[2 * i] = rows[i].kernel_sum[0];
    kernel[2 * i + 1] = rows[i].kernel_sum[1];
  }
  hsize_t const first = static_cast<hsize_t>(count_);
  append_rows(static_cast<hid_t>(ds_x_), H5T_NATIVE_INT, x.data(), first, n, 4);
  append_rows(static_cast<hid_t>(ds_y_), H5T_NATIVE_INT, y.data(), first, n, 4);
  append_rows(static_cast<hid_t>(ds_yv_), H5T_NATIVE_INT, yv.data(), first, n, 4);
  append_rows(static_cast<hid_t>(ds_window_), H5T_NATIVE_INT, window.data(),
              first, n, 0);
  append_rows(static_cast<hid_t>(ds_kernel_), H5T_NATIVE_DOUBLE, kernel.data(),
              first, n, 2);
  count_ += static_cast<long long>(n);
  check(H5Fflush(static_cast<hid_t>(file_), H5F_SCOPE_GLOBAL), "flush");
}

TimingLog::TimingLog(const std::string& path) {
  file_ = std::fopen(path.c_str(), "w");
  if (!file_) throw std::runtime_error("cannot open timing log " + path);
  std::fprintf(file_, "kind,window,x0,x1,x2,x3,y0,y1,y2,y3,seconds,iterations\n");
  std::fflush(file_);
}

TimingLog::~TimingLog() {
  if (file_) std::fclose(file_);
}

void TimingLog::solve(int window, const Vec4& p, double seconds, int iterations) {
  std::fprintf(file_, "solve,%d,%d,%d,%d,%d,,,,,%.6f,%d\n", window, p[0], p[1],
               p[2], p[3], seconds, iterations);
  std::fflush(file_);
}

void TimingLog::kernel_sum(int window, const Vec4& x, const Vec4& y,
                           double seconds) {
  std::fprintf(file_, "kernel_sum,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,\n", window,
               x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3], seconds);
  std::fflush(file_);
}

}  // namespace hlbl
