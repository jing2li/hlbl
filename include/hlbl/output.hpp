#pragma once

// Results and timings of a run, written by rank 0.

#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "hlbl/vec4.hpp"

namespace hlbl {

// One contracted pair.
struct PairResult {
  Vec4 x;   // kernel origin (the earlier source of the pair), global site
  Vec4 y;   // the other source, global site
  Vec4 yv;  // minimum-image separation y - x, in lattice units
  int window = 0;
  std::array<double, 2> kernel_sum{};  // one per flavour
};

// What identifies a run; stored as attributes of the result file.
struct RunInfo {
  Vec4 lattice{};    // global extents
  Vec4 procs{};
  Vec4 direction{};  // hypercube step direction h
  Vec4 origin{};
  int stride = 0;    // n; every pair has |y| = 2n
  int nconf = -1;    // gauge configuration number, -1 if none
  int window_begin = 0;
  int window_end = 0;  // half-open
  std::string kernel = "LLambda0.4";
};

// One HDF5 file per run, with extendable datasets, one row per pair:
//   x[N][4], y[N][4], yv[N][4]  int
//   window[N]                   int
//   kernel_sum[N][2]            double  (flavour 0, 1)
// and the RunInfo plus the source revision as attributes. Rows are flushed
// to disk on every append(), so a killed job keeps the windows it finished.
class ResultWriter {
 public:
  ResultWriter(const std::string& path, const RunInfo& info);
  ~ResultWriter();

  ResultWriter(const ResultWriter&) = delete;
  ResultWriter& operator=(const ResultWriter&) = delete;

  void append(const std::vector<PairResult>& rows);
  long long rows() const { return count_; }

 private:
  long long file_ = -1;  // hid_t
  long long ds_x_ = -1, ds_y_ = -1, ds_yv_ = -1, ds_window_ = -1,
            ds_kernel_ = -1;
  long long count_ = 0;
};

// One CSV row per solve and per kernel sum:
//   kind,window,x0..x3,y0..y3,seconds,iterations
// A solve has no y (left empty) and its iteration count; a kernel sum has
// both sites and no iteration count.
class TimingLog {
 public:
  explicit TimingLog(const std::string& path);
  ~TimingLog();

  TimingLog(const TimingLog&) = delete;
  TimingLog& operator=(const TimingLog&) = delete;

  void solve(int window, const Vec4& position, double seconds, int iterations);
  void kernel_sum(int window, const Vec4& x, const Vec4& y, double seconds);

 private:
  std::FILE* file_ = nullptr;
};

}  // namespace hlbl
