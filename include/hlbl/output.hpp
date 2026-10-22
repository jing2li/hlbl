#pragma once

// Result file (HDF5) and timing log (CSV). Rank 0 writes both.

#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "hlbl/input.hpp"
#include "hlbl/vec4.hpp"

namespace hlbl {

// One contracted pair.
struct PairResult {
  Vec4 x;   // kernel origin (the earlier source of the pair), global site
  Vec4 y;   // the other source, global site
  Vec4 yv;  // minimum-image separation y - x, in lattice units
  int chain = 0;   // index of the chain in the input
  int window = 0;  // window within that chain
  std::array<double, 2> kernel_sum{};  // one per flavour
};

// What identifies a run; stored in the result file.
struct RunInfo {
  Vec4 lattice{};    // global extents
  Vec4 procs{};
  int nconf = -1;    // gauge configuration number (-1: none, random fields)
  std::vector<Chain> chains;   // as run: steps resolved
  std::string kernel = "LLambda0.4";
  double muon_mass = 0;        // MeV
  double lattice_spacing = 0;  // fm
  double xunit = 0;            // m_mu a, the unit of the kernel arguments
};

// One HDF5 file per run (format in the README), one row per pair:
//   x[N][4], y[N][4], yv[N][4]  int
//   chain[N], window[N]         int
//   kernel_sum[N][2]            double  (flavour 0, 1)
// plus the table chains[C][10], and RunInfo and the git revision as
// attributes. Each append() is flushed, so a killed job keeps its finished
// windows.
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
  long long ds_x_ = -1, ds_y_ = -1, ds_yv_ = -1, ds_chain_ = -1,
            ds_window_ = -1, ds_kernel_ = -1;
  long long count_ = 0;
};

// One CSV row per solve and per kernel sum:
//   kind,chain,window,x0..x3,y0..y3,seconds,iterations
// A solve leaves y empty; a kernel sum leaves iterations empty.
class TimingLog {
 public:
  explicit TimingLog(const std::string& path);
  ~TimingLog();

  TimingLog(const TimingLog&) = delete;
  TimingLog& operator=(const TimingLog&) = delete;

  void solve(int chain, int window, const Vec4& position, double seconds,
             int iterations);
  void kernel_sum(int chain, int window, const Vec4& x, const Vec4& y,
                  double seconds);

 private:
  std::FILE* file_ = nullptr;
};

}  // namespace hlbl
