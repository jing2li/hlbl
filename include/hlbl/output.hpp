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
  Vec4 x;   // kernel origin (the earlier source of the pair), global site in [0, L)
  Vec4 y;   // the other source, global site in [0, L)
  Vec4 yv;  // y - x, minimum image in [-L/2, L/2) (site_map_diff)
  int chain = 0;   // index of the chain in the input
  int window = 0;  // window within that chain
  std::array<double, 2> kernel_sum{};  // one per flavour
};

// The 2+2 pieces of one source (Contractor::pieces_2p2).
struct Result2p2 {
  Vec4 w{};        // the source, global site in [0, L)
  int chain = 0;   // chain and window of its first solve
  int window = 0;
  std::vector<double> p1;   // [4][4][4][Lmax]
  std::vector<Vec4> y;      // the targets, global sites in [0, L)
  std::vector<Vec4> yv;     // y - w, minimum image in [-L/2, L/2)
  std::vector<double> p23;  // [targets][3][4][4][4]: P2_0, P2_1, P3
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
  bool with_2p2 = false;       // the run has 2p2 lines: write the 2p2 group
};

// One HDF5 file per run (format in the README):
//   kernel_sum/   one row per pair
//     x[N][4], y[N][4], yv[N][4]  int
//     chain[N], window[N]         int
//     kernel_sum[N][2]            double  (flavour 0, 1)
//   2p2/          with RunInfo::with_2p2; one row per 2+2 source
//     w[M][4], chain[M], window[M]   int
//     P1[M][4][4][4][Lmax]           double
//                 and one row per (source, target)
//     pair_w[R][4], pair_y[R][4], pair_yv[R][4]  int
//     P23[R][3][4][4][4]             double  (P2_0, P2_1, P3)
// plus the table chains[C][10], and RunInfo as attributes. Each append is
// flushed, so a killed job keeps its finished windows.
class ResultWriter {
 public:
  ResultWriter(const std::string& path, const RunInfo& info);
  ~ResultWriter();

  ResultWriter(const ResultWriter&) = delete;
  ResultWriter& operator=(const ResultWriter&) = delete;

  void append(const std::vector<PairResult>& rows);
  void append_2p2(const std::vector<Result2p2>& rows);  // needs with_2p2
  long long rows() const { return count_; }
  long long rows_2p2() const { return count_2p2_; }

 private:
  long long file_ = -1;  // hid_t
  long long group_kernel_sum_ = -1, group_2p2_ = -1;
  long long ds_x_ = -1, ds_y_ = -1, ds_yv_ = -1, ds_chain_ = -1,
            ds_window_ = -1, ds_kernel_ = -1;
  long long ds_w_ = -1, ds_chain_2p2_ = -1, ds_window_2p2_ = -1, ds_p1_ = -1,
            ds_pair_w_ = -1, ds_pair_y_ = -1, ds_pair_yv_ = -1, ds_p23_ = -1;
  int lmax_ = 0;  // largest global extent: the last dimension of P1
  long long count_ = 0;
  long long count_2p2_ = 0, count_pairs_2p2_ = 0;
};

// One CSV row per solve, per kernel sum and per 2+2 source:
//   kind,chain,window,x0..x3,y0..y3,seconds,iterations
// A solve leaves y empty; a kernel sum leaves iterations empty; a 2+2 row
// (kind 2p2) has the source as x and leaves y and iterations empty.
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
  void pieces_2p2(int chain, int window, const Vec4& w, double seconds);

 private:
  std::FILE* file_ = nullptr;
};

}  // namespace hlbl
