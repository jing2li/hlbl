#pragma once

// The input file of hlbl_hypercube: global options and a list of chains.
//
//   # comment
//   cache = host
//   chain start=0,0,0,0 dir=1,1,1,1 stride=4 steps=30 windows=10:20
//
// Global lines are `key = value`, chain lines are `chain key=value ...`.
// See scripts/input/hlbl.input.example for every key.

#include <istream>
#include <string>
#include <vector>

#include "hlbl/vec4.hpp"

namespace hlbl {

// One chain: windows 0..steps of the sliding hypercube from `start` in
// direction `dir`. The defaults are those of the command line.
struct Chain {
  Vec4 start{0, 0, 0, 0};
  Vec4 direction{1, 1, 1, 1};
  int stride = 4;
  int steps = -1;         // -1: as many as fit before sources wrap
  int window_begin = 0;   // run windows [window_begin, window_end)
  int window_end = -1;    // -1: through the last
};

struct Input {
  std::vector<Chain> chains;
  std::string out;                 // result file; empty: do not write one
  std::string timing;              // timing csv; empty: derived from `out`
  bool device_cache = true;        // cache = device | host
  bool random_fields = false;      // random numbers instead of solves
  int flavors = 2;                 // dry-run counts only
  Vec4 lattice{128, 64, 64, 64};   // dry-run and random fields; bounds auto steps
  Vec4 procs{1, 1, 1, 1};          // random fields only
  // xunit = m_mu a; required except for --dry-run. 0: not given.
  double muon_mass = 0;            // MeV
  double lattice_spacing = 0;      // fm
};

// Throws std::runtime_error, naming `name` and the line number, on a bad file.
Input parse_input(std::istream& in, const std::string& name);
Input parse_input_file(const std::string& path);

// "a,b,c,d" (commas or blanks) -> Vec4
Vec4 parse_vec4(const std::string& text);

// "a:b", half-open; "a:" runs through the last window, ":b" starts at 0.
void parse_window_range(const std::string& text, int& begin, int& end);

// A positive, finite number in decimal notation; `name` and `unit` go into
// the error message ("muon_mass must be a positive number (MeV)").
double parse_positive(const std::string& text, const std::string& name,
                      const std::string& unit);

// Throws std::runtime_error unless the direction is a sign vector, the stride
// is positive and the steps and window range make sense.
void validate(const Chain& chain);

}  // namespace hlbl
