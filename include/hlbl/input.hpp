#pragma once

// The input file of hlbl_hypercube: global options and a list of chains.
//
//   # comment
//   cache = host
//   chain start=0,0,0,0 dir=1,1,1,1 stride=4 steps=30
//
// Global lines are `key = value`, chain lines are `chain key=value ...`.
// See scripts/input/hlbl.input.example for every key.

#include <istream>
#include <optional>
#include <string>
#include <vector>

#include "hlbl/vec4.hpp"

namespace hlbl {

// One chain: windows 0..steps of the sliding hypercube from `start` in
// direction `dir`. start, dir and stride are required.
struct Chain {
  Vec4 start{};
  Vec4 direction{};
  int stride = 0;
  int steps = -1;  // -1: as many as fit before sources wrap
};

// Required keys are empty (or 0) when not given; see check_required().
struct Input {
  std::vector<Chain> chains;
  // A solver run checks lattice and procs against invert.input.
  std::optional<Vec4> lattice;     // global extents
  std::optional<Vec4> procs;       // process grid
  std::string cache;               // "device" | "host"
  std::string out;                 // result file
  std::string timing;              // optional; default <out>.timing.csv
  bool random_fields = false;      // optional; random numbers instead of solves
  double muon_mass = 0;            // MeV
  double lattice_spacing = 0;      // fm
};

// Throws std::runtime_error, naming `name` and the line number, on a bad file.
Input parse_input(std::istream& in, const std::string& name);
Input parse_input_file(const std::string& path);

// Throws std::runtime_error ("<name>: missing ...") unless `in` has chain
// lines, lattice, procs, muon_mass, lattice_spacing, out and cache.
void check_required(const Input& in, const std::string& name);

// "a,b,c,d" (commas or blanks) -> Vec4
Vec4 parse_vec4(const std::string& text);

// A positive, finite number in decimal notation; `name` and `unit` go into
// the error message ("muon_mass must be a positive number (MeV)").
double parse_positive(const std::string& text, const std::string& name,
                      const std::string& unit);

// Throws std::runtime_error unless the direction is a sign vector, the stride
// is positive and steps >= -1.
void validate(const Chain& chain);

}  // namespace hlbl
