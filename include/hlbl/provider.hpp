#pragma once

// Where propagators come from. The driver asks a provider for the
// propagator block of a point source and does not care whether that is a
// tmLQCD/QUDA solve (hlbl::SolverProvider) or seeded random noise
// (hlbl::RandomProvider), which lets the schedule, cache and output be
// exercised on a small lattice with no gauge field and no solver.

#include <cstddef>
#include <cstdint>

#include "hlbl/lattice.hpp"
#include "hlbl/mdarray.hpp"
#include "hlbl/vec4.hpp"

namespace hlbl {

// Doubles in one source position's propagator block on this rank:
// [2 flavours][12 source spin-colour][24 * local volume] (README
// "Conventions").
inline std::size_t block_length(const Lattice& lattice) {
  return static_cast<std::size_t>(2) * 12 * 24 * lattice.volume();
}

class PropagatorProvider {
 public:
  virtual ~PropagatorProvider() = default;

  // Fills dest (block_length() doubles) with the propagators from the point
  // source at the global site gsx, which lies in [0, global extents).
  // Collective over all ranks. Returns the solver iterations spent, summed
  // over the 24 solves, or 0 for a provider that does not solve.
  virtual int compute(const Vec4& gsx, DeviceBuffer<double>& dest) = 0;
};

// Fills `out` (block_length() doubles, host) with noise seeded by each
// site's global lattice index and `seed`, so the field is the same under any
// process grid that tiles the same global lattice.
void fill_random_block(const Lattice& lattice, std::uint64_t seed,
                       double* out);

// Structureless noise, a different field for every source position. No
// physics: for testing the schedule, cache and output machinery.
class RandomProvider : public PropagatorProvider {
 public:
  explicit RandomProvider(const Lattice& lattice, std::uint64_t seed = 0)
      : lattice_(lattice), seed_(seed) {}

  int compute(const Vec4& gsx, DeviceBuffer<double>& dest) override;

 private:
  const Lattice& lattice_;
  std::uint64_t seed_;
};

}  // namespace hlbl
