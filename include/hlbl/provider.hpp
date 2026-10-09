#pragma once

// Where propagators come from: a tmLQCD/QUDA solve (SolverProvider) or seeded
// random numbers (RandomProvider), which let the schedule, cache and output
// be tested without a gauge field or a solver.

#include <cstddef>
#include <cstdint>

#include "hlbl/lattice.hpp"
#include "hlbl/mdarray.hpp"
#include "hlbl/vec4.hpp"

namespace hlbl {

// Doubles in the propagator block of one source position on this rank:
// [2 flavours][12 source spin-colour][24 * local volume]. See the README.
inline std::size_t block_length(const Lattice& lattice) {
  return static_cast<std::size_t>(2) * 12 * 24 * lattice.volume();
}

class PropagatorProvider {
 public:
  virtual ~PropagatorProvider() = default;

  // Fills dest (block_length() doubles) with the propagators of the point
  // source at global site gsx. Collective over all ranks. Returns the solver
  // iterations summed over the 24 solves (0 if the provider does not solve).
  virtual int compute(const Vec4& gsx, DeviceBuffer<double>& dest) = 0;
};

// Fills `out` (host, block_length() doubles) with random numbers. Each value
// depends only on `seed` and the global site, so the field is the same for
// any process grid.
void fill_random_block(const Lattice& lattice, std::uint64_t seed,
                       double* out);

// A different random field for every source position. For tests only.
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
