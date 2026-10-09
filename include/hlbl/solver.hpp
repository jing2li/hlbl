#pragma once

// tmLQCD/QUDA solver interface. The geometry, process grid, gauge field and
// operators all come from invert.input in the working directory.

#include <array>
#include <memory>

#include "hlbl/lattice.hpp"
#include "hlbl/mdarray.hpp"
#include "hlbl/provider.hpp"
#include "hlbl/vec4.hpp"

namespace hlbl {

// Use in this order: construct, read_gauge(), warm_up(). QUDA loads the gauge
// field on the first solve, so plaquette() is only valid after warm_up().
class Solver {
 public:
  Solver(int argc, char** argv);
  ~Solver();

  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;

  const Lattice& lattice() const { return *lattice_; }
  int nconf() const { return nconf_; }

  // Reads the gauge configuration named in invert.input.
  void read_gauge();

  // Does one dummy solve. This is where QUDA loads the gauge field and sets
  // up the multigrid, so call it once before any timed solve.
  void warm_up();

  // {mean, spatial, temporal} plaquette of the gauge field QUDA holds. The
  // mean should equal the value recorded in the gauge file.
  std::array<double, 3> plaquette() const;

  // Solves for the 12 spin-colour components of a point source at global
  // site gsx, with operator op_id (0: +mu, 1: -mu). The solutions
  // (12 * 24 * local volume doubles) go into `dest` starting at `offset`.
  // Call once per flavour, with offset = flavour * 12 * 24 * local volume.
  // Returns the iterations summed over the 12 solves.
  int invert_point(const Vec4& gsx, int op_id, DeviceBuffer<double>& dest,
                    std::size_t offset);

 private:
  std::unique_ptr<Lattice> lattice_;
  int nconf_ = 0;
};

// Solves both flavours of a point source into one propagator block.
class SolverProvider : public PropagatorProvider {
 public:
  explicit SolverProvider(Solver& solver) : solver_(solver) {}

  int compute(const Vec4& gsx, DeviceBuffer<double>& dest) override;

 private:
  Solver& solver_;
};

}  // namespace hlbl
