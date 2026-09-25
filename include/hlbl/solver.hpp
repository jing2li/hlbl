#pragma once

// tmLQCD/QUDA solver interface. No cvc dependency: geometry, the process
// grid, the gauge field and the operator parameters all come from tmLQCD's
// own input file (invert.input, read from the current working directory by
// tmLQCD_invert_init) and its own wrapper API (tmLQCD.h).

#include <array>
#include <memory>

#include "hlbl/lattice.hpp"
#include "hlbl/mdarray.hpp"
#include "hlbl/vec4.hpp"

namespace hlbl {

// One process's view of the solver: geometry (from tmLQCD_get_mpi_params/
// tmLQCD_get_lat_params), the gauge field, and point-source inversions.
//
// Construction order matters and mirrors time_hlbl/time_invert.cpp: build
// the Solver (tmLQCD/QUDA init + geometry), read_gauge(), then warm_up()
// once before trusting the cost or the plaquette of anything else -- QUDA
// loads the gauge into its own device-side representation lazily, on its
// first solve, so plaquette() is meaningless before that.
class Solver {
 public:
  Solver(int argc, char** argv);
  ~Solver();

  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;

  const Lattice& lattice() const { return *lattice_; }
  int nconf() const { return nconf_; }

  // Reads the configuration invert.input's GaugeConfigInputFile names, at
  // the configuration number invert.input's InitialStoreCounter gives
  // (both already parsed by tmLQCD_invert_init; nothing here reads a file
  // path itself).
  void read_gauge();

  // A dummy volume-source solve: where the QUDA multigrid setup (or, for a
  // plain CG operator, nothing extra) is paid for. Call once, before timing
  // or trusting any other solve's cost, and before plaquette().
  void warm_up();

  // QUDA's plaqQuda(): {spatial, temporal, mean} plaquette of the gauge it
  // has loaded. Only meaningful after read_gauge() and at least one solve.
  std::array<double, 3> plaquette() const;

  // 12 point-source solves at the global site gsx, one per spin-colour
  // component, for the given tmLQCD operator (0 or 1: the two operators
  // invert.input defines, light +/-mu in production). Writes the 12 * 24 *
  // local-volume doubles of the solutions into `dest` at `offset` -- the
  // caller supplies a [2 flavours][12][24 * local volume] device buffer
  // (hlbl::Contractor's fwd_src/fwd_y layout) and calls this once per
  // flavour, offset = flavour * 12 * 24 * local volume (see
  // sanity_check/check_pair.cpp).
  void invert_point(const Vec4& gsx, int op_id, DeviceBuffer<double>& dest,
                     std::size_t offset);

 private:
  std::unique_ptr<Lattice> lattice_;
  int nconf_ = 0;
};

}  // namespace hlbl
