#include "hlbl/solver.hpp"

#include <stdexcept>
#include <vector>

extern "C" {
#include "tmLQCD.h"
}
#include "quda.h"

namespace hlbl {

namespace {

void check(int status, const char* what) {
  if (status != 0) {
    throw std::runtime_error(std::string("hlbl::Solver: ") + what +
                              " failed, status " + std::to_string(status));
  }
}

// A single 12*volume-double propagator: [source component][site][sink
// component, complex], TXYZ lexicographic order -- the layout
// invert_quda_direct's host-side source/propagator buffers use.
constexpr int kSpinColor = 12;
constexpr int kComplex = 2;

}  // namespace

Solver::Solver(int argc, char** argv) {
  check(tmLQCD_invert_init(argc, argv, /*verbose=*/1, /*external_id=*/0),
        "tmLQCD_invert_init");

  tmLQCD_mpi_params mpi_params;
  check(tmLQCD_get_mpi_params(&mpi_params), "tmLQCD_get_mpi_params");

  tmLQCD_lat_params lat_params;
  check(tmLQCD_get_lat_params(&lat_params), "tmLQCD_get_lat_params");

  Vec4 procs{static_cast<int>(mpi_params.nproc_t),
             static_cast<int>(mpi_params.nproc_x),
             static_cast<int>(mpi_params.nproc_y),
             static_cast<int>(mpi_params.nproc_z)};
  Vec4 local{static_cast<int>(lat_params.T), static_cast<int>(lat_params.LX),
             static_cast<int>(lat_params.LY), static_cast<int>(lat_params.LZ)};
  lattice_ = std::make_unique<Lattice>(
      Lattice::from_existing_cart(mpi_params.cart_grid, procs, local));

  nconf_ = static_cast<int>(lat_params.nstore);
}

Solver::~Solver() { tmLQCD_finalise(); }

void Solver::read_gauge() {
  check(tmLQCD_read_gauge(nconf_), "tmLQCD_read_gauge");
  double* gauge_field = nullptr;
  check(tmLQCD_get_gauge_field_pointer(&gauge_field),
        "tmLQCD_get_gauge_field_pointer");
  if (gauge_field == nullptr) {
    throw std::runtime_error("hlbl::Solver: gauge field pointer is null");
  }
}

namespace {

// One point-source solve, discarding nothing: op_id's twisted-mass operator,
// unit source at local site `local_index` (or none, on ranks that do not
// own it) and spin-colour component `component`, solution into `solution`
// (kappa-normalised, matching invert_quda_direct's own convention).
int solve_component(bool owns_site, std::size_t local_index, int component,
                     int op_id, std::vector<double>& source,
                     std::vector<double>& solution) {
  std::fill(source.begin(), source.end(), 0.0);
  if (owns_site) {
    source[kSpinColor * kComplex * local_index + kComplex * component] = 1.0;
  }
  int const iter = invert_quda_direct(solution.data(), source.data(), op_id);
  if (iter < 0) {
    throw std::runtime_error(
        "hlbl::Solver: invert_quda_direct did not converge");
  }
  return iter;
}

}  // namespace

void Solver::warm_up() {
  // Any solve pays for QUDA's one-time gauge load and (for an MG operator)
  // multigrid setup; the source and its solution are otherwise irrelevant,
  // so a single point solve at the origin is enough -- unlike
  // time_invert.cpp's dummy volume-source solve, which additionally wants
  // to be a representative *timing* sample, not a concern here.
  std::size_t const volume = lattice_->volume();
  std::vector<double> source(kSpinColor * kComplex * volume, 0.0);
  std::vector<double> solution(kSpinColor * kComplex * volume, 0.0);
  auto owner = lattice_->owner(Vec4{0, 0, 0, 0});
  bool const owns = (owner.rank == lattice_->cart_rank());
  solve_component(owns, lattice_->local_index(owner.local_coord), 0,
                   /*op_id=*/0, source, solution);
}

std::array<double, 3> Solver::plaquette() const {
  double plaq[3] = {0.0, 0.0, 0.0};
  plaqQuda(plaq);
  return {plaq[0], plaq[1], plaq[2]};
}

int Solver::invert_point(const Vec4& gsx, int op_id,
                          DeviceBuffer<double>& dest, std::size_t offset) {
  std::size_t const volume = lattice_->volume();
  std::size_t const spinor_len = kSpinColor * kComplex * volume;

  auto owner = lattice_->owner(gsx);
  bool const owns = (owner.rank == lattice_->cart_rank());
  std::size_t const site = owns ? lattice_->local_index(owner.local_coord) : 0;

  std::vector<double> source(spinor_len), solution(spinor_len);
  int iterations = 0;
  for (int i = 0; i < kSpinColor; ++i) {
    iterations += solve_component(owns, site, i, op_id, source, solution);
    dest.copy_from_host(offset + static_cast<std::size_t>(i) * spinor_len,
                         solution.data(), spinor_len);
  }
  return iterations;
}

int SolverProvider::compute(const Vec4& gsx, DeviceBuffer<double>& dest) {
  std::size_t const per_flavor = block_length(solver_.lattice()) / 2;
  int iterations = 0;
  for (int flavor = 0; flavor < 2; ++flavor) {
    iterations += solver_.invert_point(gsx, flavor, dest, flavor * per_flavor);
  }
  return iterations;
}

}  // namespace hlbl
