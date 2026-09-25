#include "hlbl/lattice.hpp"

#include <stdexcept>

namespace hlbl {

namespace {

void check_size(MPI_Comm comm, const Vec4& procs) {
  int want = procs[0] * procs[1] * procs[2] * procs[3];
  int have = 0;
  MPI_Comm_size(comm, &have);
  if (want != have) {
    throw std::runtime_error(
        "Lattice: process grid does not match the communicator size");
  }
}

}  // namespace

Lattice::Lattice(MPI_Comm comm, const Vec4& procs, const Vec4& local)
    : procs_(procs), local_(local) {
  for (int i = 0; i < 4; ++i) global_[i] = local_[i] * procs_[i];
  check_size(comm, procs_);

  int dims[4] = {procs_[0], procs_[1], procs_[2], procs_[3]};
  int periods[4] = {1, 1, 1, 1};
  MPI_Cart_create(comm, 4, dims, periods, /*reorder=*/0, &cart_);
  MPI_Comm_rank(cart_, &cart_rank_);
  int coords[4];
  MPI_Cart_coords(cart_, cart_rank_, 4, coords);
  for (int i = 0; i < 4; ++i) proc_coords_[i] = coords[i];

  volume_ = static_cast<std::size_t>(local_[0]) * local_[1] * local_[2] *
            local_[3];
}

Lattice::Lattice(AlreadyCartesian, MPI_Comm cart, const Vec4& procs,
                  const Vec4& local)
    : procs_(procs), local_(local) {
  for (int i = 0; i < 4; ++i) global_[i] = local_[i] * procs_[i];
  check_size(cart, procs_);

  cart_ = cart;
  MPI_Comm_rank(cart_, &cart_rank_);
  int coords[4];
  MPI_Cart_coords(cart_, cart_rank_, 4, coords);
  for (int i = 0; i < 4; ++i) proc_coords_[i] = coords[i];

  volume_ = static_cast<std::size_t>(local_[0]) * local_[1] * local_[2] *
            local_[3];
}

Lattice Lattice::from_existing_cart(MPI_Comm cart, const Vec4& procs,
                                     const Vec4& local) {
  MPI_Comm dup = MPI_COMM_NULL;
  MPI_Comm_dup(cart, &dup);
  return Lattice(AlreadyCartesian{}, dup, procs, local);
}

std::size_t Lattice::local_index(const Vec4& c) const {
  return ((static_cast<std::size_t>(c[0]) * local_[1] + c[1]) * local_[2] +
          c[2]) *
             local_[3] +
         c[3];
}

Lattice::Owner Lattice::owner(const Vec4& g) const {
  int coords[4] = {g[0] / local_[0], g[1] / local_[1], g[2] / local_[2],
                    g[3] / local_[3]};
  int rank = 0;
  MPI_Cart_rank(cart_, coords, &rank);
  Vec4 lc = {g[0] % local_[0], g[1] % local_[1], g[2] % local_[2],
             g[3] % local_[3]};
  return Owner{rank, lc};
}

}  // namespace hlbl
