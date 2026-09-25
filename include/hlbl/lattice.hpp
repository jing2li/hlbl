#pragma once

#include <mpi.h>

#include <cstddef>

#include "hlbl/vec4.hpp"

namespace hlbl {

// Local lattice geometry and the process grid it sits in.
class Lattice {
 public:
  // procs: process grid (PT,PX,PY,PZ). local: local extents (T,LX,LY,LZ).
  // Builds a periodic Cartesian communicator over comm, cvc's ordering
  // (t,x,y,z), no reorder.
  Lattice(MPI_Comm comm, const Vec4& procs, const Vec4& local);

  // Wraps an ALREADY Cartesian communicator by duplicating it, rather than
  // building a second, redundant topology on top of it. Used by
  // hlbl::Solver to wrap tmLQCD's own tmLQCD_mpi_params.cart_grid -- cvc's
  // own mpi_init.cpp does the same thing for the same reason
  // (MPI_Comm_dup(g_tmLQCD_mpi.cart_grid, &g_cart_grid)).
  static Lattice from_existing_cart(MPI_Comm cart, const Vec4& procs,
                                     const Vec4& local);

  const Vec4& local() const { return local_; }
  const Vec4& global() const { return global_; }
  const Vec4& procs() const { return procs_; }
  const Vec4& proc_coords() const { return proc_coords_; }
  MPI_Comm cart_comm() const { return cart_; }
  int cart_rank() const { return cart_rank_; }
  std::size_t volume() const { return volume_; }

  // Local site index, cvc's ((t*LX+x)*LY+y)*LZ+z: t slowest, z fastest.
  // Coordinates must be in [0, local).
  std::size_t local_index(const Vec4& local_coord) const;

  // The rank owning a global site, and that site's local coordinates on it
  // (undefined on every other rank). global_coord must be in [0, global).
  // Mirrors cvc get_point_source_info().
  struct Owner {
    int rank;
    Vec4 local_coord;
  };
  Owner owner(const Vec4& global_coord) const;

  // Minimum-image difference a - b on this lattice's global extents.
  Vec4 site_map_diff(const Vec4& a, const Vec4& b) const {
    return hlbl::site_map_diff(a, b, global_);
  }

 private:
  struct AlreadyCartesian {};
  Lattice(AlreadyCartesian, MPI_Comm cart, const Vec4& procs,
          const Vec4& local);

  Vec4 procs_;
  Vec4 local_;
  Vec4 global_;
  Vec4 proc_coords_{};
  MPI_Comm cart_ = MPI_COMM_NULL;
  int cart_rank_ = 0;
  std::size_t volume_ = 0;
};

}  // namespace hlbl
