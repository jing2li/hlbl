#pragma once

#include <mpi.h>

#include <cstddef>

#include "hlbl/vec4.hpp"

namespace hlbl {

// Local lattice geometry and the process grid it sits in. Local extents and
// the process grid are given explicitly here; later phases build this from
// tmLQCD's own params (see hlbl::Solver), which describe the same thing.
class Lattice {
 public:
  // procs: process grid (PT,PX,PY,PZ). local: local extents (T,LX,LY,LZ).
  // Builds a periodic Cartesian communicator over comm, cvc's ordering
  // (t,x,y,z), no reorder.
  Lattice(MPI_Comm comm, const Vec4& procs, const Vec4& local);

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
  Vec4 procs_;
  Vec4 local_;
  Vec4 global_;
  Vec4 proc_coords_{};
  MPI_Comm cart_ = MPI_COMM_NULL;
  int cart_rank_ = 0;
  std::size_t volume_ = 0;
};

}  // namespace hlbl
