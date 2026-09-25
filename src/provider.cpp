#include "hlbl/provider.hpp"

#include <vector>

namespace hlbl {

namespace {

// splitmix64 finalizer: cheap, well-mixed, deterministic.
std::uint64_t hash64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

double hash_to_unit(std::uint64_t x) {
  return static_cast<double>(hash64(x) >> 11) * (1.0 / 9007199254740992.0) *
             2.0 -
         1.0;
}

}  // namespace

void fill_random_block(const Lattice& lat, std::uint64_t seed, double* out) {
  std::size_t const vol = lat.volume();
  const Vec4& local = lat.local();
  const Vec4& pc = lat.proc_coords();
  for (int t = 0; t < local[0]; ++t) {
    for (int x = 0; x < local[1]; ++x) {
      for (int y = 0; y < local[2]; ++y) {
        for (int z = 0; z < local[3]; ++z) {
          std::size_t const site = lat.local_index({t, x, y, z});
          Vec4 const gc = {pc[0] * local[0] + t, pc[1] * local[1] + x,
                           pc[2] * local[2] + y, pc[3] * local[3] + z};
          std::uint64_t const g =
              ((static_cast<std::uint64_t>(gc[0]) * 1000003u + gc[1]) *
                   1000003u +
               gc[2]) *
                  1000003u +
              gc[3];
          for (int f = 0; f < 2; ++f) {
            for (int ia = 0; ia < 12; ++ia) {
              for (int c = 0; c < 24; ++c) {
                std::size_t const idx = ((f * 12 + ia) * vol + site) * 24 + c;
                std::uint64_t const key =
                    ((g * 2 + f) * 12 + ia) * 24 + c +
                    seed * 0x2545F4914F6CDD1DULL;
                out[idx] = hash_to_unit(key);
              }
            }
          }
        }
      }
    }
  }
}

int RandomProvider::compute(const Vec4& gsx, DeviceBuffer<double>& dest) {
  // one field per source position: fold the position into the seed
  std::uint64_t seed = seed_;
  for (int i = 0; i < 4; ++i) {
    seed = hash64(seed ^ static_cast<std::uint64_t>(gsx[i] + 1));
  }
  std::vector<double> host(block_length(lattice_));
  fill_random_block(lattice_, seed, host.data());
  dest.copy_from_host(host.data(), host.size());
  return 0;
}

}  // namespace hlbl
