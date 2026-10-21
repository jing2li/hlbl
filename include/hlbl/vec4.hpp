#pragma once

#include <array>

namespace hlbl {

// A lattice site or offset, (t, x, y, z) as in cvc.
using Vec4 = std::array<int, 4>;

constexpr Vec4 add(Vec4 a, Vec4 b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]};
}
constexpr Vec4 sub(Vec4 a, Vec4 b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]};
}
constexpr Vec4 scale(int s, Vec4 a) {
  return {s * a[0], s * a[1], s * a[2], s * a[3]};
}

// a mod L, folded into [0, L).
inline int mod_pos(int a, int L) {
  int r = a % L;
  return r < 0 ? r + L : r;
}
inline Vec4 mod_pos(Vec4 a, Vec4 L) {
  return {mod_pos(a[0], L[0]), mod_pos(a[1], L[1]), mod_pos(a[2], L[2]),
          mod_pos(a[3], L[3])};
}

// x in [0, L) -> centred representation in (-L/2, L/2]. cvc site_map(),
// arXiv:2006.16224 eq. 8.
inline int site_map_1d(int x, int L) { return x >= L / 2 ? x - L : x; }
inline Vec4 site_map(Vec4 x, Vec4 L) {
  return {site_map_1d(x[0], L[0]), site_map_1d(x[1], L[1]),
          site_map_1d(x[2], L[2]), site_map_1d(x[3], L[3])};
}

// Minimum-image difference a - b (cvc site_map_diff()). Use it for every
// kernel argument built as a difference: site_map(a) - site_map(b) can leave
// (-L/2, L/2] when a and b wrap differently.
inline Vec4 site_map_diff(Vec4 a, Vec4 b, Vec4 L) {
  return site_map(mod_pos(sub(a, b), L), L);
}

}  // namespace hlbl
