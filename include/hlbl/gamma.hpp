#pragma once

#include <array>

namespace hlbl {

// A spin-colour spinor: 12 complex numbers as 24 doubles (real, imaginary
// interleaved). Component i = 3*spin + colour is at [2i, 2i+1]. This is the
// layout of the propagator blocks.
using Spinor24 = std::array<double, 24>;

// gamma_0..gamma_3 and gamma_5 in the cvc basis. Row 5 is gamma_5; row 4 is
// unused.
inline constexpr int kNumGamma = 6;
extern const int gamma_permutation[kNumGamma][24];
extern const int gamma_sign[kNumGamma][24];

// s = gamma_index * t
inline Spinor24 apply_gamma(int gamma_index, const Spinor24& t) {
  Spinor24 s{};
  for (int k = 0; k < 24; ++k) {
    s[k] = t[gamma_permutation[gamma_index][k]] * gamma_sign[gamma_index][k];
  }
  return s;
}

}  // namespace hlbl
