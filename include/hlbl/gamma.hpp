#pragma once

#include <array>

namespace hlbl {

// A 12-component spin-colour spinor stored as 24 doubles, real and imaginary
// parts interleaved: component i = 3*spin + colour occupies [2i, 2i+1].
// Matches cvc's _GSI layout and the propagator blocks the vendored GPU
// kernels expect.
using Spinor24 = std::array<double, 24>;

// gamma_0 .. gamma_3 and gamma_5, in the cvc basis and sign convention
// (cvc_utils.cpp init_gamma()). Row index is the cvc gamma index: 0-3 are
// the spacetime gammas, 5 is gamma_5. Row 4 is unused and zeroed.
inline constexpr int kNumGamma = 6;
extern const int gamma_permutation[kNumGamma][24];
extern const int gamma_sign[kNumGamma][24];

// s = gamma_index . t, i.e. cvc's _fv_eq_gamma_ti_fv(s, gamma_index, t).
inline Spinor24 apply_gamma(int gamma_index, const Spinor24& t) {
  Spinor24 s{};
  for (int k = 0; k < 24; ++k) {
    s[k] = t[gamma_permutation[gamma_index][k]] * gamma_sign[gamma_index][k];
  }
  return s;
}

}  // namespace hlbl
