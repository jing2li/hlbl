// Checks the gamma tables: each row is a permutation with signs +-1,
// gamma_mu^2 = gamma_5^2 = 1, and different gammas anticommute.

#include "hlbl/gamma.hpp"

#include <cmath>
#include <cstdio>

#include "check.hpp"

using namespace hlbl;

namespace {

Spinor24 basis(int k) {
  Spinor24 e{};
  e[k] = 1.0;
  return e;
}

bool close(double a, double b) { return std::fabs(a - b) < 1e-12; }

void check_anticommute(int a, int b) {
  for (int k = 0; k < 24; ++k) {
    Spinor24 e = basis(k);
    Spinor24 ab = apply_gamma(a, apply_gamma(b, e));
    Spinor24 ba = apply_gamma(b, apply_gamma(a, e));
    for (int j = 0; j < 24; ++j) HLBL_CHECK(close(ab[j] + ba[j], 0.0));
  }
}

}  // namespace

int main() {
  for (int g : {0, 1, 2, 3, 5}) {
    bool seen[24] = {};
    for (int k = 0; k < 24; ++k) {
      int p = gamma_permutation[g][k];
      HLBL_CHECK(p >= 0 && p < 24 && !seen[p]);
      seen[p] = true;
      HLBL_CHECK(gamma_sign[g][k] == 1 || gamma_sign[g][k] == -1);
    }
  }

  for (int g : {0, 1, 2, 3, 5}) {
    for (int k = 0; k < 24; ++k) {
      Spinor24 twice = apply_gamma(g, apply_gamma(g, basis(k)));
      for (int j = 0; j < 24; ++j) HLBL_CHECK(close(twice[j], (j == k) ? 1.0 : 0.0));
    }
  }

  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = mu + 1; nu < 4; ++nu) check_anticommute(mu, nu);
  }
  for (int mu = 0; mu < 4; ++mu) check_anticommute(mu, 5);

  std::printf("test_gamma: all checks passed\n");
  return 0;
}
