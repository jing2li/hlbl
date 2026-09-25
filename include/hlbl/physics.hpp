#pragma once

// Physical constants shared by anything that evaluates the QED kernel.
// Values match hlbl_mII_invert_contract.cpp exactly.

namespace hlbl {

// Muon mass in lattice-natural units, MeV / (MeV.fm).
inline constexpr double kMuonMassMeV = 105.6583745;
inline constexpr double kMuonMassMeVFm = 197.3269804;
inline constexpr double kMuonMass = kMuonMassMeV / kMuonMassMeVFm;

// cB211a.072.64 lattice spacing, fm. Its uncertainty (alat[1] in cvc) is
// carried there only for error propagation, never used in a kernel
// argument, so it has no counterpart here.
inline constexpr double kLatticeSpacing = 0.07957;

// x, y and every kernel argument built from them are in units of the muon
// mass: xm = x_lattice_units * xunit().
inline constexpr double xunit() { return kMuonMass * kLatticeSpacing; }

}  // namespace hlbl
