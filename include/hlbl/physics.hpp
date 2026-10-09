#pragma once

// Physical constants for the QED kernel (same values as cvc).

namespace hlbl {

// Muon mass in lattice-natural units, MeV / (MeV.fm).
inline constexpr double kMuonMassMeV = 105.6583745;
inline constexpr double kMuonMassMeVFm = 197.3269804;
inline constexpr double kMuonMass = kMuonMassMeV / kMuonMassMeVFm;

// Lattice spacing of cB211a.072.64, in fm.
inline constexpr double kLatticeSpacing = 0.07957;

// Converts lattice units to units of the muon mass.
inline constexpr double xunit() { return kMuonMass * kLatticeSpacing; }

}  // namespace hlbl
