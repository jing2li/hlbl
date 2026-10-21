#pragma once

// Unit conversion for the QED kernel. The muon mass and lattice spacing are
// run inputs; only hbar c is built in.

namespace hlbl {

// hbar c in MeV fm (same value as cvc).
inline constexpr double kHbarCMeVFm = 197.3269804;

// xunit = m_mu a converts lattice units to units of the muon mass.
inline constexpr double xunit(double muon_mass_mev, double lattice_spacing_fm) {
  return muon_mass_mev / kHbarCMeVFm * lattice_spacing_fm;
}

}  // namespace hlbl
