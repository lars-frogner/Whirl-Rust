//! SI units.

use super::Units;
use crate::num::fvar;

/// Constants written in SI units.
#[derive(Debug)]
pub struct SIUnits;

impl Units for SIUnits {
    /// Boltzmann constant [m^2 kg/(s^2 K)]
    const K_B: fvar = 1.380_648_52e-23;
    /// Atomic mass unit [kg]
    const AMU: fvar = 1.66054e-27;
}
