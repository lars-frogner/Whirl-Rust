//! Physical units.

pub mod si;

use crate::num::fvar;

/// Constants written in the same system of physical units.
pub trait Units {
    /// Boltzmann constant.
    const K_B: fvar;
    /// Atomic mass unit.
    const AMU: fvar;
}
