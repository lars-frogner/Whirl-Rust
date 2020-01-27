//! Initialization of the fluid state.

pub mod grid;
pub mod rest;

use crate::{kernel::Kernel, num::fvar};

/// Initial 1D distribution of fluid mass.
pub trait InitialMassDistribution1D {
    /// Computes the fluid state corresponding to this initial mass distribition.
    fn compute_state<K: Kernel>(&self, kernel: &K) -> MassDistributionState1D;
}

pub trait InitialVelocityDistribution1D {
    /// Computes the fluid particle velocities corresponding to this initial velocity
    /// distribution, for the given mass distribution state.
    fn compute_velocities(&self, mass_distribution_state: &MassDistributionState1D) -> Vec<fvar>;
}

/// Fluid state associated with an initial mass distribution.
#[derive(Clone, Debug)]
pub struct MassDistributionState1D {
    /// Mass of the fluid particles.
    pub particle_mass: fvar,
    /// Fluid particle positions.
    pub positions: Vec<fvar>,
    /// Fluid particle smoothing kernel widths.
    pub kernel_widths: Vec<fvar>,
    /// Fluid particle mass densities.
    pub mass_densities: Vec<fvar>,
}
