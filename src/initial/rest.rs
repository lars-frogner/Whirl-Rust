//! Initial velocity distribution with all particles at rest.

use super::{InitialVelocityDistribution1D, MassDistributionState1D};
use crate::num::fvar;

/// Initial 1D velocity distribution with all particles at rest.
#[derive(Clone, Copy, Debug)]
pub struct StaticVelocityDistribution1D;

impl InitialVelocityDistribution1D for StaticVelocityDistribution1D {
    fn compute_velocities(&self, mass_distribution_state: &MassDistributionState1D) -> Vec<fvar> {
        vec![0.0; mass_distribution_state.mass_densities.len()]
    }
}
