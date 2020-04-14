//! Initial velocity distribution with all particles at rest.

use super::{super::mass::MassDistributionState, InitialVelocityDistribution};
use crate::{fluid::ParticleVelocities, geometry::dim::*};

/// Initial velocity distribution with all particles at rest.
#[derive(Clone, Copy, Debug)]
pub struct StaticVelocityDistribution;

impl<D: Dim> InitialVelocityDistribution<D> for StaticVelocityDistribution
where
    DefaultAllocator: Allocator<D>,
{
    fn compute_velocities(
        &self,
        mass_distribution_state: &MassDistributionState<D>,
    ) -> ParticleVelocities<D> {
        vec![Vector::zeros(); mass_distribution_state.mass_densities.len()]
    }
}
