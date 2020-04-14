//! Initial velocity distributions.

pub mod rest;

use super::mass::MassDistributionState;
use crate::{fluid::ParticleVelocities, geometry::dim::*};

/// Initial distribution of fluid velocity.
pub trait InitialVelocityDistribution<D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    /// Computes the fluid particle velocities corresponding to this initial velocity
    /// distribution, for the given mass distribution state.
    fn compute_velocities(
        &self,
        mass_distribution_state: &MassDistributionState<D>,
    ) -> ParticleVelocities<D>;
}
