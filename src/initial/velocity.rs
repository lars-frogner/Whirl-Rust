//! Initial velocity distributions.

pub mod rest;

use crate::{
    fluid::{ParticlePositions, ParticleVelocities},
    geometry::dim::*,
    num::fvar,
};

/// Initial distribution of fluid velocity.
pub trait InitialVelocityDistribution<D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    /// Computes the fluid particle velocities corresponding to this initial velocity
    /// distribution, for the given positions and mass densities.
    fn compute_velocities(
        &self,
        positions: &ParticlePositions<D>,
        mass_densities: &[fvar],
    ) -> ParticleVelocities<D>;
}
