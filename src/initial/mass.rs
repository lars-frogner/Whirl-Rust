//! Initial mass distributions.

pub mod piecewise_uniform;

use crate::{fluid::ParticlePositions, geometry::dim::*, kernel::Kernel, num::fvar};

/// Initial distribution of fluid mass.
pub trait InitialMassDistribution<D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    /// Computes the fluid state corresponding to this initial mass distribition.
    fn compute_state<K: Kernel<D>>(&self, kernel: &K) -> MassDistributionState<D>;
}

/// Fluid state associated with an initial mass distribution.
#[derive(Clone, Debug)]
pub struct MassDistributionState<D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    /// Mass of the fluid particles.
    pub particle_mass: fvar,
    /// Scaling for the kernel widths (typically between 1.2 and 1.5).
    pub coupling_constant: fvar,
    /// Fluid particle positions.
    pub positions: ParticlePositions<D>,
    /// Fluid particle smoothing kernel widths.
    pub kernel_widths: Vec<fvar>,
    /// Fluid particle mass densities.
    pub mass_densities: Vec<fvar>,
}
