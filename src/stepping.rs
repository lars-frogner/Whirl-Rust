//! Time stepping.

pub mod euler;

use crate::{
    fluid::{ParticlePositions, ParticleVelocities},
    geometry::dim::*,
    num::fvar,
};

/// Time stepping scheme.
pub trait Stepper<D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    /// Returns a reference to the positions of the fluid particles.
    fn positions(&self) -> &ParticlePositions<D>;

    /// Returns a reference to the velocities of the fluid particles.
    fn velocities(&self) -> &ParticleVelocities<D>;

    /// Returns a slice with the widths of the particle smoothing kernels.
    fn kernel_widths(&self) -> &[fvar];

    /// Returns a slice with the mass densities of the fluid particles.
    fn mass_densities(&self) -> &[fvar];

    /// Returns a slice with the energies per mass of the fluid particles.
    fn specific_energies(&self) -> &[fvar];

    /// Returns a slice with the gas pressures of the fluid particles.
    fn gas_pressures(&self) -> &[fvar];

    /// Advance the state of the fluid by the given time step.
    fn step(&mut self, time_step: fvar);
}
