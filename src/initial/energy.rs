//! Initial energy distributions.

pub mod isothermal;
pub mod piecewise_uniform;
pub mod uniform;

use crate::{eos::EquationOfState, fluid::ParticlePositions, geometry::dim::*, num::fvar};

/// Initial distribution of fluid internal energy.
pub trait InitialEnergyDistribution<D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    /// Computes the fluid particle specific energies corresponding to this initial energy
    /// distribution, for the given positions and mass densities.
    fn compute_specific_energies<EOS: EquationOfState>(
        &self,
        positions: &ParticlePositions<D>,
        mass_densities: &[fvar],
        equation_of_state: &EOS,
    ) -> Vec<fvar>;
}
