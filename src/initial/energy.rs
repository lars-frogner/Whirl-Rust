//! Initial energy distributions.

pub mod isothermal;
pub mod piecewise_uniform;
pub mod uniform;

use super::mass::MassDistributionState;
use crate::{eos::EquationOfState, geometry::dim::*, num::fvar};

/// Initial distribution of fluid internal energy.
pub trait InitialEnergyDistribution<D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    /// Computes the fluid particle specific energies corresponding to this initial energy
    /// distribution, for the given mass distribution state.
    fn compute_specific_energies<EOS: EquationOfState>(
        &self,
        mass_distribution_state: &MassDistributionState<D>,
        equation_of_state: &EOS,
    ) -> Vec<fvar>;
}
