//! Initial isothermal energy distribution.

use super::InitialEnergyDistribution;
use crate::{
    eos::EquationOfState,
    error::{WhirlError, WhirlResult},
    fluid::ParticlePositions,
    geometry::dim::*,
    num::fvar,
};

/// Initial isothermal energy distribution.
#[derive(Clone, Copy, Debug)]
pub struct IsothermalEnergyDistribution {
    temperature: fvar,
}

impl IsothermalEnergyDistribution {
    /// Creates a new isothermal energy distribution with the given temperature.
    pub fn new(temperature: fvar) -> WhirlResult<Self> {
        if temperature <= 0.0 {
            return Err(WhirlError::from(format!(
                "Non-positive temperature: {}",
                temperature
            )));
        }
        Ok(Self { temperature })
    }

    /// Returns the temperature of the isothermal energy distribution.
    pub fn temperature(&self) -> fvar {
        self.temperature
    }
}

impl<D: Dim> InitialEnergyDistribution<D> for IsothermalEnergyDistribution
where
    DefaultAllocator: Allocator<D>,
{
    fn compute_specific_energies<EOS: EquationOfState>(
        &self,
        positions: &ParticlePositions<D>,
        mass_densities: &[fvar],
        equation_of_state: &EOS,
    ) -> Vec<fvar> {
        debug_assert_eq!(positions.len(), mass_densities.len());
        vec![equation_of_state.compute_specific_energy(self.temperature); positions.len()]
    }
}
