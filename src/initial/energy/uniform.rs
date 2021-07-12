//! Initial uniform energy distribution.

use super::InitialEnergyDistribution;
use crate::{
    eos::EquationOfState,
    error::{WhirlError, WhirlResult},
    fluid::ParticlePositions,
    geometry::dim::*,
    num::fvar,
};

/// Initial uniform energy distribution.
#[derive(Clone, Copy, Debug)]
pub struct UniformEnergyDistribution {
    specific_energy: fvar,
}

impl UniformEnergyDistribution {
    /// Creates a new uniform energy distribution with the given specific energy.
    pub fn new(specific_energy: fvar) -> WhirlResult<Self> {
        if specific_energy <= 0.0 {
            return Err(WhirlError::from(format!(
                "Non-positive specific energy: {}",
                specific_energy
            )));
        }
        Ok(Self { specific_energy })
    }

    /// Returns the specific energy of the uniform energy distribution.
    pub fn specific_energy(&self) -> fvar {
        self.specific_energy
    }
}

impl<D: Dim> InitialEnergyDistribution<D> for UniformEnergyDistribution
where
    DefaultAllocator: Allocator<D>,
{
    fn compute_specific_energies<EOS: EquationOfState>(
        &self,
        positions: &ParticlePositions<D>,
        mass_densities: &[fvar],
        _equation_of_state: &EOS,
    ) -> Vec<fvar> {
        debug_assert_eq!(positions.len(), mass_densities.len());
        vec![self.specific_energy; positions.len()]
    }
}
