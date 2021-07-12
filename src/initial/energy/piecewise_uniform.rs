//! Initial piecewise uniform energy distribution.

use super::InitialEnergyDistribution;
use crate::{
    eos::EquationOfState,
    error::{WhirlError, WhirlResult},
    fluid::ParticlePositions,
    geometry::dim::*,
    num::fvar,
};

/// Initial 1D piecewise uniform energy distribution.
#[derive(Clone, Debug)]
pub struct PiecewiseUniformEnergyDistribution1D {
    interior_boundary_positions: Vec<fvar>,
    specific_energies: Vec<fvar>,
}

impl PiecewiseUniformEnergyDistribution1D {
    /// Creates a new piecewise uniform energy distribution with the given specific energies
    /// between the given interior boundary positions.
    pub fn new(
        interior_boundary_positions: Vec<fvar>,
        specific_energies: Vec<fvar>,
    ) -> WhirlResult<Self> {
        if specific_energies.len() != interior_boundary_positions.len() + 1 {
            return Err(WhirlError::from(format!(
                "Number of energy densities ({}) are not one more than number of interior boundaries ({})",
                specific_energies.len(),
                interior_boundary_positions.len()
            )));
        }
        if interior_boundary_positions
            .iter()
            .zip(interior_boundary_positions.iter().skip(1))
            .any(|(&lower, &upper)| upper <= lower)
        {
            return Err(WhirlError::from(format!(
                "Boundary positions are not monotonically increasing: {:?}",
                interior_boundary_positions
            )));
        }
        if specific_energies
            .iter()
            .any(|&specific_energy| specific_energy <= 0.0)
        {
            return Err(WhirlError::from(format!(
                "Non-positive specific energy: {:?}",
                specific_energies
            )));
        }
        Ok(Self {
            interior_boundary_positions,
            specific_energies,
        })
    }

    /// Returns the energy density of the piecewise uniform energy distribution
    /// at the given position.
    pub fn specific_energy(&self, position: fvar) -> fvar {
        if self.interior_boundary_positions.is_empty()
            || position < self.interior_boundary_positions[0]
        {
            self.specific_energies[0]
        } else {
            self.specific_energies[1 + super::super::search_idx_of_coord(
                &self.interior_boundary_positions,
                position,
            )
            .unwrap()]
        }
    }
}

impl InitialEnergyDistribution<OneDim> for PiecewiseUniformEnergyDistribution1D {
    fn compute_specific_energies<EOS: EquationOfState>(
        &self,
        positions: &ParticlePositions<OneDim>,
        mass_densities: &[fvar],
        _equation_of_state: &EOS,
    ) -> Vec<fvar> {
        debug_assert_eq!(positions.len(), mass_densities.len());
        positions
            .iter()
            .map(|position| self.specific_energy(point_to_scalar(position)))
            .collect()
    }
}
