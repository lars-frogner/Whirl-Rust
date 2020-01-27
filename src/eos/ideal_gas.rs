//! Ideal gas equation of state

use super::EquationOfState;
use crate::{geometry::Dimensionality, num::fvar, units::Units};

/// Equation of state for a particular ideal gas.
#[derive(Debug, Clone)]
pub struct IdealGasEOS {
    pg_from_e: fvar,
    e_from_rho_tg: fvar,
}

/// Whether gas particles are monoatomic or diatomic.
#[derive(Debug, Clone, Copy)]
pub enum GasParticleType {
    Monatomic,
    Diatomic,
}

impl IdealGasEOS {
    /// Creates a new ideal equation of state for a gas with the given
    /// dimensionality, particle type and mean molecular mass.
    pub fn new<U: Units>(
        dimensionality: Dimensionality,
        particle_type: GasParticleType,
        mean_molecular_mass: fvar,
    ) -> Self {
        assert!(
            mean_molecular_mass > 0.0,
            "Non-positive mean molecular mass: {}",
            mean_molecular_mass
        );
        let degrees_of_freedom = particle_type.degrees_of_freedom(dimensionality);
        let adiabatic_index = 1.0 + 2.0 / (degrees_of_freedom as fvar);
        let pg_from_e = adiabatic_index - 1.0;
        let e_from_rho_tg = U::K_B / (mean_molecular_mass * pg_from_e);
        Self {
            pg_from_e,
            e_from_rho_tg,
        }
    }
}

impl EquationOfState for IdealGasEOS {
    fn compute_pressure(&self, energy_density: fvar) -> fvar {
        debug_assert!(
            energy_density > 0.0,
            "Non-positive energy density: {}",
            energy_density
        );
        self.pg_from_e * energy_density
    }

    fn compute_temperature(&self, mass_density: fvar, energy_density: fvar) -> fvar {
        debug_assert!(
            mass_density > 0.0,
            "Non-positive mass density: {}",
            mass_density
        );
        debug_assert!(
            energy_density > 0.0,
            "Non-positive energy density: {}",
            energy_density
        );
        energy_density / (self.e_from_rho_tg * mass_density)
    }

    fn compute_mass_density(&self, pressure: fvar, temperature: fvar) -> fvar {
        debug_assert!(pressure > 0.0, "Non-positive pressure: {}", pressure);
        debug_assert!(
            temperature > 0.0,
            "Non-positive temperature: {}",
            temperature
        );
        pressure / (self.e_from_rho_tg * self.pg_from_e * temperature)
    }

    fn compute_energy_density(&self, mass_density: fvar, temperature: fvar) -> fvar {
        debug_assert!(
            mass_density > 0.0,
            "Non-positive mass density: {}",
            mass_density
        );
        debug_assert!(
            temperature > 0.0,
            "Non-positive temperature: {}",
            temperature
        );
        self.e_from_rho_tg * mass_density * temperature
    }
}

impl GasParticleType {
    /// Returns the number of degrees of freedom of the particle type, given the
    /// dimensionality of the gas.
    ///
    /// A diatomic particle has two rotational degrees of freedom in addition to the
    /// translational ones.
    pub fn degrees_of_freedom(self, dimensionality: Dimensionality) -> usize {
        let dimensions = dimensionality.number();
        match self {
            Self::Monatomic => dimensions,
            Self::Diatomic => dimensions + 2,
        }
    }
}
