//! Ideal gas equation of state

use super::EquationOfState;
use crate::{
    error::{WhirlError, WhirlResult},
    geometry::dim::*,
    num::fvar,
    units::Units,
};

/// Equation of state for a particular ideal adiabatic gas.
#[derive(Debug, Clone)]
pub struct AdiabaticIdealGasEOS {
    adiabatic_index: fvar,
    pg_from_e: fvar,
    e_from_rho_tg: fvar,
}

/// Whether gas particles are monoatomic or diatomic.
#[derive(Debug, Clone, Copy)]
pub enum GasParticleAtomicity {
    Monatomic,
    Diatomic,
}

impl AdiabaticIdealGasEOS {
    /// Creates a new ideal equation of state for an adiabatic gas with the given
    /// dimensionality, particle type and mean molecular mass.
    pub fn new<D: Dim, U: Units>(
        atomicity: GasParticleAtomicity,
        mean_molecular_mass: fvar,
    ) -> WhirlResult<Self> {
        if mean_molecular_mass <= 0.0 {
            return Err(WhirlError::from(format!(
                "Non-positive mean molecular mass: {}",
                mean_molecular_mass
            )));
        }
        let degrees_of_freedom = atomicity.degrees_of_freedom::<D>();
        let adiabatic_index = 1.0 + 2.0 / (degrees_of_freedom as fvar);
        let pg_from_e = adiabatic_index - 1.0;
        let e_from_rho_tg = U::K_B / (mean_molecular_mass * pg_from_e);
        Ok(Self {
            adiabatic_index,
            pg_from_e,
            e_from_rho_tg,
        })
    }

    /// Creates a new ideal equation of state, normalized so `P = rho*T`, for an
    /// adiabatic gas with the given adiabatic index.
    pub fn new_normalized(adiabatic_index: fvar) -> WhirlResult<Self> {
        if adiabatic_index <= 1.0 {
            return Err(WhirlError::from(format!(
                "Adiabatic index not larger than one: {}",
                adiabatic_index
            )));
        }
        let pg_from_e = adiabatic_index - 1.0;
        let e_from_rho_tg = 1.0 / (pg_from_e);
        Ok(Self {
            adiabatic_index,
            pg_from_e,
            e_from_rho_tg,
        })
    }
}

impl EquationOfState for AdiabaticIdealGasEOS {
    fn compute_pressure(&self, mass_density: fvar, specific_energy: fvar) -> fvar {
        debug_assert!(
            specific_energy > 0.0,
            "Non-positive specific energy: {}",
            specific_energy
        );
        self.pg_from_e * specific_energy * mass_density
    }

    fn compute_temperature(&self, specific_energy: fvar) -> fvar {
        debug_assert!(
            specific_energy > 0.0,
            "Non-positive energy density: {}",
            specific_energy
        );
        specific_energy / self.e_from_rho_tg
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

    fn compute_energy_density(&self, mass_density: fvar, specific_energy: fvar) -> fvar {
        debug_assert!(
            mass_density > 0.0,
            "Non-positive mass density: {}",
            mass_density
        );
        debug_assert!(
            specific_energy > 0.0,
            "Non-positive energy density: {}",
            specific_energy
        );
        specific_energy * mass_density
    }

    fn compute_specific_energy(&self, temperature: fvar) -> fvar {
        debug_assert!(
            temperature > 0.0,
            "Non-positive temperature: {}",
            temperature
        );
        self.e_from_rho_tg * temperature
    }

    fn compute_squared_sound_speed(&self, specific_energy: fvar) -> fvar {
        self.adiabatic_index * self.pg_from_e * specific_energy
    }
}

impl GasParticleAtomicity {
    /// Returns the number of degrees of freedom of the particle type, given the
    /// dimensionality of the gas.
    ///
    /// A diatomic particle has two rotational degrees of freedom in addition to the
    /// translational ones.
    pub fn degrees_of_freedom<D: Dim>(self) -> usize {
        let dimensions = D::dim();
        match self {
            Self::Monatomic => dimensions,
            Self::Diatomic => dimensions + 2,
        }
    }
}

/// Computes specific energy from mass density and pressure for an adiabatic gas given the adiabatic index.
pub fn compute_specific_energy_from_density_and_pressure(
    mass_density: fvar,
    pressure: fvar,
    adiabatic_index: fvar,
) -> fvar {
    pressure / ((adiabatic_index - 1.0) * mass_density)
}
