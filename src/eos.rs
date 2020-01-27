//! Equation of state.

pub mod ideal_gas;

use crate::num::fvar;

pub trait EquationOfState {
    /// Computes the pressure for the given internal energy density.
    fn compute_pressure(&self, energy_density: fvar) -> fvar;

    /// Computes the temperature for the given mass density and internal energy density.
    fn compute_temperature(&self, mass_density: fvar, energy_density: fvar) -> fvar;

    /// Computes the mass density for the given pressure and temperature.
    fn compute_mass_density(&self, pressure: fvar, temperature: fvar) -> fvar;

    /// Computes the internal energy density for the given mass density and temperature.
    fn compute_energy_density(&self, mass_density: fvar, temperature: fvar) -> fvar;
}
