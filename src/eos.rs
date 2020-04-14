//! Equation of state.

pub mod adiabatic_ideal_gas;

use crate::num::fvar;

pub trait EquationOfState {
    /// Computes the pressure for the given mass density and internal specific energy.
    fn compute_pressure(&self, mass_density: fvar, specific_energy: fvar) -> fvar;

    /// Computes the temperature for the given internal specific energy.
    fn compute_temperature(&self, specific_energy: fvar) -> fvar;

    /// Computes the mass density for the given pressure and temperature.
    fn compute_mass_density(&self, pressure: fvar, temperature: fvar) -> fvar;

    /// Computes the internal energy density for the given mass density and specific energy.
    fn compute_energy_density(&self, mass_density: fvar, specific_energy: fvar) -> fvar;

    /// Computes the internal specific energy for the given temperature.
    fn compute_specific_energy(&self, temperature: fvar) -> fvar;

    /// Computes the squared sound speed from the given internal specific energy.
    fn compute_squared_sound_speed(&self, specific_energy: fvar) -> fvar;
}
