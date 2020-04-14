//! Templates for initial fluid setups.

use super::SimulationBuilder1D;
use crate::{eos::adiabatic_ideal_gas, error::WhirlResult, geometry::Bounds1D, num::fvar};

/// Creates a `SimulationBuilder1D` for the rarefaction wave experiment.
pub fn rarefaction_wave(
    initial_width: fvar,
    initial_mass_density: fvar,
    initial_energy_per_mass: fvar,
    number_of_particles: usize,
    adiabatic_index: fvar,
) -> WhirlResult<SimulationBuilder1D> {
    let coupling_constant = 1.5;

    let exterior_boundary_positions = Bounds1D::new(-0.5 * initial_width, 0.5 * initial_width)?;
    let initial_energy_density = initial_energy_per_mass * initial_mass_density;

    let mut builder = SimulationBuilder1D::new();
    builder
        .with_normalized_adiabatic_ideal_gas_eos(adiabatic_index)
        .with_piecewise_uniform_mass_distribution(
            number_of_particles,
            exterior_boundary_positions,
            Vec::new(),
            vec![initial_mass_density],
            coupling_constant,
        )
        .with_static_velocity_distribution()
        .with_uniform_energy_distribution(initial_energy_density)
        // .with_isothermal_energy_distribution(
        //     initial_energy_density / (1.0 / (adiabatic_index - 1.0) * initial_mass_density),
        // )
        .with_inviscid_fluid();
    // .with_inviscid_isothermal_fluid();
    Ok(builder)
}

/// Creates a `SimulationBuilder1D` for the Sod shock tube experiment.
pub fn sod_shock_tube(
    number_of_particles: usize,
    adiabatic_index: fvar,
) -> WhirlResult<SimulationBuilder1D> {
    let coupling_constant = 1.5;

    let exterior_boundary_positions = Bounds1D::new(0.0, 1.0)?;
    let interior_boundary_positions = vec![0.5];

    let mass_densities = (1.0, 0.125);
    let pressures = (1.0, 0.1);

    let mut builder = SimulationBuilder1D::new();
    builder
        .with_normalized_adiabatic_ideal_gas_eos(adiabatic_index)
        .with_piecewise_uniform_mass_distribution(
            number_of_particles,
            exterior_boundary_positions,
            interior_boundary_positions,
            vec![mass_densities.0, mass_densities.1],
            coupling_constant,
        )
        .with_static_velocity_distribution()
        .with_piecewise_uniform_energy_distribution(
            vec![0.5],
            vec![
                adiabatic_ideal_gas::compute_specific_energy_from_density_and_pressure(
                    mass_densities.0,
                    pressures.0,
                    adiabatic_index,
                ),
                adiabatic_ideal_gas::compute_specific_energy_from_density_and_pressure(
                    mass_densities.1,
                    pressures.1,
                    adiabatic_index,
                ),
            ],
        )
        .with_inviscid_fluid();
    Ok(builder)
}
