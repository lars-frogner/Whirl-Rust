//! Initial piecewise uniform mass distribution.

use super::{InitialMassDistribution, MassDistributionState};
use crate::{
    error::{WhirlError, WhirlResult},
    fluid,
    geometry::{dim::*, Bounds1D},
    kernel::Kernel,
    num::fvar,
};
use std::iter;

/// Initial 1D piecewise uniform mass distribution.
#[derive(Clone, Debug)]
pub struct PiecewiseUniformMassDistribution1D {
    number_of_particles: usize,
    boundary_positions: Vec<fvar>,
    mass_densities: Vec<fvar>,
    coupling_constant: fvar,
}

impl PiecewiseUniformMassDistribution1D {
    /// Creates a new piecewise uniform mass distribution with the given mass densities
    /// between the given boundary positions.
    pub fn new(
        number_of_particles: usize,
        exterior_boundary_positions: Bounds1D,
        interior_boundary_positions: Vec<fvar>,
        mass_densities: Vec<fvar>,
        coupling_constant: fvar,
    ) -> WhirlResult<Self> {
        if number_of_particles == 0 {
            return Err(WhirlError::from(
                "Number of particles must be larger than zero",
            ));
        }
        if mass_densities.len() != interior_boundary_positions.len() + 1 {
            return Err(WhirlError::from(format!(
                "Number of mass densities ({}) are not one more than number of interior boundaries ({})",
                mass_densities.len(),
                interior_boundary_positions.len()
            )));
        }
        let boundary_positions: Vec<_> = iter::once(exterior_boundary_positions.lower())
            .chain(interior_boundary_positions.into_iter())
            .chain(iter::once(exterior_boundary_positions.upper()))
            .collect();
        if boundary_positions
            .iter()
            .zip(boundary_positions.iter().skip(1))
            .any(|(&lower, &upper)| upper <= lower)
        {
            return Err(WhirlError::from(format!(
                "Boundary positions are not monotonically increasing: {:?}",
                boundary_positions
            )));
        }
        if mass_densities
            .iter()
            .any(|&mass_density| mass_density <= 0.0)
        {
            return Err(WhirlError::from(format!(
                "Non-positive mass density: {:?}",
                mass_densities
            )));
        }
        if coupling_constant <= 0.0 {
            return Err(WhirlError::from(
                "Coupling constant must be larger than zero",
            ));
        }
        Ok(Self {
            number_of_particles,
            boundary_positions,
            mass_densities,
            coupling_constant,
        })
    }
}

impl InitialMassDistribution<OneDim> for PiecewiseUniformMassDistribution1D {
    fn compute_state<K: Kernel<OneDim>>(&self, _kernel: &K) -> MassDistributionState<OneDim> {
        let piecewise_iter = self.mass_densities.iter().copied().zip(
            self.boundary_positions
                .iter()
                .copied()
                .zip(self.boundary_positions.iter().skip(1).copied()),
        );

        let mut particle_mass = piecewise_iter.clone().fold(
            0.0,
            |accum, (mass_density, (lower_bound, upper_bound))| {
                accum + mass_density * (upper_bound - lower_bound)
            },
        );
        particle_mass /= self.number_of_particles as fvar;

        let mut positions = Vec::with_capacity(self.number_of_particles);
        let mut mass_densities = Vec::with_capacity(self.number_of_particles);

        for (mass_density, (lower_bound, upper_bound)) in piecewise_iter {
            let extent = upper_bound - lower_bound;
            let local_number_of_particles =
                fvar::round(mass_density * extent / particle_mass) as usize;
            let cell_extent = extent / (local_number_of_particles as fvar);
            let start = lower_bound + 0.5 * cell_extent;
            positions.extend(
                (0..local_number_of_particles)
                    .map(|i| Point1D::new(start + (i as fvar) * cell_extent)),
            );

            mass_densities.extend(iter::repeat(mass_density).take(local_number_of_particles));
        }

        let kernel_widths = fluid::compute_initial_kernel_widths::<OneDim>(
            particle_mass,
            self.coupling_constant,
            &mass_densities,
        );

        MassDistributionState {
            particle_mass,
            positions,
            kernel_widths,
            mass_densities,
        }
    }
}
