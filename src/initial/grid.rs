//! Initial mass distribution with fluid particles placed on regular grids.

use super::{InitialMassDistribution1D, MassDistributionState1D};
use crate::{fluid, geometry::Bounds1D, kernel::Kernel, num::fvar};
use std::iter;

/// Initial mass distribution with fluid particles placed on regular 1D grids.
#[derive(Clone, Debug)]
pub struct GridMassDistribution1D {
    particle_mass: fvar,
    kernel_grid_cell_widths: fvar,
    populated_grids: Vec<PopulatedGrid1D>,
}

/// 1D particle grid defined by the grid bounds and number of particles.
#[derive(Clone, Debug)]
pub struct PopulatedGrid1D {
    number_of_particles: usize,
    bounds: Bounds1D,
}

impl PopulatedGrid1D {
    /// Create a new particle grid given the number of particles and grid bounds.
    pub fn new(number_of_particles: usize, bounds: Bounds1D) -> Self {
        assert!(
            number_of_particles > 0,
            "Number of particles must be larger than zero"
        );
        Self {
            number_of_particles,
            bounds,
        }
    }
}

impl GridMassDistribution1D {
    /// Create a new grid mass distribution for the given particle grids,
    /// using the given particle mass and kernel widths relative to the grid cell
    /// extents.
    pub fn new(
        particle_mass: fvar,
        kernel_grid_cell_widths: fvar,
        populated_grids: Vec<PopulatedGrid1D>,
    ) -> Self {
        assert!(
            particle_mass > 0.0,
            "Particle mass must be larger than zero"
        );
        assert!(
            kernel_grid_cell_widths > 0.0,
            "Kernel width must be larger than zero"
        );
        assert!(
            !populated_grids.is_empty(),
            "At least one particle grid is required to generate initial distribution"
        );
        Self {
            particle_mass,
            kernel_grid_cell_widths,
            populated_grids,
        }
    }
}

impl InitialMassDistribution1D for GridMassDistribution1D {
    fn compute_state<K: Kernel>(&self, kernel: &K) -> MassDistributionState1D {
        let total_number_of_particles = self
            .populated_grids
            .iter()
            .map(
                |&PopulatedGrid1D {
                     number_of_particles,
                     ..
                 }| number_of_particles,
            )
            .sum();
        let mut positions = Vec::with_capacity(total_number_of_particles);
        let mut kernel_widths = Vec::with_capacity(total_number_of_particles);
        for &PopulatedGrid1D {
            number_of_particles,
            ref bounds,
        } in &self.populated_grids
        {
            let cell_extent = bounds.compute_extent() / (number_of_particles as fvar);
            let start = bounds.lower() + 0.5 * cell_extent;
            positions.extend((0..number_of_particles).map(|i| start + (i as fvar) * cell_extent));
            kernel_widths.extend(
                iter::repeat(cell_extent * self.kernel_grid_cell_widths).take(number_of_particles),
            );
        }

        let mass_densities = fluid::compute_mass_densities_1d(
            self.particle_mass,
            &positions,
            &kernel_widths,
            kernel,
        );

        MassDistributionState1D {
            particle_mass: self.particle_mass,
            positions,
            kernel_widths,
            mass_densities,
        }
    }
}
