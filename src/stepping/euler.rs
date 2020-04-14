//! Time stepping using the Euler-Cromer scheme.

use super::Stepper;
use crate::{
    eos::EquationOfState,
    fluid::{Fluid, ParticlePositions, ParticleVelocities},
    geometry::dim::*,
    kernel::Kernel,
    num::fvar,
};
use std::iter;

/// Euler-Cromer time stepping scheme.
#[derive(Clone, Debug)]
pub struct EulerStepper<D, E, K, F>
where
    D: Dim,
    E: EquationOfState,
    K: Kernel<D>,
    F: Fluid<D>,
    DefaultAllocator: Allocator<D>,
{
    equation_of_state: E,
    kernel: K,
    fluid: F,
    time_derivatives: F::TimeDerivatives,
}

impl<D, E, K, F> EulerStepper<D, E, K, F>
where
    D: Dim,
    E: EquationOfState,
    K: Kernel<D>,
    F: Fluid<D>,
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new Euler-Cromer stepper for the given fluid.
    pub fn new(
        equation_of_state: E,
        kernel: K,
        fluid: F,
        time_derivatives: F::TimeDerivatives,
    ) -> Self {
        Self {
            equation_of_state,
            kernel,
            fluid,
            time_derivatives,
        }
    }
}

impl<D, E, K, F> Stepper<D> for EulerStepper<D, E, K, F>
where
    D: Dim,
    E: EquationOfState,
    K: Kernel<D>,
    F: Fluid<D>,
    DefaultAllocator: Allocator<D>,
{
    fn positions(&self) -> &ParticlePositions<D> {
        self.fluid.positions()
    }

    fn velocities(&self) -> &ParticleVelocities<D> {
        self.fluid.velocities()
    }

    fn kernel_widths(&self) -> &[fvar] {
        self.fluid.kernel_widths()
    }

    fn mass_densities(&self) -> &[fvar] {
        self.fluid.mass_densities()
    }

    fn specific_energies(&self) -> &[fvar] {
        self.fluid.specific_energies()
    }

    fn gas_pressures(&self) -> &[fvar] {
        self.fluid.gas_pressures()
    }

    fn step(&mut self, time_step: fvar) {
        self.fluid
            .evolve_primary_variables(iter::once((1.0, &self.time_derivatives)), time_step);
        self.fluid.evolve_positions(1.0, iter::empty(), time_step);
        self.fluid.update(
            &self.equation_of_state,
            &self.kernel,
            &mut self.time_derivatives,
        );
    }
}
