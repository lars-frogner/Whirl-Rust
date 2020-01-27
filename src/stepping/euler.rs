//! Time stepping using the Euler-Cromer scheme.

use super::Stepper;
use crate::{eos::EquationOfState, fluid::Fluid, kernel::Kernel, num::fvar};
use std::iter;

/// Euler-Cromer time stepping scheme.
#[derive(Clone, Debug)]
pub struct EulerStepper<E, K, F>
where
    E: EquationOfState,
    K: Kernel,
    F: Fluid,
{
    equation_of_state: E,
    kernel: K,
    fluid: F,
    time_derivatives: F::TimeDerivatives,
}

impl<E, K, F> EulerStepper<E, K, F>
where
    E: EquationOfState,
    K: Kernel,
    F: Fluid,
{
    /// Creates a new Euler-Cromer stepper for the given fluid.
    pub fn new(equation_of_state: E, kernel: K, fluid: F) -> Self {
        let time_derivatives = fluid.compute_time_derivatives(&equation_of_state, &kernel);
        Self {
            equation_of_state,
            kernel,
            fluid,
            time_derivatives,
        }
    }
}

impl<E, K, F> Stepper<F> for EulerStepper<E, K, F>
where
    E: EquationOfState,
    K: Kernel,
    F: Fluid,
{
    fn fluid(&self) -> &F {
        &self.fluid
    }

    fn step(&mut self, time_step: fvar) {
        self.fluid
            .evolve_primary_variables(iter::once((1.0, &self.time_derivatives)), time_step);
        self.fluid.evolve_positions(1.0, iter::empty(), time_step);
        self.fluid.update_static_properties(&self.kernel);
        self.fluid
            .compute_time_derivatives(&self.equation_of_state, &self.kernel);
    }
}
