//! Time stepping.

pub mod euler;

use crate::{fluid::Fluid, num::fvar};

/// Time stepping scheme.
pub trait Stepper<F: Fluid> {
    /// Returns a reference to the fluid state.
    fn fluid(&self) -> &F;

    /// Advance the state of the fluid by the given time step.
    fn step(&mut self, time_step: fvar);
}
