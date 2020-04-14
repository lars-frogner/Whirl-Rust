//! SPH smoothing kernel.

pub mod gaussian;

use crate::{geometry::dim::*, num::fvar};

/// SPH smoothing kernel.
pub trait Kernel<D: Dim> {
    /// Whether the kernel is non-zero at relative distance `q = r/h`.
    fn is_nonzero(q: fvar) -> bool;

    /// Evaluate kernel with width `h` at relative distance `q = r/h`.
    fn evaluate(&self, q: fvar, h: fvar) -> fvar;

    /// Evaluate the derivative of the kernel with width `h` at relative distance `q = r/h`.
    /// The derivative is with respect to `r`.
    fn distance_derivative(&self, q: fvar, h: fvar) -> fvar;

    /// Evaluate the derivative of the kernel with width `h` at relative distance `q = r/h`.
    /// The derivative is with respect to `h`.
    fn width_derivative(&self, q: fvar, h: fvar) -> fvar;
}
