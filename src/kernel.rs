//! SPH smoothing kernel.

pub mod gaussian;

use crate::{geometry::Dimensionality, num::fvar};

/// SPH smoothing kernel.
pub trait Kernel {
    /// Whether the kernel is non-zero at relative distance `q = r/h`.
    fn is_nonzero(q: fvar) -> bool;

    /// Returns the dimensionality of the kernel;
    fn dimensionality(&self) -> Dimensionality;

    /// Evaluate kernel with width `h` at relative distance `q = r/h`.
    fn evaluate(&self, q: fvar, h: fvar) -> fvar;

    /// Evaluate the gradient of the kernel with width `h` at relative distance `q = r/h`.
    /// The gradient is with respect to `r`.
    fn gradient(&self, q: fvar, h: fvar) -> fvar;

    /// Panics if the kernel is not one-dimensional.
    fn assert_1d(&self) {
        assert_eq!(
            self.dimensionality(),
            Dimensionality::One,
            "Kernel is not one-dimensional"
        );
    }

    /// Panics if the kernel is not two-dimensional.
    fn assert_2d(&self) {
        assert_eq!(
            self.dimensionality(),
            Dimensionality::Two,
            "Kernel is not two-dimensional"
        );
    }

    /// Panics if the kernel is not three-dimensional.
    fn assert_3d(&self) {
        assert_eq!(
            self.dimensionality(),
            Dimensionality::Three,
            "Kernel is not three-dimensional"
        );
    }

    /// Panics if the kernel is not one-dimensional (only in non-optimized builds).
    fn debug_assert_1d(&self) {
        debug_assert_eq!(
            self.dimensionality(),
            Dimensionality::One,
            "Kernel is not one-dimensional"
        );
    }

    /// Panics if the kernel is not two-dimensional (only in non-optimized builds).
    fn debug_assert_2d(&self) {
        debug_assert_eq!(
            self.dimensionality(),
            Dimensionality::Two,
            "Kernel is not two-dimensional"
        );
    }

    /// Panics if the kernel is not three-dimensional (only in non-optimized builds).
    fn debug_assert_3d(&self) {
        debug_assert_eq!(
            self.dimensionality(),
            Dimensionality::Three,
            "Kernel is not three-dimensional"
        );
    }
}
