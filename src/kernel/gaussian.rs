//! Gaussian smoothing kernel.

use super::Kernel;
use crate::{
    geometry::Dimensionality,
    num::{fvar, PI},
};

/// Gaussian smoothing kernel.
#[derive(Clone, Debug)]
pub struct GaussianKernel {
    dimensionality: Dimensionality,
    dimensions: i32,
    sigma: fvar,
}

impl GaussianKernel {
    /// Creates a new Gaussian smoothing kernel with the given dimensionality.
    pub fn new(dimensionality: Dimensionality) -> Self {
        let dimensions = dimensionality.number() as i32;
        let sigma = match dimensionality {
            Dimensionality::One => 1.0 / fvar::sqrt(PI),
            Dimensionality::Two => 1.0 / PI,
            Dimensionality::Three => 1.0 / (PI * fvar::sqrt(PI)),
        };
        Self {
            dimensionality,
            dimensions,
            sigma,
        }
    }
}

impl Kernel for GaussianKernel {
    fn is_nonzero(_q: fvar) -> bool {
        true
    }

    fn dimensionality(&self) -> Dimensionality {
        self.dimensionality
    }

    fn evaluate(&self, q: fvar, h: fvar) -> fvar {
        debug_assert!(h > 0.0, "Non-positive kernel width: {}", h);
        self.sigma * fvar::exp(-q * q) / fvar::powi(h, self.dimensions)
    }

    fn gradient(&self, q: fvar, h: fvar) -> fvar {
        debug_assert!(h > 0.0, "Non-positive kernel width: {}", h);
        -2.0 * q * self.sigma * fvar::exp(-q * q) / fvar::powi(h, self.dimensions + 1)
    }
}
