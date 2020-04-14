//! Gaussian smoothing kernel.

use super::Kernel;
use crate::{
    geometry::dim::*,
    num::{fvar, PI},
};
use std::marker::PhantomData;

/// Gaussian smoothing kernel.
#[derive(Clone, Debug)]
pub struct GaussianKernel<D: Dim> {
    sigma: fvar,
    _phantom: PhantomData<D>,
}

impl<D: Dim> GaussianKernel<D> {
    /// Creates a new Gaussian smoothing kernel with the given dimensionality.
    pub fn new() -> Self {
        let sigma = match D::dimensionality() {
            Dimensionality::One => 1.0 / fvar::sqrt(PI),
            Dimensionality::Two => 1.0 / PI,
            Dimensionality::Three => 1.0 / (PI * fvar::sqrt(PI)),
        };
        Self {
            sigma,
            _phantom: PhantomData,
        }
    }
}

impl<D: Dim> Kernel<D> for GaussianKernel<D> {
    fn is_nonzero(_q: fvar) -> bool {
        true
    }

    fn evaluate(&self, q: fvar, h: fvar) -> fvar {
        debug_assert!(h > 0.0, "Non-positive kernel width: {}", h);
        self.sigma * fvar::exp(-q * q) / fvar::powi(h, D::dim() as i32)
    }

    fn distance_derivative(&self, q: fvar, h: fvar) -> fvar {
        debug_assert!(h > 0.0, "Non-positive kernel width: {}", h);
        -2.0 * q * self.evaluate(q, h) / h
    }

    fn width_derivative(&self, q: fvar, h: fvar) -> fvar {
        debug_assert!(h > 0.0, "Non-positive kernel width: {}", h);
        (2.0 * q * q - (D::dim() as fvar)) * self.evaluate(q, h) / h
    }
}
