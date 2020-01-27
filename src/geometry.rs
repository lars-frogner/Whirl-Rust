//! Geometry utility objects.

use crate::num::fvar;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dimensionality {
    One = 1,
    Two = 2,
    Three = 3,
}

impl Dimensionality {
    /// Returns the number of dimensions.
    pub fn number(self) -> usize {
        self as usize
    }
}

/// A lower and upper bound in 1D.
#[derive(Debug, Clone)]
pub struct Bounds1D {
    lower: fvar,
    upper: fvar,
}

impl Bounds1D {
    /// Creates new bounds from the given lower and upper limits.
    pub fn new(lower: fvar, upper: fvar) -> Self {
        assert!(
            upper > lower,
            "Upper bound is not strictly larger than lower bound: {} <= {}",
            upper,
            lower
        );
        Self { lower, upper }
    }

    /// Returns the lower bound.
    pub fn lower(&self) -> fvar {
        self.lower
    }

    /// Returns the upper bound.
    pub fn upper(&self) -> fvar {
        self.upper
    }

    /// Computes the extent of the region within the bounds.
    pub fn compute_extent(&self) -> fvar {
        self.upper - self.lower
    }
}
