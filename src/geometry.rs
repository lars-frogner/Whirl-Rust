//! Geometry utility objects.

use crate::{
    error::{WhirlError, WhirlResult},
    num::fvar,
};

pub mod dim {
    use crate::num::fvar;
    use nalgebra as na;

    pub type OneDim = na::U1;
    pub type TwoDim = na::U2;
    pub type ThreeDim = na::U3;

    pub type DefaultAllocator = na::DefaultAllocator;

    pub type Vector<D> = na::VectorN<fvar, D>;
    pub type Point<D> = na::Point<fvar, D>;

    pub type Vector1D = Vector<OneDim>;
    pub type Vector2D = Vector<TwoDim>;
    pub type Vector3D = Vector<ThreeDim>;
    pub type Point1D = Point<OneDim>;
    pub type Point2D = Point<TwoDim>;
    pub type Point3D = Point<ThreeDim>;

    pub trait Allocator<D: Dim>: na::allocator::Allocator<f64, D> {}

    pub trait Dim: na::DimName {
        fn dimensionality() -> Dimensionality;
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Dimensionality {
        One,
        Two,
        Three,
    }

    impl Dim for OneDim {
        fn dimensionality() -> Dimensionality {
            Dimensionality::One
        }
    }
    impl Dim for TwoDim {
        fn dimensionality() -> Dimensionality {
            Dimensionality::Two
        }
    }
    impl Dim for ThreeDim {
        fn dimensionality() -> Dimensionality {
            Dimensionality::One
        }
    }

    impl Allocator<OneDim> for DefaultAllocator {}
    impl Allocator<TwoDim> for DefaultAllocator {}
    impl Allocator<ThreeDim> for DefaultAllocator {}

    pub fn point_to_scalar(point: &Point1D) -> fvar {
        unsafe { *point.get_unchecked(0) }
    }
    pub fn vector_to_scalar(vector: &Vector1D) -> fvar {
        unsafe { *vector.get_unchecked(0) }
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
    pub fn new(lower: fvar, upper: fvar) -> WhirlResult<Self> {
        if upper <= lower {
            return Err(WhirlError::from(format!(
                "Upper bound is not strictly larger than lower bound: {} <= {}",
                upper, lower
            )));
        }
        Ok(Self { lower, upper })
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
