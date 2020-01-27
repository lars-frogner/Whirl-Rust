//! State and dynamics of a fluid.

pub mod isothermal;

use crate::{eos::EquationOfState, kernel::Kernel, num::fvar};
use itertools::izip;

/// Positions of the fluid particles.
pub trait ParticlePositions {}

/// Velocities of the fluid particles.
pub trait ParticleVelocities {}

/// Time derivatives of the primary fluid variables.
///
/// This does not include velocity, since it is itself a primary variable.
pub trait TimeDerivatives {}

/// State of a fluid.
pub trait Fluid {
    type Positions: ParticlePositions;
    type Velocities: ParticlePositions;
    type TimeDerivatives: TimeDerivatives;

    /// Returns the number of particles making up the fluid.
    fn number_of_particles(&self) -> usize;

    /// Returns the mass of a fluid particle.
    fn particle_mass(&self) -> fvar;

    /// Returns a reference to the positions of the fluid particles.
    fn positions(&self) -> &Self::Positions;

    /// Returns a  reference to the velocities of the fluid particles.
    fn velocities(&self) -> &Self::Velocities;

    /// Returns a slice with the widths of the particle smoothing kernels.
    fn kernel_widths(&self) -> &[fvar];

    /// Returns a slice with the mass densities of the fluid particles.
    fn mass_densities(&self) -> &[fvar];

    /// Computes the time derivatives of the primary variables.
    fn compute_time_derivatives<E, K>(
        &self,
        equation_of_state: &E,
        kernel: &K,
    ) -> Self::TimeDerivatives
    where
        E: EquationOfState,
        K: Kernel;

    /// Updates properties that only depend on the current configuration of particle positions.
    fn update_static_properties<K: Kernel>(&mut self, kernel: &K);

    /// Updates the given set of time derivatives of the primary variables.
    ///
    /// `update_static_properties` should always be called prior to this method,
    /// as the time derivatives in general depend on the static properties.
    fn update_time_derivatives<E, K>(
        &self,
        equation_of_state: &E,
        kernel: &K,
        time_derivatives: &mut Self::TimeDerivatives,
    ) where
        E: EquationOfState,
        K: Kernel;

    /// Evolves the primary variables for the given time step using the given linear combination
    /// of time derivatives.
    ///
    /// Position is not modified, since the velocity describing its evolution is a primary variable.
    /// Positions should be evolved separately using the `evolve_positions` method.
    fn evolve_primary_variables<'a, D>(&'a mut self, weighted_time_derivatives: D, time_step: fvar)
    where
        D: IntoIterator<Item = (fvar, &'a Self::TimeDerivatives)>;

    /// Evolves positions for the given time step using the given linear combination of velocities.
    fn evolve_positions<'a, V>(
        &'a mut self,
        current_velocity_weight: fvar,
        additional_weighted_velocities: V,
        time_step: fvar,
    ) where
        V: IntoIterator<Item = (fvar, &'a Self::Velocities)>;
}

/// Computes the mass densities of the fluid particles with the given
/// masses, 1D positions and kernels.
pub fn compute_mass_densities_1d<K: Kernel>(
    particle_mass: fvar,
    positions: &[fvar],
    kernel_widths: &[fvar],
    kernel: &K,
) -> Vec<fvar> {
    debug_assert!(
        particle_mass > 0.0,
        "Non-positive particle mass: {}",
        particle_mass
    );
    debug_assert_eq!(
        positions.len(),
        kernel_widths.len(),
        "Inconsistent number of particles"
    );
    kernel.debug_assert_1d();

    let n = kernel_widths.len();
    let mut mass_densities = vec![0.0; n];

    for i in 0..n {
        let (&r_a, r_upper) = positions.split_at(i).1.split_first().unwrap();
        let (&h_a, h_upper) = kernel_widths.split_at(i).1.split_first().unwrap();

        let (rho_a, rho_upper) = mass_densities.split_at_mut(i).1.split_first_mut().unwrap();

        for (&r_b, &h_b, rho_b) in izip!(r_upper, h_upper, rho_upper) {
            let r_ab = r_a - r_b;
            let h_ab = 0.5 * (h_a + h_b);
            let q = r_ab / h_ab;
            if K::is_nonzero(q) {
                let w_ab = kernel.evaluate(q, h_ab);
                let rho_a_term = particle_mass * w_ab;
                *rho_a += rho_a_term;
                *rho_b += rho_a_term;
            }
        }
    }

    mass_densities
}
