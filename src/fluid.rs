//! State and dynamics of a fluid.

pub mod inviscid;
pub mod inviscid_isothermal;

use crate::{eos::EquationOfState, geometry::dim::*, kernel::Kernel, num::fvar};
use itertools::izip;

/// Time derivatives of the primary fluid variables.
///
/// This does not include velocity, since it is itself a primary variable.
pub trait TimeDerivatives<D: Dim> {}

/// Positions of the fluid particles.
pub type ParticlePositions<D> = Vec<Point<D>>;

/// Velocities of the fluid particles.
pub type ParticleVelocities<D> = Vec<Vector<D>>;

pub type ParticlePositions1D = ParticlePositions<OneDim>;
pub type ParticleVelocities1D = ParticleVelocities<OneDim>;
pub type ParticlePositions2D = ParticlePositions<TwoDim>;
pub type ParticleVelocities2D = ParticleVelocities<TwoDim>;
pub type ParticlePositions3D = ParticlePositions<ThreeDim>;
pub type ParticleVelocities3D = ParticleVelocities<ThreeDim>;

/// State of a fluid.
pub trait Fluid<D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    type TimeDerivatives: TimeDerivatives<D>;

    /// Returns the number of particles making up the fluid.
    fn number_of_particles(&self) -> usize;

    /// Returns the mass of a fluid particle.
    fn particle_mass(&self) -> fvar;

    /// Returns a reference to the positions of the fluid particles.
    fn positions(&self) -> &ParticlePositions<D>;

    /// Returns a  reference to the velocities of the fluid particles.
    fn velocities(&self) -> &ParticleVelocities<D>;

    /// Returns a slice with the widths of the particle smoothing kernels.
    fn kernel_widths(&self) -> &[fvar];

    /// Returns a slice with the mass densities of the fluid particles.
    fn mass_densities(&self) -> &[fvar];

    /// Returns a slice with the energies per mass of the fluid particles.
    fn specific_energies(&self) -> &[fvar];

    /// Returns a slice with the gas pressures of the fluid particles.
    fn gas_pressures(&self) -> &[fvar];

    /// Updates properties and derivatives based on the current configuration of particles.
    fn update<EOS, K>(
        &mut self,
        equation_of_state: &EOS,
        kernel: &K,
        time_derivatives: &mut Self::TimeDerivatives,
    ) where
        EOS: EquationOfState,
        K: Kernel<D>;

    /// Evolves the primary variables for the given time step using the given linear combination
    /// of time derivatives.
    ///
    /// Position is not modified, since the velocity describing its evolution is a primary variable.
    /// Positions should be evolved separately using the `evolve_positions` method.
    fn evolve_primary_variables<'a, T>(&'a mut self, weighted_time_derivatives: T, time_step: fvar)
    where
        T: IntoIterator<Item = (fvar, &'a Self::TimeDerivatives)>;

    /// Evolves positions for the given time step using the given linear combination of velocities.
    fn evolve_positions<'a, V>(
        &'a mut self,
        current_velocity_weight: fvar,
        additional_weighted_velocities: V,
        time_step: fvar,
    ) where
        V: IntoIterator<Item = (fvar, &'a ParticleVelocities<D>)>;
}

pub fn update_particle_volumes<D: Dim, K: Kernel<D>>(
    kernel: &K,
    particle_mass: fvar,
    coupling_constant: fvar,
    tolerance: fvar,
    positions: &ParticlePositions<D>,
    kernel_widths: &mut [fvar],
    mass_densities: &mut [fvar],
    correction_scales: &mut [fvar],
) where
    DefaultAllocator: Allocator<D>,
{
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
    debug_assert_eq!(
        positions.len(),
        mass_densities.len(),
        "Inconsistent number of particles"
    );
    debug_assert_eq!(
        positions.len(),
        correction_scales.len(),
        "Inconsistent number of particles"
    );

    let dimensions = D::dim();
    let dimensions_i32 = dimensions as i32;
    let dimensions_fvar = dimensions as fvar;

    for (r_a, h_a, rho_a, omega_a) in
        izip!(positions, kernel_widths, mass_densities, correction_scales)
    {
        debug_assert!(*h_a > 0.0, "Non-positive kernel width: {}", *h_a);
        let mut h = *h_a;
        let mut rho;
        let mut omega;

        loop {
            rho = 0.0;
            omega = 0.0;

            for r_b in positions {
                let r_ab = (r_a - r_b).norm();
                let q = r_ab / h;
                if K::is_nonzero(q) {
                    rho += kernel.evaluate(q, h);
                    omega += kernel.width_derivative(q, h);
                }
            }
            rho *= particle_mass;
            omega = 1.0 + omega * particle_mass * h / (dimensions_fvar * rho);

            let zeta = particle_mass * fvar::powi(coupling_constant / h, dimensions_i32) - rho;
            let h_perturbation = fvar::min(
                fvar::max(zeta / (dimensions_fvar * rho * omega), -0.99),
                0.99,
            );

            h *= 1.0 + h_perturbation;

            if fvar::abs(h_perturbation) < tolerance {
                break;
            }
        }

        *h_a = h;
        *rho_a = rho;
        *omega_a = omega;
    }
}

pub fn compute_initial_kernel_widths<D: Dim>(
    particle_mass: fvar,
    coupling_constant: fvar,
    mass_densities: &[fvar],
) -> Vec<fvar> {
    let exponent = 1.0 / (D::dim() as fvar);
    mass_densities
        .iter()
        .map(|&rho| coupling_constant * fvar::powf(particle_mass / rho, exponent))
        .collect()
}
