//! State and dynamics of an inviscid isothermal fluid.

use super::{Fluid, ParticlePositions, ParticleVelocities, TimeDerivatives};
use crate::{
    eos::EquationOfState,
    error::WhirlResult,
    geometry::dim::*,
    initial::{
        energy::isothermal::IsothermalEnergyDistribution,
        mass::{InitialMassDistribution, MassDistributionState},
        velocity::InitialVelocityDistribution,
    },
    kernel::Kernel,
    num::fvar,
};
use itertools::izip;
use std::iter;

/// State of an inviscid isothermal fluid.
#[derive(Clone, Debug)]
pub struct InviscidIsothermalFluid<D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    particle_mass: fvar,
    coupling_constant: fvar,
    positions: ParticlePositions<D>,
    velocities: ParticleVelocities<D>,
    kernel_widths: Vec<fvar>,
    mass_densities: Vec<fvar>,
    correction_scales: Vec<fvar>,
    specific_energies: Vec<fvar>,
    gas_pressures: Vec<fvar>,
    temperature: fvar,
}

/// Time derivatives of the primary fluid variables.
#[derive(Clone, Debug)]
pub struct InviscidIsothermalTimeDerivatives<D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    dv_dt: Vec<Vector<D>>,
}

impl<D: Dim> InviscidIsothermalFluid<D>
where
    DefaultAllocator: Allocator<D>,
{
    const DENSITY_TOLERANCE: fvar = 1e-3;
    const DIFFUSION_ALPHA: fvar = 1.0;
    const DIFFUSION_BETA: fvar = 2.0;
    const DIFFUSION_EPS: fvar = 1e-3;

    /// Creates a new fluid state with the given initial mass, velocity
    /// and isothermal energy distribution.
    pub fn new<EOS, K, M, V>(
        equation_of_state: &EOS,
        kernel: &K,
        mass_distribution: &M,
        velocity_distribution: &V,
        energy_distribution: IsothermalEnergyDistribution,
    ) -> WhirlResult<(Self, <Self as Fluid<D>>::TimeDerivatives)>
    where
        EOS: EquationOfState,
        K: Kernel<D>,
        M: InitialMassDistribution<D>,
        V: InitialVelocityDistribution<D>,
        DefaultAllocator: Allocator<D>,
    {
        let mass_distribution_state = mass_distribution.compute_state(kernel);
        let velocities = velocity_distribution.compute_velocities(&mass_distribution_state);
        let temperature = energy_distribution.temperature();

        let MassDistributionState {
            particle_mass,
            coupling_constant,
            positions,
            kernel_widths,
            mass_densities,
        } = mass_distribution_state;

        let number_of_particles = mass_densities.len();

        debug_assert!(
            particle_mass > 0.0,
            "Non-positive particle mass: {}",
            particle_mass
        );
        debug_assert_eq!(
            positions.len(),
            number_of_particles,
            "Inconsistent length of state vectors"
        );
        debug_assert_eq!(
            velocities.len(),
            number_of_particles,
            "Inconsistent length of state vectors"
        );
        debug_assert_eq!(
            kernel_widths.len(),
            number_of_particles,
            "Inconsistent length of state vectors"
        );
        debug_assert_ne!(number_of_particles, 0, "No initial fluid particles");

        let correction_scales = vec![1.0; number_of_particles];
        let specific_energies = vec![0.0; number_of_particles];
        let gas_pressures = vec![0.0; number_of_particles];

        let mut fluid = Self {
            particle_mass,
            coupling_constant,
            positions,
            velocities,
            kernel_widths,
            mass_densities,
            correction_scales,
            specific_energies,
            gas_pressures,
            temperature,
        };
        let mut time_derivatives = <Self as Fluid<D>>::TimeDerivatives::new(number_of_particles);
        fluid.update(equation_of_state, kernel, &mut time_derivatives);
        Ok((fluid, time_derivatives))
    }

    /// Returns the temperature of the fluid.
    pub fn temperature(&self) -> fvar {
        self.temperature
    }

    fn update_specific_energies_and_gas_pressures<EOS: EquationOfState>(
        &mut self,
        equation_of_state: &EOS,
    ) {
        let specific_energy = equation_of_state.compute_specific_energy(self.temperature);
        for (&rho, u, pg) in izip!(
            &self.mass_densities,
            &mut self.specific_energies,
            &mut self.gas_pressures
        ) {
            *u = specific_energy;
            *pg = equation_of_state.compute_pressure(rho, specific_energy);
        }
    }

    fn compute_mean_squared_sound_speed<EOS: EquationOfState>(
        &self,
        equation_of_state: &EOS,
    ) -> fvar {
        equation_of_state.compute_squared_sound_speed(
            equation_of_state.compute_specific_energy(self.temperature),
        )
    }
}

impl<D: Dim> Fluid<D> for InviscidIsothermalFluid<D>
where
    DefaultAllocator: Allocator<D>,
{
    type TimeDerivatives = InviscidIsothermalTimeDerivatives<D>;

    fn number_of_particles(&self) -> usize {
        self.mass_densities.len()
    }

    fn particle_mass(&self) -> fvar {
        self.particle_mass
    }

    fn positions(&self) -> &ParticlePositions<D> {
        &self.positions
    }

    fn velocities(&self) -> &ParticleVelocities<D> {
        &self.velocities
    }

    fn kernel_widths(&self) -> &[fvar] {
        &self.kernel_widths
    }

    fn mass_densities(&self) -> &[fvar] {
        &self.mass_densities
    }

    fn specific_energies(&self) -> &[fvar] {
        &self.specific_energies
    }

    fn gas_pressures(&self) -> &[fvar] {
        &self.gas_pressures
    }

    fn update<EOS, K>(
        &mut self,
        equation_of_state: &EOS,
        kernel: &K,
        time_derivatives: &mut Self::TimeDerivatives,
    ) where
        EOS: EquationOfState,
        K: Kernel<D>,
    {
        super::update_particle_volumes(
            kernel,
            self.particle_mass,
            self.coupling_constant,
            Self::DENSITY_TOLERANCE,
            &self.positions,
            &mut self.kernel_widths,
            &mut self.mass_densities,
            &mut self.correction_scales,
        );

        self.update_specific_energies_and_gas_pressures(equation_of_state);

        time_derivatives.reset();

        let mean_squared_sound_speed = self.compute_mean_squared_sound_speed(equation_of_state);

        let m = self.particle_mass;

        for i in 0..self.number_of_particles() {
            let (r_a, r_upper) = self.positions.split_at(i).1.split_first().unwrap();
            let (v_a, v_upper) = self.velocities.split_at(i).1.split_first().unwrap();
            let (&h_a, h_upper) = self.kernel_widths.split_at(i).1.split_first().unwrap();
            let (&rho_a, rho_upper) = self.mass_densities.split_at(i).1.split_first().unwrap();
            let (&omega_a, omega_upper) =
                self.correction_scales.split_at(i).1.split_first().unwrap();
            let (&pg_a, pg_upper) = self.gas_pressures.split_at(i).1.split_first().unwrap();

            let (dv_a_dt, dv_dt_upper) = time_derivatives
                .dv_dt
                .split_at_mut(i)
                .1
                .split_first_mut()
                .unwrap();

            for (r_b, v_b, &h_b, &rho_b, &omega_b, &pg_b, dv_b_dt) in izip!(
                r_upper,
                v_upper,
                h_upper,
                rho_upper,
                omega_upper,
                pg_upper,
                dv_dt_upper
            ) {
                let r_ab = r_a - r_b;
                let distance = r_ab.norm();
                let q_a = distance / h_a;
                let q_b = -distance / h_b;
                if distance > 0.0 && K::is_nonzero(q_a) || K::is_nonzero(q_b) {
                    let direction = r_ab / distance;
                    let v_ab = v_a - v_b;
                    let v_ab_parallel = v_ab.dot(&direction);

                    let grad_a_w_ab_h_a = kernel.distance_derivative(q_a, h_a);
                    let grad_a_w_ab_h_b = kernel.distance_derivative(-q_b, h_b);

                    let grad_term_a = m * (pg_a / (omega_a * rho_a * rho_a)) * grad_a_w_ab_h_a;
                    let grad_term_b = m * (pg_b / (omega_b * rho_b * rho_b)) * grad_a_w_ab_h_b;

                    let h_mean = 0.5 * (h_a + h_b);
                    let rho_mean = 0.5 * (rho_a + rho_b);
                    let mu_ab = v_ab_parallel * distance * h_mean
                        / (fvar::powi(distance, 2) + Self::DIFFUSION_EPS * fvar::powi(h_mean, 2));
                    let pi_ab = if mu_ab < 0.0 {
                        (Self::DIFFUSION_BETA * mu_ab
                            - Self::DIFFUSION_ALPHA * mean_squared_sound_speed)
                            * mu_ab
                            / rho_mean
                    } else {
                        0.0
                    };
                    let grad_a_w_ab_mean = 0.5 * (grad_a_w_ab_h_a + grad_a_w_ab_h_b);
                    let diffusion_term = m * pi_ab * grad_a_w_ab_mean;

                    let dv_dt = -(grad_term_a + grad_term_b + diffusion_term) * (&direction);
                    *dv_a_dt += &dv_dt;
                    *dv_b_dt -= &dv_dt;
                }
            }
        }
    }

    fn evolve_primary_variables<'a, T>(&'a mut self, weighted_time_derivatives: T, time_step: fvar)
    where
        T: IntoIterator<Item = (fvar, &'a Self::TimeDerivatives)>,
    {
        for (weight, time_derivatives) in weighted_time_derivatives {
            debug_assert_eq!(time_derivatives.dv_dt.len(), self.number_of_particles());
            let weighted_time_step = weight * time_step;
            for (v, dv_dt) in izip!(&mut self.velocities, &time_derivatives.dv_dt) {
                *v += weighted_time_step * dv_dt;
            }
        }
    }

    fn evolve_positions<'a, V>(
        &'a mut self,
        current_velocity_weight: fvar,
        additional_weighted_velocities: V,
        time_step: fvar,
    ) where
        V: IntoIterator<Item = (fvar, &'a ParticleVelocities<D>)>,
    {
        for (coefficient, velocities) in iter::once((current_velocity_weight, &self.velocities))
            .chain(additional_weighted_velocities)
        {
            debug_assert_eq!(velocities.len(), self.number_of_particles());
            let scaled_time_step = coefficient * time_step;
            for (r, v) in izip!(&mut self.positions, velocities) {
                *r += scaled_time_step * v;
            }
        }
    }
}

impl<D: Dim> InviscidIsothermalTimeDerivatives<D>
where
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new set of time derivatives for the given number of particles.
    fn new(number_of_particles: usize) -> Self {
        Self {
            dv_dt: vec![Vector::zeros(); number_of_particles],
        }
    }

    /// Sets all derivatives to zero.
    fn reset(&mut self) {
        self.dv_dt
            .iter_mut()
            .for_each(|dv_a_dt| *dv_a_dt = Vector::zeros());
    }
}

impl<D: Dim> TimeDerivatives<D> for InviscidIsothermalTimeDerivatives<D> where
    DefaultAllocator: Allocator<D>
{
}
