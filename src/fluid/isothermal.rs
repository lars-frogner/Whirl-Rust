//! State and dynamics of an isothermal fluid.

use super::{Fluid, ParticlePositions, ParticleVelocities, TimeDerivatives};
use crate::{
    eos::EquationOfState,
    initial::{InitialMassDistribution1D, InitialVelocityDistribution1D, MassDistributionState1D},
    kernel::Kernel,
    num::fvar,
};
use itertools::izip;
use std::iter;

/// State of an isothermal fluid in 1D.
#[derive(Clone, Debug)]
pub struct IsothermalFluid1D {
    temperature: fvar,
    particle_mass: fvar,
    positions: Vec<fvar>,
    velocities: Vec<fvar>,
    kernel_widths: Vec<fvar>,
    mass_densities: Vec<fvar>,
}

/// Time derivatives of the primary isothermal fluid variables in 1D.
#[derive(Clone, Debug)]
pub struct IsothermalTimeDerivatives1D {
    dv_dt: Vec<fvar>,
    drho_dt: Vec<fvar>,
}

impl IsothermalFluid1D {
    /// Creates a new isothermal fluid state with the given temperature and initial mass
    /// and velocity distribution.
    pub fn new<K, M, V>(
        temperature: fvar,
        kernel: &K,
        mass_distribution: &M,
        velocity_distribution: &V,
    ) -> Self
    where
        K: Kernel,
        M: InitialMassDistribution1D,
        V: InitialVelocityDistribution1D,
    {
        kernel.assert_1d();
        assert!(
            temperature > 0.0,
            "Non-positive temperature: {}",
            temperature
        );

        let mass_distribution_state = mass_distribution.compute_state(kernel);
        let velocities = velocity_distribution.compute_velocities(&mass_distribution_state);

        let MassDistributionState1D {
            particle_mass,
            positions,
            kernel_widths,
            mass_densities,
        } = mass_distribution_state;

        debug_assert!(
            particle_mass > 0.0,
            "Non-positive particle mass: {}",
            particle_mass
        );
        debug_assert_eq!(
            positions.len(),
            mass_densities.len(),
            "Inconsistent length of state vectors"
        );
        debug_assert_eq!(
            velocities.len(),
            mass_densities.len(),
            "Inconsistent length of state vectors"
        );
        debug_assert_eq!(
            kernel_widths.len(),
            mass_densities.len(),
            "Inconsistent length of state vectors"
        );
        debug_assert_ne!(mass_densities.len(), 0, "No initial fluid particles");

        let mut fluid = Self {
            temperature,
            particle_mass,
            positions,
            velocities,
            kernel_widths,
            mass_densities,
        };
        fluid.update_static_properties(kernel);
        fluid
    }

    fn compute_pressure<E: EquationOfState>(
        &self,
        equation_of_state: &E,
        mass_density: fvar,
    ) -> fvar {
        equation_of_state.compute_pressure(
            equation_of_state.compute_energy_density(mass_density, self.temperature),
        )
    }
}

impl Fluid for IsothermalFluid1D {
    type Positions = Vec<fvar>;
    type Velocities = Vec<fvar>;
    type TimeDerivatives = IsothermalTimeDerivatives1D;

    fn number_of_particles(&self) -> usize {
        self.mass_densities.len()
    }

    fn particle_mass(&self) -> fvar {
        self.particle_mass
    }

    fn positions(&self) -> &Self::Positions {
        &self.positions
    }

    fn velocities(&self) -> &Self::Velocities {
        &self.velocities
    }

    fn kernel_widths(&self) -> &[fvar] {
        &self.kernel_widths
    }

    fn mass_densities(&self) -> &[fvar] {
        &self.mass_densities
    }

    fn compute_time_derivatives<E, K>(
        &self,
        equation_of_state: &E,
        kernel: &K,
    ) -> Self::TimeDerivatives
    where
        E: EquationOfState,
        K: Kernel,
    {
        kernel.assert_1d();
        let n = self.number_of_particles();
        let mut time_derivatives = Self::TimeDerivatives {
            dv_dt: vec![0.0; n],
            drho_dt: vec![0.0; n],
        };
        self.update_time_derivatives(equation_of_state, kernel, &mut time_derivatives);
        time_derivatives
    }

    fn update_static_properties<K: Kernel>(&mut self, _kernel: &K) {}

    fn update_time_derivatives<E, K>(
        &self,
        equation_of_state: &E,
        kernel: &K,
        time_derivatives: &mut Self::TimeDerivatives,
    ) where
        E: EquationOfState,
        K: Kernel,
    {
        time_derivatives.reset();

        let m = self.particle_mass;

        for i in 0..self.number_of_particles() {
            let (&r_a, r_upper) = self.positions.split_at(i).1.split_first().unwrap();
            let (&v_a, v_upper) = self.velocities.split_at(i).1.split_first().unwrap();
            let (&h_a, h_upper) = self.kernel_widths.split_at(i).1.split_first().unwrap();
            let (&rho_a, rho_upper) = self.mass_densities.split_at(i).1.split_first().unwrap();

            let pg_a = self.compute_pressure(equation_of_state, rho_a);

            let (dv_a_dt, dv_dt_upper) = time_derivatives
                .dv_dt
                .split_at_mut(i)
                .1
                .split_first_mut()
                .unwrap();
            let (drho_a_dt, drho_dt_upper) = time_derivatives
                .drho_dt
                .split_at_mut(i)
                .1
                .split_first_mut()
                .unwrap();

            for (&r_b, &v_b, &h_b, &rho_b, dv_b_dt, drho_b_dt) in izip!(
                r_upper,
                v_upper,
                h_upper,
                rho_upper,
                dv_dt_upper,
                drho_dt_upper,
            ) {
                let r_ab = r_a - r_b;
                let h_ab = 0.5 * (h_a + h_b);
                let q = r_ab / h_ab;
                if K::is_nonzero(q) {
                    let pg_b = self.compute_pressure(equation_of_state, rho_b);
                    let v_ab = v_a - v_b;
                    let grad_a_w_ab = kernel.gradient(q, h_ab);

                    let dv_a_dt_term =
                        -m * (pg_a / (rho_a * rho_a) + pg_b / (rho_b * rho_b)) * grad_a_w_ab;

                    *dv_a_dt += dv_a_dt_term;
                    *dv_b_dt -= dv_a_dt_term;

                    let drho_a_dt_term = m * v_ab * grad_a_w_ab;

                    *drho_a_dt += drho_a_dt_term;
                    *drho_b_dt += drho_a_dt_term;
                }
            }
        }
    }

    fn evolve_primary_variables<'a, D>(&'a mut self, weighted_time_derivatives: D, time_step: fvar)
    where
        D: IntoIterator<Item = (fvar, &'a Self::TimeDerivatives)>,
    {
        for (weight, time_derivatives) in weighted_time_derivatives {
            debug_assert_eq!(time_derivatives.dv_dt.len(), self.number_of_particles());
            let weighted_time_step = weight * time_step;
            for (v, rho, dv_dt, drho_dt) in izip!(
                &mut self.velocities,
                &mut self.mass_densities,
                &time_derivatives.dv_dt,
                &time_derivatives.drho_dt
            ) {
                *v += weighted_time_step * dv_dt;
                *rho += weighted_time_step * drho_dt;
            }
        }
    }

    fn evolve_positions<'a, V>(
        &'a mut self,
        current_velocity_weight: fvar,
        additional_weighted_velocities: V,
        time_step: fvar,
    ) where
        V: IntoIterator<Item = (fvar, &'a Self::Velocities)>,
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

impl IsothermalTimeDerivatives1D {
    /// Sets all derivatives to zero.
    fn reset(&mut self) {
        self.dv_dt.iter_mut().for_each(|dv_a_dt| *dv_a_dt = 0.0);
        self.drho_dt
            .iter_mut()
            .for_each(|drho_a_dt| *drho_a_dt = 0.0);
    }
}

impl ParticlePositions for Vec<fvar> {}
impl ParticleVelocities for Vec<fvar> {}
impl TimeDerivatives for IsothermalTimeDerivatives1D {}
