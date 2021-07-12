use crate::{
    eos::EquationOfState,
    geometry::dim::*,
    initial::{
        energy::InitialEnergyDistribution,
        mass::{InitialMassDistribution, MassDistributionState},
    },
    kernel::Kernel,
    num::fvar,
};
use itertools::izip;

/// Positions of the fluid particles.
pub type ParticlePositions<D> = Vec<Point<D>>;

/// Velocities of the fluid particles.
pub type ParticleVelocities<D> = Vec<Vector<D>>;

pub trait MassBehavior {
    fn compute_initial_kernel_widths<D: Dim>(
        &self,
        particle_mass: fvar,
        mass_densities: &[fvar],
    ) -> Vec<fvar>;
}

pub trait MassComponent {
    type Behavior: MassBehavior;
    type Derivatives;

    fn mass_densities(&self) -> &[fvar];

    fn kernel_widths(&self) -> &[fvar];

    fn update_particle_volumes<D, K>(&mut self, positions: &ParticlePositions<D>, kernel: &K)
    where
        D: Dim,
        K: Kernel<D>,
        DefaultAllocator: Allocator<D>;
}

pub struct AdaptiveMassBehavior {
    /// Scaling for the kernel widths (typically between 1.2 and 1.5).
    coupling_constant: fvar,
}

impl MassBehavior for AdaptiveMassBehavior {
    fn compute_initial_kernel_widths<D: Dim>(
        &self,
        particle_mass: fvar,
        mass_densities: &[fvar],
    ) -> Vec<fvar> {
        let exponent = 1.0 / (D::dim() as fvar);
        mass_densities
            .iter()
            .map(|&rho| self.coupling_constant * fvar::powf(particle_mass / rho, exponent))
            .collect()
    }
}

pub struct AdaptiveMassComponent {
    behavior: AdaptiveMassBehavior,
    particle_mass: fvar,
    kernel_widths: Vec<fvar>,
    mass_densities: Vec<fvar>,
    correction_scales: Vec<fvar>,
}

impl AdaptiveMassComponent {
    const TOLERANCE: fvar = 1e-3;

    pub fn new<D, M, K>(
        behavior: AdaptiveMassBehavior,
        mass_distribution: &M,
        kernel: &K,
    ) -> (Self, ParticlePositions<D>)
    where
        D: Dim,
        K: Kernel<D>,
        M: InitialMassDistribution<D>,
        DefaultAllocator: Allocator<D>,
    {
        let MassDistributionState {
            particle_mass,
            positions,
            kernel_widths,
            mass_densities,
        } = mass_distribution.compute_state(&behavior, kernel);
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

        let correction_scales = vec![1.0; mass_densities.len()];

        (
            Self {
                behavior,
                particle_mass,
                kernel_widths,
                mass_densities,
                correction_scales,
            },
            positions,
        )
    }
}

impl MassComponent for AdaptiveMassComponent {
    type Behavior = AdaptiveMassBehavior;
    type Derivatives = ();

    fn kernel_widths(&self) -> &[fvar] {
        &self.kernel_widths
    }

    fn mass_densities(&self) -> &[fvar] {
        &self.mass_densities
    }

    fn update_particle_volumes<D, K>(&mut self, positions: &ParticlePositions<D>, kernel: &K)
    where
        D: Dim,
        K: Kernel<D>,
        DefaultAllocator: Allocator<D>,
    {
        debug_assert_eq!(
            positions.len(),
            self.mass_densities.len(),
            "Inconsistent number of particles"
        );

        let dimensions = D::dim();
        let dimensions_i32 = dimensions as i32;
        let dimensions_fvar = dimensions as fvar;

        for (r_a, h_a, rho_a, omega_a) in izip!(
            positions,
            &self.kernel_widths,
            &self.mass_densities,
            &self.correction_scales
        ) {
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
                rho *= self.particle_mass;
                omega = 1.0 + omega * self.particle_mass * h / (dimensions_fvar * rho);

                let zeta = self.particle_mass
                    * fvar::powi(self.behavior.coupling_constant / h, dimensions_i32)
                    - rho;
                let h_perturbation = fvar::min(
                    fvar::max(zeta / (dimensions_fvar * rho * omega), -0.99),
                    0.99,
                );

                h *= 1.0 + h_perturbation;

                if fvar::abs(h_perturbation) < Self::TOLERANCE {
                    break;
                }
            }

            *h_a = h;
            *rho_a = rho;
            *omega_a = omega;
        }
    }
}

pub trait EnergyComponent {
    type Derivatives;
    fn specific_energies(&self) -> &[fvar];
}

pub struct IsothermalEnergyComponent {
    specific_energies: Vec<fvar>,
}

impl IsothermalEnergyComponent {
    pub fn new<D, M, E, EOS>(
        positions: &ParticlePositions<D>,
        mass_densities: &[fvar],
        energy_distribution: &E,
        equation_of_state: &EOS,
    ) -> Self
    where
        D: Dim,
        M: MassComponent,
        E: InitialEnergyDistribution<D>,
        EOS: EquationOfState,
    {
        Self {
            specific_energies: energy_distribution.compute_specific_energies(
                positions,
                mass_densities,
                equation_of_state,
            ),
        }
    }
}

impl EnergyComponent for IsothermalEnergyComponent {
    type Derivatives = ();
    fn specific_energies(&self) -> &[fvar] {
        &self.specific_energies
    }
}

pub struct PressureComponent {
    gas_pressures: Vec<fvar>,
}

impl PressureComponent {}

pub struct FluidDerivatives<D, M, E>
where
    D: Dim,
    M: MassComponent,
    E: EnergyComponent,
    DefaultAllocator: Allocator<D>,
{
    velocities: ParticleVelocities<D>,
    mass_state_derivatives: M::Derivatives,
    energy_state_derivatives: E::Derivatives,
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
fn update_gas_pressures<EOS: EquationOfState>(&mut self, equation_of_state: &EOS) {
    for (&rho, &u, pg) in izip!(
        &self.mass_densities,
        &self.specific_energies,
        &mut self.gas_pressures
    ) {
        *pg = equation_of_state.compute_pressure(rho, u);
    }
}

fn update_fluid<D, EOS, K, M, E>(
    equation_of_state: &EOS,
    kernel: &K,
    positions: &ParticlePositions<D>,
    mass_component: &mut M,
    energy_component: &mut E,
    time_derivatives: &mut FluidDerivatives<D, M, E>,
) where
    D: Dim,
    M: MassComponent,
    E: EnergyComponent,
    EOS: EquationOfState,
    K: Kernel<D>,
    DefaultAllocator: Allocator<D>,
{
    mass_component.update_particle_volumes(positions, kernel);

    self.update_specific_energies_and_gas_pressures(equation_of_state);

    time_derivatives.reset();

    let mean_squared_sound_speed = self.compute_mean_squared_sound_speed(equation_of_state);

    let m = self.particle_mass;

    for i in 0..self.number_of_particles() {
        let (r_a, r_upper) = self.positions.split_at(i).1.split_first().unwrap();
        let (v_a, v_upper) = self.velocities.split_at(i).1.split_first().unwrap();
        let (&h_a, h_upper) = self.kernel_widths.split_at(i).1.split_first().unwrap();
        let (&rho_a, rho_upper) = self.mass_densities.split_at(i).1.split_first().unwrap();
        let (&omega_a, omega_upper) = self.correction_scales.split_at(i).1.split_first().unwrap();
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
