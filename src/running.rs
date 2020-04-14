//! Simulation setup and execution.

pub mod experiments;

use crate::{
    eos::adiabatic_ideal_gas::{AdiabaticIdealGasEOS, GasParticleAtomicity},
    error::{WhirlError, WhirlResult},
    fluid::{
        inviscid::InviscidFluid, inviscid_isothermal::InviscidIsothermalFluid, ParticlePositions1D,
        ParticleVelocities1D,
    },
    geometry::{dim::*, Bounds1D},
    initial::{
        energy::{
            isothermal::IsothermalEnergyDistribution,
            piecewise_uniform::PiecewiseUniformEnergyDistribution1D,
            uniform::UniformEnergyDistribution,
        },
        mass::piecewise_uniform::PiecewiseUniformMassDistribution1D,
        velocity::rest::StaticVelocityDistribution,
    },
    kernel::gaussian::GaussianKernel,
    num::fvar,
    stepping::{euler::EulerStepper, Stepper},
    units::si::SIUnits,
};

/// Builder for setting up a 1D fluid simulation.
#[derive(Clone, Debug)]
pub struct SimulationBuilder1D {
    unit_type: UnitType,
    kernel_type: KernelType,
    eos_type: Option<EquationOfStateType>,
    mass_distr_type: Option<MassDistributionType>,
    vel_distr_type: VelocityDistributionType,
    energy_distr_type: Option<EnergyDistributionType>,
    fluid_type: FluidType,
    stepper_type: StepperType,
}

/// Representation of a 1D fluid simulation.
pub struct Simulation1D {
    stepper: Box<dyn Stepper<OneDim>>,
}

impl SimulationBuilder1D {
    /// Initializes a new 1D simulation builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Use SI units.
    pub fn with_si_units(&mut self) -> &mut Self {
        self.unit_type = UnitType::SI;
        self
    }

    /// Use a Gaussian smoothing kernel.
    pub fn with_gaussian_kernel(&mut self) -> &mut Self {
        self.kernel_type = KernelType::Gaussian;
        self
    }

    /// Use an ideal adiabatic equation of state for particles with the given atomicity
    /// and mean molecular mass.
    pub fn with_adiabatic_ideal_gas_eos(
        &mut self,
        atomicity: GasParticleAtomicity,
        mean_molecular_mass: fvar,
    ) -> &mut Self {
        self.eos_type = Some(EquationOfStateType::AdiabaticIdealGas(
            AdiabaticIdealGasEOSParams {
                atomicity,
                mean_molecular_mass,
            },
        ));
        self
    }

    /// Use a ideal adiabatic equation of state, normalized so `P = rho*T`, with the given
    /// adiabatic index.
    pub fn with_normalized_adiabatic_ideal_gas_eos(&mut self, adiabatic_index: fvar) -> &mut Self {
        self.eos_type = Some(EquationOfStateType::NormAdiabaticIdealGas(
            NormAdiabaticIdealGasEOSParams { adiabatic_index },
        ));
        self
    }

    /// Use a piecewise uniform initial mass distribution.
    pub fn with_piecewise_uniform_mass_distribution(
        &mut self,
        number_of_particles: usize,
        exterior_boundary_positions: Bounds1D,
        interior_boundary_positions: Vec<fvar>,
        mass_densities: Vec<fvar>,
        coupling_constant: fvar,
    ) -> &mut Self {
        self.mass_distr_type = Some(MassDistributionType::PiecewiseUniform1D(
            PiecewiseUniformMassDistribution1DParams {
                number_of_particles,
                exterior_boundary_positions,
                interior_boundary_positions,
                mass_densities,
                coupling_constant,
            },
        ));
        self
    }

    /// Use a static initial velocity distribution.
    pub fn with_static_velocity_distribution(&mut self) -> &mut Self {
        self.vel_distr_type = VelocityDistributionType::Static;
        self
    }

    /// Use an initial uniform energy distribution with the given energy density.
    pub fn with_uniform_energy_distribution(&mut self, energy_density: fvar) -> &mut Self {
        self.energy_distr_type = Some(EnergyDistributionType::Uniform(
            UniformEnergyDistributionParams { energy_density },
        ));
        self
    }

    /// Use an initial piecewise uniform energy distribution with the given energy densities
    /// between the given boundary positions.
    pub fn with_piecewise_uniform_energy_distribution(
        &mut self,
        boundary_positions: Vec<fvar>,
        energy_densities: Vec<fvar>,
    ) -> &mut Self {
        self.energy_distr_type = Some(EnergyDistributionType::PiecewiseUniform1D(
            PiecewiseUniformEnergyDistribution1DParams {
                boundary_positions,
                energy_densities,
            },
        ));
        self
    }

    /// Use an initial isothermal energy distribution with the given temperature.
    pub fn with_isothermal_energy_distribution(&mut self, temperature: fvar) -> &mut Self {
        self.energy_distr_type = Some(EnergyDistributionType::Isothermal(
            IsothermalEnergyDistributionParams { temperature },
        ));
        self
    }

    /// Use an inviscid fluid.
    pub fn with_inviscid_fluid(&mut self) -> &mut Self {
        self.fluid_type = FluidType::Inviscid;
        self
    }

    /// Use an inviscid isothermal fluid.
    pub fn with_inviscid_isothermal_fluid(&mut self) -> &mut Self {
        self.fluid_type = FluidType::InviscidIsothermal;
        self
    }

    /// Use the Euler-Cromer scheme for time stepping.
    pub fn with_euler_stepper(&mut self) -> &mut Self {
        self.stepper_type = StepperType::Euler;
        self
    }

    /// Builds a new `Simulation1D` from the current builder configuration.
    pub fn build(&self) -> WhirlResult<Simulation1D> {
        let unit_type = self.unit_type;
        let eos_type = self
            .eos_type
            .as_ref()
            .ok_or_else(|| WhirlError::from("Equation of state has not been specified"))?
            .clone();
        let kernel_type = self.kernel_type;
        let mass_distr_type = self
            .mass_distr_type
            .as_ref()
            .ok_or_else(|| WhirlError::from("Initial mass distribution has not been specified"))?
            .clone();
        let vel_distr_type = self.vel_distr_type;
        let energy_distr_type = self
            .energy_distr_type
            .as_ref()
            .ok_or_else(|| WhirlError::from("Initial energy distribution has not been specified"))?
            .clone();
        let fluid_type = self.fluid_type;
        let stepper_type = self.stepper_type;
        Ok(Simulation1D {
            stepper: create_stepper_1d(
                unit_type,
                eos_type,
                kernel_type,
                mass_distr_type,
                vel_distr_type,
                energy_distr_type,
                fluid_type,
                stepper_type,
            )?,
        })
    }
}

impl Simulation1D {
    /// Perform a time step with the given duration.
    pub fn step(&mut self, time_step: fvar) {
        self.stepper.step(time_step);
    }

    /// Returns a reference to the positions of the fluid particles.
    pub fn positions(&self) -> &ParticlePositions1D {
        self.stepper.positions()
    }

    /// Returns a reference to the velocities of the fluid particles.
    pub fn velocities(&self) -> &ParticleVelocities1D {
        self.stepper.velocities()
    }

    /// Returns a slice with the widths of the particle smoothing kernels.
    pub fn kernel_widths(&self) -> &[fvar] {
        self.stepper.kernel_widths()
    }

    /// Returns a slice with the mass densities of the fluid particles.
    pub fn mass_densities(&self) -> &[fvar] {
        self.stepper.mass_densities()
    }

    /// Returns a slice with the energies per mass of the fluid particles.
    pub fn specific_energies(&self) -> &[fvar] {
        self.stepper.specific_energies()
    }

    /// Returns a slice with the gas pressure of the fluid particles.
    pub fn gas_pressures(&self) -> &[fvar] {
        self.stepper.gas_pressures()
    }
}

impl Default for SimulationBuilder1D {
    fn default() -> Self {
        Self {
            unit_type: UnitType::SI,
            kernel_type: KernelType::Gaussian,
            eos_type: None,
            mass_distr_type: None,
            vel_distr_type: VelocityDistributionType::Static,
            energy_distr_type: None,
            fluid_type: FluidType::Inviscid,
            stepper_type: StepperType::Euler,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum UnitType {
    SI,
}

#[derive(Clone, Debug)]
enum EquationOfStateType {
    AdiabaticIdealGas(AdiabaticIdealGasEOSParams),
    NormAdiabaticIdealGas(NormAdiabaticIdealGasEOSParams),
}

#[derive(Clone, Debug)]
struct AdiabaticIdealGasEOSParams {
    atomicity: GasParticleAtomicity,
    mean_molecular_mass: fvar,
}

#[derive(Clone, Copy, Debug)]
struct NormAdiabaticIdealGasEOSParams {
    adiabatic_index: fvar,
}

#[derive(Clone, Copy, Debug)]
enum KernelType {
    Gaussian,
}

#[derive(Clone, Debug)]
enum MassDistributionType {
    PiecewiseUniform1D(PiecewiseUniformMassDistribution1DParams),
}

enum BuiltMassDistributionType {
    PiecewiseUniform1D(PiecewiseUniformMassDistribution1D),
}

#[derive(Clone, Debug)]
struct PiecewiseUniformMassDistribution1DParams {
    number_of_particles: usize,
    exterior_boundary_positions: Bounds1D,
    interior_boundary_positions: Vec<fvar>,
    mass_densities: Vec<fvar>,
    coupling_constant: fvar,
}

#[derive(Clone, Copy, Debug)]
enum VelocityDistributionType {
    Static,
}

enum BuiltVelocityDistributionType {
    Static(StaticVelocityDistribution),
}

#[derive(Clone, Debug)]
enum EnergyDistributionType {
    Uniform(UniformEnergyDistributionParams),
    PiecewiseUniform1D(PiecewiseUniformEnergyDistribution1DParams),
    Isothermal(IsothermalEnergyDistributionParams),
}

enum BuiltEnergyDistributionType {
    Uniform(UniformEnergyDistribution),
    PiecewiseUniform1D(PiecewiseUniformEnergyDistribution1D),
    Isothermal(IsothermalEnergyDistribution),
}

#[derive(Clone, Copy, Debug)]
struct UniformEnergyDistributionParams {
    energy_density: fvar,
}

#[derive(Clone, Debug)]
struct PiecewiseUniformEnergyDistribution1DParams {
    boundary_positions: Vec<fvar>,
    energy_densities: Vec<fvar>,
}

#[derive(Clone, Copy, Debug)]
struct IsothermalEnergyDistributionParams {
    temperature: fvar,
}

#[derive(Clone, Copy, Debug)]
enum FluidType {
    Inviscid,
    InviscidIsothermal,
}

#[derive(Clone, Copy, Debug)]
enum StepperType {
    Euler,
}

macro_rules! create_inviscid_fluid {
    ($eos:expr, $kernel:expr, $mass_distr:expr, $vel_distr:expr, $energy_distr:expr) => {
        InviscidFluid::new(&$eos, &$kernel, &$mass_distr, &$vel_distr, &$energy_distr)
    };
}

macro_rules! create_inviscid_fluid_with_energy_distr {
    ($eos:expr, $kernel:expr, $mass_distr:expr, $vel_distr:expr, $built_energy_distr:expr) => {
        match $built_energy_distr {
            BuiltEnergyDistributionType::Uniform(energy_distr) => {
                create_inviscid_fluid!($eos, $kernel, $mass_distr, $vel_distr, energy_distr)
            }
            BuiltEnergyDistributionType::PiecewiseUniform1D(energy_distr) => {
                create_inviscid_fluid!($eos, $kernel, $mass_distr, $vel_distr, energy_distr)
            }
            BuiltEnergyDistributionType::Isothermal(energy_distr) => {
                create_inviscid_fluid!($eos, $kernel, $mass_distr, $vel_distr, energy_distr)
            }
            _ => Err(WhirlError::from(
                "Invalid energy distribution for inviscid fluid",
            )),
        }
    };
}

macro_rules! create_inviscid_fluid_with_vel_distr {
    ($eos:expr, $kernel:expr, $mass_distr:expr, $built_vel_distr:expr, $built_energy_distr:expr) => {
        match $built_vel_distr {
            BuiltVelocityDistributionType::Static(vel_distr) => {
                create_inviscid_fluid_with_energy_distr!(
                    $eos,
                    $kernel,
                    $mass_distr,
                    vel_distr,
                    $built_energy_distr
                )
            }
            _ => Err(WhirlError::from(
                "Invalid velocity distribution for inviscid fluid",
            )),
        }
    };
}

macro_rules! create_inviscid_fluid_with_mass_distr {
    ($eos:expr, $kernel:expr, $built_mass_distr:expr, $built_vel_distr:expr, $built_energy_distr:expr) => {
        match $built_mass_distr {
            BuiltMassDistributionType::PiecewiseUniform1D(mass_distr) => {
                create_inviscid_fluid_with_vel_distr!(
                    $eos,
                    $kernel,
                    mass_distr,
                    $built_vel_distr,
                    $built_energy_distr
                )
            }
            _ => Err(WhirlError::from(
                "Invalid mass distribution for inviscid fluid",
            )),
        }
    };
}

macro_rules! create_inviscid_isothermal_fluid {
    ($eos:expr, $kernel:expr, $mass_distr:expr, $vel_distr:expr, $energy_distr:expr) => {
        InviscidIsothermalFluid::new(&$eos, &$kernel, &$mass_distr, &$vel_distr, $energy_distr)
    };
}

macro_rules! create_inviscid_isothermal_fluid_with_energy_distr {
    ($eos:expr, $kernel:expr, $mass_distr:expr, $vel_distr:expr, $built_energy_distr:expr) => {
        match $built_energy_distr {
            BuiltEnergyDistributionType::Isothermal(energy_distr) => {
                create_inviscid_isothermal_fluid!(
                    $eos,
                    $kernel,
                    $mass_distr,
                    $vel_distr,
                    energy_distr
                )
            }
            _ => Err(WhirlError::from(
                "Invalid energy distribution for inviscid isothermal fluid",
            )),
        }
    };
}

macro_rules! create_inviscid_isothermal_fluid_with_vel_distr {
    ($eos:expr, $kernel:expr, $mass_distr:expr, $built_vel_distr:expr, $built_energy_distr:expr) => {
        match $built_vel_distr {
            BuiltVelocityDistributionType::Static(vel_distr) => {
                create_inviscid_isothermal_fluid_with_energy_distr!(
                    $eos,
                    $kernel,
                    $mass_distr,
                    vel_distr,
                    $built_energy_distr
                )
            }
            _ => Err(WhirlError::from(
                "Invalid velocity distribution for inviscid isothermal fluid",
            )),
        }
    };
}

macro_rules! create_inviscid_isothermal_fluid_with_mass_distr {
    ($eos:expr, $kernel:expr, $built_mass_distr:expr, $built_vel_distr:expr, $built_energy_distr:expr) => {
        match $built_mass_distr {
            BuiltMassDistributionType::PiecewiseUniform1D(mass_distr) => {
                create_inviscid_isothermal_fluid_with_vel_distr!(
                    $eos,
                    $kernel,
                    mass_distr,
                    $built_vel_distr,
                    $built_energy_distr
                )
            }
            _ => Err(WhirlError::from(
                "Invalid mass distribution for inviscid isothermal fluid",
            )),
        }
    };
}

macro_rules! create_boxed_stepper {
    ($return_trait_obj:ty, $dim:ty, $units:ty, $eos:expr, $kernel:expr, $fluid:expr, $time_derivatives:expr, $stepper_type:expr) => {
        match $stepper_type {
            StepperType::Euler => {
                Ok(
                    Box::new(EulerStepper::new($eos, $kernel, $fluid, $time_derivatives))
                        as Box<$return_trait_obj>,
                )
            }
        }
    };
}

macro_rules! create_stepper_with_fluid {
    ($return_trait_obj:ty, $dim:ty, $units:ty, $eos:expr, $kernel:expr, $built_mass_distr:expr, $built_vel_distr:expr, $built_energy_distr:expr, $fluid_type:expr, $stepper_type:expr) => {
        match $fluid_type {
            FluidType::Inviscid => {
                let (fluid, time_derivatives) = create_inviscid_fluid_with_mass_distr!(
                    $eos,
                    $kernel,
                    $built_mass_distr,
                    $built_vel_distr,
                    $built_energy_distr
                )?;
                create_boxed_stepper!(
                    $return_trait_obj,
                    $dim,
                    $units,
                    $eos,
                    $kernel,
                    fluid,
                    time_derivatives,
                    $stepper_type
                )
            }
            FluidType::InviscidIsothermal => {
                let (fluid, time_derivatives) = create_inviscid_isothermal_fluid_with_mass_distr!(
                    $eos,
                    $kernel,
                    $built_mass_distr,
                    $built_vel_distr,
                    $built_energy_distr
                )?;
                create_boxed_stepper!(
                    $return_trait_obj,
                    $dim,
                    $units,
                    $eos,
                    $kernel,
                    fluid,
                    time_derivatives,
                    $stepper_type
                )
            }
        }
    };
}

macro_rules! create_stepper_with_energy_distr {
    ($return_trait_obj:ty, $dim:ty, $units:ty, $eos:expr, $kernel:expr, $built_mass_distr:expr, $built_vel_distr:expr, $energy_distr_type:expr, $fluid_type:expr, $stepper_type:expr) => {
        match $energy_distr_type {
            EnergyDistributionType::Uniform(params) => create_stepper_with_fluid!(
                $return_trait_obj,
                $dim,
                $units,
                $eos,
                $kernel,
                $built_mass_distr,
                $built_vel_distr,
                BuiltEnergyDistributionType::Uniform(UniformEnergyDistribution::new(
                    params.energy_density
                )?),
                $fluid_type,
                $stepper_type
            ),
            EnergyDistributionType::PiecewiseUniform1D(params) => create_stepper_with_fluid!(
                $return_trait_obj,
                $dim,
                $units,
                $eos,
                $kernel,
                $built_mass_distr,
                $built_vel_distr,
                BuiltEnergyDistributionType::PiecewiseUniform1D(
                    PiecewiseUniformEnergyDistribution1D::new(
                        params.boundary_positions,
                        params.energy_densities
                    )?
                ),
                $fluid_type,
                $stepper_type
            ),
            EnergyDistributionType::Isothermal(params) => create_stepper_with_fluid!(
                $return_trait_obj,
                $dim,
                $units,
                $eos,
                $kernel,
                $built_mass_distr,
                $built_vel_distr,
                BuiltEnergyDistributionType::Isothermal(IsothermalEnergyDistribution::new(
                    params.temperature
                )?),
                $fluid_type,
                $stepper_type
            ),
        }
    };
}

macro_rules! create_stepper_with_vel_distr {
    ($return_trait_obj:ty, $dim:ty, $units:ty, $eos:expr, $kernel:expr, $built_mass_distr:expr, $vel_distr_type:expr, $energy_distr_type:expr, $fluid_type:expr, $stepper_type:expr) => {
        match $vel_distr_type {
            VelocityDistributionType::Static => create_stepper_with_energy_distr!(
                $return_trait_obj,
                $dim,
                $units,
                $eos,
                $kernel,
                $built_mass_distr,
                BuiltVelocityDistributionType::Static(StaticVelocityDistribution),
                $energy_distr_type,
                $fluid_type,
                $stepper_type
            ),
        }
    };
}

macro_rules! create_stepper_with_mass_distr {
    ($return_trait_obj:ty, $dim:ty, $units:ty, $eos:expr, $kernel:expr, $mass_distr_type:expr, $vel_distr_type:expr, $energy_distr_type:expr, $fluid_type:expr, $stepper_type:expr) => {
        match $mass_distr_type {
            MassDistributionType::PiecewiseUniform1D(params) => create_stepper_with_vel_distr!(
                $return_trait_obj,
                $dim,
                $units,
                $eos,
                $kernel,
                BuiltMassDistributionType::PiecewiseUniform1D(
                    PiecewiseUniformMassDistribution1D::new(
                        params.number_of_particles,
                        params.exterior_boundary_positions,
                        params.interior_boundary_positions,
                        params.mass_densities,
                        params.coupling_constant,
                    )?
                ),
                $vel_distr_type,
                $energy_distr_type,
                $fluid_type,
                $stepper_type
            ),
        }
    };
}

macro_rules! create_stepper_with_kernel {
    ($return_trait_obj:ty, $dim:ty, $units:ty, $eos:expr, $kernel_type:expr, $mass_distr_type:expr, $vel_distr_type:expr, $energy_distr_type:expr, $fluid_type:expr, $stepper_type:expr) => {
        match $kernel_type {
            KernelType::Gaussian => create_stepper_with_mass_distr!(
                $return_trait_obj,
                $dim,
                $units,
                $eos,
                GaussianKernel::new(),
                $mass_distr_type,
                $vel_distr_type,
                $energy_distr_type,
                $fluid_type,
                $stepper_type
            ),
        }
    };
}

macro_rules! create_stepper_with_eos {
    ($return_trait_obj:ty, $dim:ty, $units:ty, $eos_type:expr, $kernel_type:expr, $mass_distr_type:expr, $vel_distr_type:expr, $energy_distr_type:expr, $fluid_type:expr, $stepper_type:expr) => {
        match $eos_type {
            EquationOfStateType::AdiabaticIdealGas(params) => create_stepper_with_kernel!(
                $return_trait_obj,
                $dim,
                $units,
                AdiabaticIdealGasEOS::new::<$dim, $units>(
                    params.atomicity,
                    params.mean_molecular_mass
                )?,
                $kernel_type,
                $mass_distr_type,
                $vel_distr_type,
                $energy_distr_type,
                $fluid_type,
                $stepper_type
            ),
            EquationOfStateType::NormAdiabaticIdealGas(params) => create_stepper_with_kernel!(
                $return_trait_obj,
                $dim,
                $units,
                AdiabaticIdealGasEOS::new_normalized(params.adiabatic_index)?,
                $kernel_type,
                $mass_distr_type,
                $vel_distr_type,
                $energy_distr_type,
                $fluid_type,
                $stepper_type
            ),
        }
    };
}

macro_rules! create_stepper_with_units {
    ($return_trait_obj:ty, $dim:ty, $unit_type:expr, $eos_type:expr, $kernel_type:expr, $mass_distr_type:expr, $vel_distr_type:expr, $energy_distr_type:expr, $fluid_type:expr, $stepper_type:expr) => {
        match $unit_type {
            UnitType::SI => create_stepper_with_eos!(
                $return_trait_obj,
                $dim,
                SIUnits,
                $eos_type,
                $kernel_type,
                $mass_distr_type,
                $vel_distr_type,
                $energy_distr_type,
                $fluid_type,
                $stepper_type
            ),
        }
    };
}

#[allow(unreachable_patterns)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::cognitive_complexity)]
fn create_stepper_1d(
    unit_type: UnitType,
    eos_type: EquationOfStateType,
    kernel_type: KernelType,
    mass_distr_type: MassDistributionType,
    vel_distr_type: VelocityDistributionType,
    energy_distr_type: EnergyDistributionType,
    fluid_type: FluidType,
    stepper_type: StepperType,
) -> WhirlResult<Box<dyn Stepper<OneDim>>>
where
{
    create_stepper_with_units!(
        dyn Stepper<OneDim>,
        OneDim,
        unit_type,
        eos_type,
        kernel_type,
        mass_distr_type,
        vel_distr_type,
        energy_distr_type,
        fluid_type,
        stepper_type
    )
}
