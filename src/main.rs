use whirl::{
    eos::ideal_gas::{GasParticleType, IdealGasEOS},
    fluid::{isothermal::IsothermalFluid1D, Fluid},
    geometry::{Bounds1D, Dimensionality},
    initial::{
        grid::{GridMassDistribution1D, PopulatedGrid1D},
        rest::StaticVelocityDistribution1D,
    },
    kernel::gaussian::GaussianKernel,
    stepping::{euler::EulerStepper, Stepper},
    units::{si::SIUnits, Units},
};

fn main() {
    let mass_distribution = GridMassDistribution1D::new(
        1e-3,
        2.0,
        vec![PopulatedGrid1D::new(10, Bounds1D::new(0.0, 1.0))],
    );
    let velocity_distribution = StaticVelocityDistribution1D;
    let equation_of_state = IdealGasEOS::new::<SIUnits>(
        Dimensionality::One,
        GasParticleType::Monatomic,
        14.0 * SIUnits::AMU,
    );
    let kernel = GaussianKernel::new(Dimensionality::One);
    let fluid = IsothermalFluid1D::new(300.0, &kernel, &mass_distribution, &velocity_distribution);
    let mut stepper = EulerStepper::new(equation_of_state, kernel, fluid);

    let time_step = 1e-3;
    for _ in 0..10 {
        println!("{:?}", stepper.fluid().positions());
        stepper.step(time_step);
    }
}
