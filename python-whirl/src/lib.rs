#![feature(trivial_bounds)]
#![allow(trivial_bounds)]

use numpy::{PyArray1, ToPyArray};
use pyo3::{exceptions::ValueError, prelude::*, wrap_pyfunction};
use whirl::{
    eos::adiabatic_ideal_gas::GasParticleAtomicity,
    geometry::{dim::*, Bounds1D},
    num::fvar,
    running::{experiments, Simulation1D, SimulationBuilder1D},
};

/// Builder for setting up a 1D fluid simulation.
#[pyclass(name=SimulationBuilder1D)]
#[text_signature = "()"]
pub struct PySimulationBuilder1D {
    builder: SimulationBuilder1D,
}

/// Representation of a 1D fluid simulation.
#[pyclass(name=Simulation1D)]
pub struct PySimulation1D {
    simulation: Simulation1D,
}

#[pymethods]
impl PySimulationBuilder1D {
    /// Initializes a new 1D simulation builder.
    #[allow(clippy::new_ret_no_self)]
    #[new]
    pub fn new(init: &PyRawObject) {
        init.init(Self {
            builder: SimulationBuilder1D::new(),
        })
    }

    /// Use SI units.
    #[text_signature = "($self)"]
    pub fn with_si_units<'a>(mut slf: PyRefMut<'a, Self>, _py: Python) -> PyRefMut<'a, Self> {
        slf.builder.with_si_units();
        slf
    }

    /// Use a Gaussian smoothing kernel.
    #[text_signature = "($self)"]
    pub fn with_gaussian_kernel<'a>(
        mut slf: PyRefMut<'a, Self>,
        _py: Python,
    ) -> PyRefMut<'a, Self> {
        slf.builder.with_gaussian_kernel();
        slf
    }

    /// Use an ideal adiabatic equation of state for particles with the given atomicity
    /// and mean molecular mass.
    #[args(atomicity = "\"monatomic\"")]
    #[text_signature = "($self, mean_molecular_mass, / atomicity=\"monatomic\")"]
    pub fn with_adiabatic_ideal_gas_eos<'a>(
        mut slf: PyRefMut<'a, Self>,
        _py: Python,
        mean_molecular_mass: fvar,
        atomicity: &str,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let atomicity = match atomicity {
            "monatomic" => GasParticleAtomicity::Monatomic,
            "diatomic" => GasParticleAtomicity::Diatomic,
            other => {
                return Err(ValueError::py_err(format!(
                "Argument `atomicity` is \"{}\" but must be one of [\"monatomic\", \"diatomic\"]",
                other
            )))
            }
        };
        slf.builder
            .with_adiabatic_ideal_gas_eos(atomicity, mean_molecular_mass);
        Ok(slf)
    }

    /// Use a ideal adiabatic equation of state, normalized so `P = rho*T`, with the given
    /// adiabatic index.
    #[text_signature = "($self, adiabatic_index, /)"]
    pub fn with_normalized_adiabatic_ideal_gas_eos<'a>(
        mut slf: PyRefMut<'a, Self>,
        _py: Python,
        adiabatic_index: fvar,
    ) -> PyResult<PyRefMut<'a, Self>> {
        slf.builder
            .with_normalized_adiabatic_ideal_gas_eos(adiabatic_index);
        Ok(slf)
    }

    /// Use a piecewise uniform initial mass distribution.
    #[text_signature = "($self, number_of_particles, exterior_boundary_positions, interior_boundary_positions, mass_densities, coupling_constant, /)"]
    pub fn with_piecewise_uniform_mass_distribution<'a>(
        mut slf: PyRefMut<'a, Self>,
        _py: Python,
        number_of_particles: usize,
        exterior_boundary_positions: (fvar, fvar),
        interior_boundary_positions: Vec<fvar>,
        mass_densities: Vec<fvar>,
        coupling_constant: fvar,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let exterior_boundary_positions =
            Bounds1D::new(exterior_boundary_positions.0, exterior_boundary_positions.1)
                .map_err(|err| ValueError::py_err(err.into_string()))?;
        slf.builder.with_piecewise_uniform_mass_distribution(
            number_of_particles,
            exterior_boundary_positions,
            interior_boundary_positions,
            mass_densities,
            coupling_constant,
        );
        Ok(slf)
    }

    /// Use a static initial velocity distribution.
    #[text_signature = "($self)"]
    pub fn with_static_velocity_distribution<'a>(
        mut slf: PyRefMut<'a, Self>,
        _py: Python,
    ) -> PyRefMut<'a, Self> {
        slf.builder.with_static_velocity_distribution();
        slf
    }

    /// Use an initial uniform energy distribution with the given energy density.
    #[text_signature = "($self, energy_density, /)"]
    pub fn with_uniform_energy_distribution<'a>(
        mut slf: PyRefMut<'a, Self>,
        _py: Python,
        energy_density: fvar,
    ) -> PyResult<PyRefMut<'a, Self>> {
        slf.builder.with_uniform_energy_distribution(energy_density);
        Ok(slf)
    }

    /// Use an initial piecewise uniform energy distribution with the given energy density.
    #[text_signature = "($self, boundary_positions, energy_densities, /)"]
    pub fn with_piecewise_uniform_energy_distribution<'a>(
        mut slf: PyRefMut<'a, Self>,
        _py: Python,
        boundary_positions: Vec<fvar>,
        energy_densities: Vec<fvar>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        slf.builder
            .with_piecewise_uniform_energy_distribution(boundary_positions, energy_densities);
        Ok(slf)
    }

    /// Use an initial isothermal energy distribution with the given temperature.
    #[text_signature = "($self, temperature, /)"]
    pub fn with_isothermal_energy_distribution<'a>(
        mut slf: PyRefMut<'a, Self>,
        _py: Python,
        temperature: fvar,
    ) -> PyResult<PyRefMut<'a, Self>> {
        slf.builder.with_isothermal_energy_distribution(temperature);
        Ok(slf)
    }

    /// Use an inviscid fluid.
    #[text_signature = "($self)"]
    pub fn with_inviscid_fluid<'a>(mut slf: PyRefMut<'a, Self>, _py: Python) -> PyRefMut<'a, Self> {
        slf.builder.with_inviscid_fluid();
        slf
    }

    /// Use an inviscid isothermal fluid.
    #[text_signature = "($self)"]
    pub fn with_inviscid_isothermal_fluid<'a>(
        mut slf: PyRefMut<'a, Self>,
        _py: Python,
    ) -> PyRefMut<'a, Self> {
        slf.builder.with_inviscid_isothermal_fluid();
        slf
    }

    /// Use the Euler-Cromer scheme for time stepping.
    #[text_signature = "($self)"]
    pub fn with_euler_stepper<'a>(
        mut slf: PyRefMut<'a, Self>,
        _py: Python,
    ) -> PyResult<PyRefMut<'a, Self>> {
        slf.builder.with_euler_stepper();
        Ok(slf)
    }

    /// Builds a new `Simulation1D` from the current builder configuration.
    #[text_signature = "($self)"]
    pub fn build(&self) -> PyResult<PySimulation1D> {
        Ok(PySimulation1D {
            simulation: self
                .builder
                .build()
                .map_err(|err| ValueError::py_err(err.into_string()))?,
        })
    }
}

#[pymethods]
impl PySimulation1D {
    /// Perform a time step with the given duration.
    #[text_signature = "($self, time_step, /)"]
    pub fn step(&mut self, time_step: fvar) {
        self.simulation.step(time_step);
    }

    /// Returns an array with the current positions of the fluid particles.
    #[text_signature = "($self)"]
    pub fn get_positions<'py>(&self, py: Python<'py>) -> &'py PyArray1<fvar> {
        self.simulation
            .positions()
            .iter()
            .map(|point| point_to_scalar(point))
            .collect::<Vec<_>>()
            .to_pyarray(py)
    }

    /// Returns an array with the current velocities of the fluid particles.
    #[text_signature = "($self)"]
    pub fn get_velocities<'py>(&self, py: Python<'py>) -> &'py PyArray1<fvar> {
        self.simulation
            .velocities()
            .iter()
            .map(|vector| vector_to_scalar(vector))
            .collect::<Vec<_>>()
            .to_pyarray(py)
    }

    /// Returns an array with the current widths of the particle smoothing kernels.
    #[text_signature = "($self)"]
    pub fn get_kernel_widths<'py>(&self, py: Python<'py>) -> &'py PyArray1<fvar> {
        self.simulation.kernel_widths().to_pyarray(py)
    }

    /// Returns an array with the current mass densities of the fluid particles.
    #[text_signature = "($self)"]
    pub fn get_mass_densities<'py>(&self, py: Python<'py>) -> &'py PyArray1<fvar> {
        self.simulation.mass_densities().to_pyarray(py)
    }

    /// Returns an array with the current specific energies of the fluid particles.
    #[text_signature = "($self)"]
    pub fn get_specific_energies<'py>(&self, py: Python<'py>) -> &'py PyArray1<fvar> {
        self.simulation.specific_energies().to_pyarray(py)
    }

    /// Returns an array with the current gas pressure of the fluid particles.
    #[text_signature = "($self)"]
    pub fn get_gas_pressures<'py>(&self, py: Python<'py>) -> &'py PyArray1<fvar> {
        self.simulation.gas_pressures().to_pyarray(py)
    }
}

#[pyfunction(
    initial_width = "1.0",
    initial_mass_density = "1.0",
    initial_energy_per_mass = "2.0",
    number_of_particles = "200",
    adiabatic_index = "1.4"
)]
/// Creates a `SimulationBuilder1D` for the rarefaction wave experiment.
#[text_signature = "(initial_width=1.0, initial_mass_density=1.0, initial_energy_per_mass=2.0, number_of_particles=200, adiabatic_index=1.4)"]
pub fn rarefaction_wave(
    initial_width: fvar,
    initial_mass_density: fvar,
    initial_energy_per_mass: fvar,
    number_of_particles: usize,
    adiabatic_index: fvar,
) -> PyResult<PySimulationBuilder1D> {
    Ok(PySimulationBuilder1D {
        builder: experiments::rarefaction_wave(
            initial_width,
            initial_mass_density,
            initial_energy_per_mass,
            number_of_particles,
            adiabatic_index,
        )
        .map_err(|err| ValueError::py_err(err.into_string()))?,
    })
}

#[pyfunction(number_of_particles = "200", adiabatic_index = "1.4")]
/// Creates a `SimulationBuilder1D` for the Sod shock tube experiment.
#[text_signature = "(number_of_particles=200, adiabatic_index=1.4)"]
pub fn sod_shock_tube(
    number_of_particles: usize,
    adiabatic_index: fvar,
) -> PyResult<PySimulationBuilder1D> {
    Ok(PySimulationBuilder1D {
        builder: experiments::sod_shock_tube(number_of_particles, adiabatic_index)
            .map_err(|err| ValueError::py_err(err.into_string()))?,
    })
}

#[pymodule]
pub fn whirl(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<PySimulationBuilder1D>()?;
    module.add_class::<PySimulation1D>()?;

    module.add_wrapped(wrap_pyfunction!(rarefaction_wave))?;
    module.add_wrapped(wrap_pyfunction!(sod_shock_tube))?;
    Ok(())
}
