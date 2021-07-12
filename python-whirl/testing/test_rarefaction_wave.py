import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import constants
import whirl
import whirl.experiments

simulation = whirl.experiments.rarefaction_wave(
    number_of_particles=1000).build()
#print(simulation.get_mass_densities())
time_step = 1e-4

fig, ax = plt.subplots()
sc = ax.scatter([], [])

xlim = (-1.0, 1.0)
ylim = (-3, 3)
vmin = 0.1
vmax = 0.95
alpha = 0.8

radius_to_point_area = lambda radius: (
    2 * radius * fig.bbox_inches.width * 72 /
    (ax.get_xlim()[1] - ax.get_xlim()[0]))**2


def init():
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    sc.set_clim(vmin, vmax)
    sc.set_alpha(alpha)
    sc.set_offsets(
        np.stack((simulation.get_positions(), simulation.get_velocities()),
                 axis=-1))
    sc.set_array(simulation.get_gas_pressures())
    sc.set_sizes(radius_to_point_area(simulation.get_kernel_widths()))
    return sc,


def update(frame):
    simulation.step(time_step)
    sc.set_offsets(
        np.stack((simulation.get_positions(), simulation.get_velocities()),
                 axis=-1))
    sc.set_array(simulation.get_gas_pressures())
    sc.set_sizes(radius_to_point_area(simulation.get_kernel_widths()))
    return sc,


ani = animation.FuncAnimation(fig,
                              update,
                              frames=10000,
                              init_func=init,
                              blit=True,
                              repeat=False)
plt.show()
