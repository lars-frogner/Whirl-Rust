import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import constants
import whirl
import whirl.experiments

simulation = whirl.experiments.sod_shock_tube(number_of_particles=1000).build()
# print(simulation.get_mass_densities())
# print(simulation.get_gas_pressures())
time_step = 1e-4
import time
start = time.time()
for _ in range(1000):
    simulation.step(time_step)
end = time.time()
print(end - start)
assert 0

fig, ax = plt.subplots()
sc = ax.scatter([], [], s=2)
time_text = ax.text(0.01, 0.01, '', transform=ax.transAxes)

xlim = (0.0, 1.0)
ylim = (0.0, 1.1)
vmin = 0.0
vmax = 1.5
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
        np.stack((simulation.get_positions(), simulation.get_mass_densities()),
                 axis=-1))
    sc.set_array(simulation.get_gas_pressures())
    sc.set_sizes(radius_to_point_area(simulation.get_kernel_widths()))
    time_text.set_text('t = {:g}'.format(0.0))
    return sc, time_text


def update(frame):
    simulation.step(time_step)
    sc.set_offsets(
        np.stack((simulation.get_positions(), simulation.get_mass_densities()),
                 axis=-1))
    sc.set_array(simulation.get_gas_pressures())
    sc.set_sizes(radius_to_point_area(simulation.get_kernel_widths()))
    time_text.set_text('t = {:g}'.format(frame * time_step))
    return sc, time_text


ani = animation.FuncAnimation(fig,
                              update,
                              frames=10000,
                              init_func=init,
                              blit=True,
                              repeat=False)
plt.show()
