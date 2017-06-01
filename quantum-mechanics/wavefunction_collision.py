#################################################
# Wavefunction collision                        #
# ============================================= #
# Simulates the collision of wavefunctions in a #
# harmonic oscillator potential.                #
#################################################
import matplotlib
matplotlib.use("TkAgg") # fix things for macOS

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the x space (and the correspondent p space)
x_min = -20.0
x_max = +20.0
num_samples = 1024
dx = (x_max - x_min)/num_samples
dp = 2*np.pi/(x_max - x_min)

x = np.arange(start=x_min, stop=x_max, step=dx)
x = np.linspace(x_min + dx/2, x_max - dx/2, num_samples)
p = np.fft.fftfreq(num_samples, 1./(num_samples*dp))

# Potential and kinetic energy
V = 0.5 * x**2
K = 0.5 * p**2

# We use two gaussian wave packets
x_1 = -10.0
x_2 = +5.0
sigma_1 = 1.0
sigma_2 = 2.0

# Don't care about normalization since is enforced later
psi_1 = np.exp(-(x-x_1)**2/(2.0*sigma_1**2)) + 0j
psi_2 = np.exp(-(x-x_2)**2/(2.0*sigma_2**2)) + 0j

def normalize(wavefunction):
    return wavefunction/np.sqrt(np.vdot(wavefunction, wavefunction)*dx)

# Total wavefunction
psi = normalize(psi_1 + psi_2)

# Time evolution operator
# I order Trotter's splitting
def time_evolver_1st_order(wavefunction, K, V, timestep):
    wavefunction_p = np.fft.fft(wavefunction)
    wavefunction_p = wavefunction_p * np.exp(-1j*K*timestep)
    wavefunction_x = np.fft.ifft(wavefunction_p)
    wavefunction_x = wavefunction_x * np.exp(-1j*V*timestep)

    return wavefunction_x

# II order Trotter's splitting
def time_evolver(wavefunction, K, V, timestep):
    wavefunction_x = wavefunction * np.exp(-0.5j*V*timestep)
    wavefunction_p = np.fft.fft(wavefunction_x)
    wavefunction_p = wavefunction_p * np.exp(-1j*K*timestep)
    wavefunction_x = np.fft.ifft(wavefunction_p)
    wavefunction_x = wavefunction_x * np.exp(-0.5j*V*timestep)

    return wavefunction_x


dt = 0.001

# Define plot & animation
fig,  ax = plt.subplots()
psi_line, = ax.plot(x, np.abs(psi)**2)

# Plot the potential (rescaled)
ax.plot(x, V/max(V), "--")


def animate_plot(step):
    # Evolve psi for 20 steps
    for i in range(20):
        animate_plot.psi = time_evolver(animate_plot.psi, K, V, dt)

    # Redraw psi in the plot
    psi_line.set_data(x, np.abs(animate_plot.psi)**2)

    return psi_line,

animate_plot.psi = psi

animation = FuncAnimation(fig, animate_plot, 250, interval=40, blit=True)

plt.show()
