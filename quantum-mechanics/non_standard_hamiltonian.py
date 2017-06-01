# coding=utf-8
"""
################################################################################
# Non standard potential Hamiltonian analysis                                  #
# ============================================================================ #
# Finds eigenvalues & spectrum of a Hamiltonian with a non standard potential  #
# (e.g. x**2.5), using exact diagonalization and imaginary time evolution.     #
# Then simulates the time evolution of the ground state wavefunction displaced #
# with respect to the equilibrium position.                                    #
################################################################################
"""
import matplotlib
matplotlib.use("TkAgg")  # fix things for macOS

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Simulation:
    """
    A Simulation represents the numerical framework in which we are operating:
    the space (in x and k), the Hamiltonian, and all the relative helpers (e.g.
    normalization, energy evaluation, etc.).

    Attributes:
        K: The kinetic energy operator (in the x basis)
        K_k: The kinetic energy operator (in the k basis)
        V: The potential energy operator (in the x basis)
        H: The total Hamiltonian (K + V)

        N: The dimension of the space (number of samples in x)
        x: The x space
        dx: The x space step size
        k: The k space
        dk: The k space step size
    """

    def __init__(self, N=128, bounds=[-5.0, 5.0], exp=2):
        """
        Initialize a Simulation.

        Args:
            N: The space dimension (number of samples)
            bounds: An array of two values representing the x boundaries
                    (optional, default is [-5, +5])
            exp: The exponent of the potential (optional, default is 2 i.e. the
                 harmonic oscillator potential)
        """
        self.N = N
        self.dx = (bounds[1]-bounds[0])/N
        self.x = np.linspace(bounds[0]+self.dx/2, bounds[1]-self.dx/2, N)

        self.dk = 2*np.pi/(N*self.dx)
        self.k = np.fft.fftfreq(N, 1./N)*self.dk

        # Kinetic energy
        self.K_k = np.diag(0.5 * self.k**2)          # in the momentum basis (k)
        U = np.sqrt(self.dx) * np.array([
                np.conj(self.plane_wave(m)) for m in self.k/self.dk
            ])
        K = np.dot(U.T.conj(), np.dot(self.K_k, U))  # in the space basis (x)

        # Potential energy
        V = np.diag(0.5*np.abs(self.x)**exp)         # in the space basis (x)

        # Hamiltonian
        self.K = K
        self.V = V
        self.H = K + V

    def plane_wave(self, m):
        """
        Generates a plane wave with coefficient m.

        Args:
            m: The plane wave coefficient
        """
        return self.normalize(np.exp(1j*self.dk*m*self.x))

    def normalize(self, wf):
        """
        Normalizes a wavefunction in x basis.
        """
        return wf/np.sqrt(np.vdot(wf, wf)*self.dx)

    def time_evolver(self, wf, dt, iterations=1):
        """
        Applies the time evolution operator to a given wavefunction using the
        second order Trotter splitting.

        Args:
            wf: The wavefunction (in x basis)
            dt: The time interval (note: it may be an imaginary time)
            iterations: the number of iterations (optional, default is 1)
        """
        wf_x = wf * np.exp(-0.5j*np.diag(self.V)*dt)
        wf_k = np.fft.fft(wf_x)
        wf_k = wf_k * np.exp(-1j*np.diag(self.K_k)*dt)
        wf_x = np.fft.ifft(wf_k)

        for i in range(iterations-1):
            # This is an optimization: we evolve the wavefunction for a whole
            # step dt in V, then the last half step is done outside the loop.
            wf_x = wf_x * np.exp(-1j*np.diag(self.V)*dt)

            wf_k = np.fft.fft(wf_x)
            wf_k = wf_k * np.exp(-1j*np.diag(self.K_k)*dt)

            wf_x = np.fft.ifft(wf_k)

        # Evolve for the last half step, and we’re done!
        return wf_x * np.exp(-0.5j*np.diag(self.V)*dt)

    def kinetic_energy(self, wf):
        """
        Evaluates the kinetic energy of a wavefunction.

        Args:
            wf: The wavefunction (in x basis)
        """
        wf_k = np.fft.fft(wf)

        return 0.5 * self.dx/self.N * np.sum(np.abs(wf_k)**2 * self.k**2)

    def potential_energy(self, wf):
        """
        Evaluates the potential energy of a wavefunction.

        Args:
            wf: The wavefunction (in x basis)
        """
        return self.dx * np.dot(np.abs(wf)**2, np.diag(self.V))

    def energy(self, wf):
        """
        Evaluates the total energy of a wavefunction (kinetic + potential).

        Args:
            wf: The wavefunction (in x basis)
        """
        return self.kinetic_energy(wf) + self.potential_energy(wf)

print('''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~ Find the eigenvalues with N = 50 by means of exact diagonalization.          ~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
''')
exponent = 2.3 # the potential exponent
sim = Simulation(N=50, exp=exponent)

eigvalues = np.linalg.eigvalsh(sim.H)

print("Plotting eigenvalues for simulation with N = 50")
print("(close the plot to continue)")
plot_title = "Eigenvalues (N = 50)"
plt.figure(1).canvas.set_window_title(plot_title)
plt.title(plot_title)
plt.plot(eigvalues, "+")
plt.show()

print("Done.")

print('''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~ Find the value for N such that the first 10 eigenvalues do not change their  ~
~ value more than 1e-3.                                                        ~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
''')
tolerance = 1e-3
eigvalues = np.linalg.eigvalsh(Simulation(N=10, exp=exponent).H)[:10]

print("Looking for convergence N with tolerance t = %.1e" % tolerance)
for n in range(11, 500):
    sim = Simulation(N=n, exp=exponent)
    new_eigvalues = np.linalg.eigvalsh(sim.H)[:10]
    error = np.abs(new_eigvalues - eigvalues)

    if (np.all(error < tolerance)):
        print("Convergence found for N = %i." % n)
        break

    eigvalues = new_eigvalues

print("Done.")

print('''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~ Plot the probability density for the first two eigenvectors.                 ~                                                        ~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
''')
n = max(n, 256)  # set minimum value of N to 256 so we get a decent smoothness
print("Plotting eigenstates with N = %i" % n)
print("(close the plot to continue)")
sim = Simulation(N=n, exp=exponent)
_, eigvectors = np.linalg.eigh(sim.H)

p0 = sim.normalize(np.abs(eigvectors[:, 0]))**2
p1 = sim.normalize(np.abs(eigvectors[:, 1]))**2

ylimit = np.max([p0, p1]) * 1.1
plot_title = "Eigenstates (N = %i)" % n
plt.figure(2).canvas.set_window_title(plot_title)
plt.suptitle(plot_title)

plt.subplot(211)
plt.ylabel("State 0")
plt.plot(sim.x, p0)
plt.ylim(0, ylimit)

plt.subplot(212)
plt.ylabel("State 1")
plt.plot(sim.x, p1)
plt.ylim(0, ylimit)

plt.show()

print("Done.")

print('''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~ Find the first two states by imaginary time evolution and plot a comparison  ~
~ with the states found previously.                                            ~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
''')
dt = -0.001j   # imaginary time interval
nsteps = 5000  # number of steps to evolve the wavefunction

print("Computing the states using dt = %.1e i, %i steps" % (dt.imag, nsteps))
print("Please wait for an imaginary time…")

# Start from a symmetric wavefunction to get the ground state (we know it will
# have no nodes, so it must be symmetric).
psi_gs = np.ones(sim.N)
for i in range(nsteps):
    psi_gs = sim.normalize(sim.time_evolver(psi_gs, dt))

# Start from an asymmetric wavefunction to get the first asymmetric state
psi_1 = np.copy(sim.x)
for i in range(nsteps):
    psi_1 = sim.normalize(sim.time_evolver(psi_1, dt))

print("Plotting eigenstates\n(close the plot to continue)")

plot_title = "Eigenstates comparison"
plt.figure(3).canvas.set_window_title(plot_title)
plt.suptitle(plot_title)

plt.subplot(211)
plt.ylabel("Ground state")
plt.plot(sim.x, np.abs(psi_gs)**2, label="Imaginary time")
plt.plot(sim.x, p0, label="Exact diagonalization")
plt.legend()

plt.subplot(212)
plt.ylabel("First asymmetric state")
plt.plot(sim.x, np.abs(psi_1)**2, label="Imaginary time")
plt.plot(sim.x, p1, label="Exact diagonalization")
plt.legend()

plt.show()

print("Done.\n")

print('''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~ Real time evolution of the shifted ground state wavefunction.                ~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
''')
dt = 0.001  # real time interval
psi = np.roll(psi_gs, sim.N/10)  # shifted wavefunction

class WavefunctionAnimation:
    def __init__(self, sim, psi, dt, steps=20):
        self.sim, self.psi, self.dt, self.steps = sim, psi, dt, steps
        self.fig, self.ax = plt.subplots()
        self.psi_line, = self.ax.plot(self.sim.x, np.abs(self.psi)**2)

        # Plot the current time
        self.time_text = self.ax.text(0.80, 0.95, "Time = 0",
            transform=self.ax.transAxes, bbox=dict(
                facecolor="#ffdd00", alpha=0.5,
                capstyle="round"
        ))

        # Plot the potential rescaled
        self.ax.plot(sim.x, np.diag(sim.V)/np.max(np.diag(sim.V)), "--")

    def __call__(self, frame_num):
        self.psi = sim.time_evolver(self.psi, self.dt, self.steps)
        self.psi_line.set_ydata(np.abs(self.psi)**2)
        self.time_text.set_text("Time = %6.2f" % (frame_num*self.steps*dt))

        return self.psi_line, self.time_text

# Run the animation
wa = WavefunctionAnimation(sim, psi, dt, steps=50)

print("Starting animation\n(close the animation to continue)")
_ = FuncAnimation(wa.fig, wa, frames=1000, interval=20, blit=True)
plt.show()

print("Done.")

print('''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~ Evaluate average energy and position and the time period of the average      ~
~ position oscillations.                                                       ~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
''')
steps = 1000
stepsize = 100
pos = np.zeros(steps)
energy = np.zeros(steps)

print("Calculating expectation values of energy and position")
print("Be patient, this will take some time…")

for i in range(steps):
    # Computing the time evolution for `stepsize` iterations, otherwise it would
    # take eons to do the computations for a reasonable time. However, we will
    # lose some high frequency components (if any).
    psi = sim.time_evolver(psi, dt, iterations=stepsize)

    pos[i] = np.sum(np.dot(np.abs(psi)**2, sim.x)*sim.dx)
    energy[i] = sim.energy(psi)

print("Plotting average position and energy\n(close the plot to continue)")

time = np.arange(steps*stepsize*dt, step=(stepsize*dt))

plot_title = "Average position and energy"
plt.figure(5).canvas.set_window_title(plot_title)
plt.title(plot_title)
plt.xlabel("Time")
plt.plot(time, pos, label="Position")
plt.plot(time, energy, label="Energy")

plt.legend()
plt.show()

print("Calculating the period as the frequency peak in the position average")
freq = np.abs(np.fft.rfft(pos))
freqspace = np.fft.rfftfreq(steps, stepsize*dt)

freq_peak = np.abs(freqspace[np.argmax(freq)])

print("The frequency peak is: %g\nTime period: %g" % (freq_peak, 1/freq_peak))

print("Plotting the spectrum\n(close the plot to continue)")
plot_title = "Frequency spectrum"

plt.figure(6).canvas.set_window_title(plot_title)
plt.title(plot_title)
plt.plot(freqspace, freq)
plt.xlabel("Frequency")
plt.show()

print('''


           _ _       _                  _
     /\   | | |     | |                | |
    /  \  | | |   __| | ___  _ __   ___| |
   / /\ \ | | |  / _` |/ _ \| '_ \ / _ \ |
  / ____ \| | | | (_| | (_) | | | |  __/_|
 /_/    \_\_|_|  \__,_|\___/|_| |_|\___(_)


''')
