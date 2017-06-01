#######################################
# Simple Monte Carlo simulation       #
# =================================== #
# Simulates the motion of a particle  #
# in a harmonic potential.            #
#######################################

import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(20161128)

######################################
# Parameters definitions             #
######################################
delta = np.linspace(0.15, 0.25, 20)
num_steps = 100000
q0 = 1

######################################
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
######################################

class Simulation:
    def __init__(self, delta = 0.1, num_steps = 100000, q0 = 0):
        self.delta = delta
        self.num_steps = num_steps
        self.q0 = q0

    def P(self, q):
        return np.exp(-(q**2)/2)

    def generateGuess(self):
        return self.q0 + random.uniform(-self.delta, +self.delta)

    def accepted(self, q):
        return min(1, self.P(q)/self.P(self.q0)) >= random.random()

    def run(self):
        q2acc = 0

        for t in range(self.num_steps):
            q = self.generateGuess()

            if (self.accepted(q)):
                self.q0 = q

            q2acc += q**2

        return float(q2acc)/self.num_steps


# Evaluate <q^2> for different values of delta
q2s = np.zeros(len(delta))
for i in range(len(delta)):
 sim = Simulation(delta=delta[i], num_steps=num_steps, q0=q0)
    q2, qs = sim.run()
    q2s[i] = q2

plt.plot(delta, np.abs(q2s - 1), '+-')
plt.title("q^2")
plt.show()

# steps = np.logspace(start=2, stop=6, num=50)
# steps = [1000000000]
# q2s = np.zeros(len(steps))
# for i in range(len(steps)):
#     sim = Simulation(delta=0.2, num_steps=int(steps[i]), q0=q0)
#     q2 = sim.run()
#     q2s[i] = q2
#
# print q2s
