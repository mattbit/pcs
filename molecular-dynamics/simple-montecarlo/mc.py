#######################################
# Simple Monte Carlo simulation       #
# =================================== #
# Simulates the motion of a particle  #
# in a harmonic potential using a C   #
# FFI to achieve better performance.  #
#######################################

# Run mc_build.py to compile
# the C module.
from _montecarlo_sim import ffi, lib

######################################
# Parameters definitions             #
######################################
delta = 0.2
num_steps = 100000000
q0 = 1

######################################
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
######################################

q2 = lib.runSimulation(delta, num_steps, q0)

print(q2)

# Plot graphs with matplotlib!
