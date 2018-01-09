import numpy as np
import matplotlib.pyplot as plt
import flow_func
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

####################################################################
######################## Initialization ############################
####################################################################

nRows = 10
nCols = 10
nCh = 9

timesteps = 300
# velocity vector for matrix indices starting top left
c = np.array([[0, 0], [0, 1], [-1, 0], [0, -1], [1, 0], [-1, 1], [-1, -1], [1, -1], [1, 1]])
# lattice
f = np.zeros((nRows, nCols, nCh), dtype=float)
# weights for initial distribution
w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])

# initial lattice distribution f
f = flow_func.f_init(f, w)

# attenuation factor
omega = 1
assert 0 < omega < 2.0, 'Limits of attenuation factor exceeded'

# initialize shear wave decay
epsilon = 0.01
assert epsilon <= 0.1, 'Limits of shear wave decay exceeded'

####################################################################
##################### Start with initial rho #######################
####################################################################

# set distribution offset
rhoOffset = 0.8
assert 0 < rhoOffset < 1.0, 'Limits of rhoT0 exceeded'
# initial distribution at t = 0
rhoTZero = flow_func.init_rho(epsilon, rhoOffset, nRows, nCols)
u = np.zeros((nRows, nCols, 2))

# Calculate lattice equilibrium according to given rho -> equal to f_eq
f = flow_func.calc_equilibrium(rhoTZero, u, c, w)

# sanity check if calculation of rho is equal to predefined rho
assert np.isclose(rhoTZero, flow_func.get_rho(f), rtol=1e-15, atol=1e-20, equal_nan=False).all(), 'Rho init failed'

####################################################################
######################### Scattering ###############################
####################################################################

rho = rhoTZero.copy()
row = 0

# Plotting
plt.figure()
plt.title('Rho row # %s' % row)
plt.xlabel('Rho column #')
plt.ylabel('a.U.')
ax = plt.subplot(111)

print(rho[row, :])

# time loop
for t in range(timesteps):

    if t % 20 == 0:
        ax.plot(rho[row, :], label='t = %s' %t)
        plt.draw()
        plt.pause(1e-4)
    # shift distribution f
    f = flow_func.shift_f(f)
    # get partial current density j
    j = flow_func.calc_j(c, f)
    # get current density rho
    rho = flow_func.get_rho(f)
    # get average velocity
    u = flow_func.calc_avg_vel(rho, j)
    # get local equilibrium distributions
    feQ = flow_func.calc_equilibrium(rho, u, c, w)
    # update distribution
    f += omega * (feQ - f)

print('After %f timesteps: %s' % (timesteps, rho[0, :]))

# Plotting
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()  # command blocks

