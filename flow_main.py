import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cbook
import warnings
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

def shift_f(grid):
	# center (0,0)
	# stays constant

	# east (0,1)
	ch = 1
	grid[:, :, ch] = np.roll(grid[:, :, ch], shift=1, axis=1)

	# north (-1, 0)
	ch = 2
	grid[:, :, ch] = np.roll(grid[:, :, ch], shift=-1, axis=0)

	# west (0, -1)
	ch = 3
	grid[:, :, ch] = np.roll(grid[:, :, ch], shift=-1, axis=1)

	# south (1, 0)
	ch = 4
	grid[:, :, ch] = np.roll(grid[:, :, ch], shift=1, axis=0)

	# north-east(-1, 1)
	ch = 5
	grid[:, :, ch] = np.roll(np.roll(grid[:, :, ch], shift=-1, axis=0), shift=1, axis=1)

	# north-west(-1, -1)
	ch = 6
	grid[:, :, ch] = np.roll(np.roll(grid[:, :, ch], shift=-1, axis=0), shift=-1, axis=1)

	# south-west(1, -1)
	ch = 7
	grid[:, :, ch] = np.roll(np.roll(grid[:, :, ch], shift=1, axis=0), shift=-1, axis=1)

	# south-east(1, 1)
	ch = 8
	grid[:, :, ch] = np.roll(np.roll(grid[:, :, ch], shift=1, axis=0), shift=1, axis=1)

	return grid


def f_init(f, w):
	nCh = len(w)
	for i in range(nCh):
		f[:, :, i] = w[i]
	return f


def get_rho(f):
	rho = np.sum(f, axis=2)
	pMax = np.ones(rho.shape) + 1e-6
	try:
		np.testing.assert_array_less(rho, pMax)
	except AssertionError:
		with warnings.catch_warnings():
			warnings.simplefilter("once")
			warnings.warn("Rho bigger than one")
	assert (rho.flatten() < 0).any() == False, 'Negative occupation / rho.'
	return rho


def calc_j(c, f):
	j = f.dot(c)
	return j


def calc_avg_vel(rho, j):
	u = j / rho.reshape(rho.shape[0], rho.shape[1], 1)
	return u


def calc_equilibrium(rho, u, c, w):  # TODO: Use Einsum
	nRows = rho.shape[0]
	nCols = rho.shape[1]
	nCh = len(w)

	fEq = np.zeros((nRows, nCols, nCh))

	cu = np.einsum('ab, cdb -> cda', c, u)
	cu2 = cu**2
	u2 = np.einsum('abc, abc -> ab', u, u)

	for i in range(nCh):
		fEq[:, :, i] = w[i] * rho * (1 + 3 * cu[:,:,i] + 9 / 2 * cu2[:,:,i] - 3 / 2 * u2)

	return fEq


def check_dim(f, c, w, j, rho, u):
	nCh = f.shape[2]
	print('Number of channels:', nCh)
	print('Shape of distribution lattice:', f.shape)
	print('Shape of c:', c.shape, '. Vector of finite velocity directions')
	print('Shape of w:', w.shape, '. Weight vector')
	print('Shape of j:', j.shape, '= lattice.dot(c)', '. Weighted c  per node')
	print('Shape of rho:', rho.shape, '. sum of particles over all channels per node')
	print('Shape of u:', u.shape, '= j/rho', '. Weighted vector of directions per node')
	print('\n')


def init_rho(epsilon, rhoOffset, rows, cols):  # TODO: make dim variable. current is x
	# set rho offset
	rho = np.full((rows, cols), rhoOffset, dtype=float)
	x = (2 * np.pi * np.arange(cols)) / (cols-1)  # why cols-1 ?
	rho += epsilon * np.sin(x).reshape(1, cols)
	return rho


def init_u(epsilon, rows, cols):  # TODO: make dim variable. current is u_x with y pos
	# set offset plus a sinusoidal variation of the velocities u_x with the position y
	u = np.zeros((rows, cols, 2))
	# y vector with one sinusodial period
	y = (2 * np.pi * np.arange(rows)) / (rows)
	# only set velocities u_x
	u[:,:,0] = epsilon * np.sin(y).reshape(rows, 1)
	return u
		
		

####################################################################
######################## Initialization ############################
####################################################################

# lattice dimensions
nRows = 50
nCols = 50
nCh = 9

# number of timesteps
timesteps = 400

# velocity vector for matrix indices starting top left
c = np.array([[0, 0], [1, 0], [0, -1], [-1, 0], [0, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]])
# lattice
f = np.zeros((nRows, nCols, nCh), dtype=float)
# weights for initial lattice distribution
w = np.array([4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0])

# initial lattice distribution f
f = f_init(f, w)

# attenuation factor
omega = 0.1
assert 0 < omega < 1.7, 'Limits of attenuation factor exceeded'

# initialize shear wave decay factor
epsilon = 0.01
assert epsilon < 0.1, 'Limits of shear wave decay exceeded'

####################################################################
##################### Start with initial rho #######################
####################################################################

# set density distribution offset
rhoOffset = 0.8
assert 0 < rhoOffset < 1.0, 'Limits of rhoT0 exceeded'
# initial density distribution at t = 0
rhoTZero = init_rho(epsilon, rhoOffset, nRows, nCols)
# set u to zero
uTZero = np.zeros((nRows, nCols, 2))

# Calculate lattice equilibrium according to given rho -> equal to f_eq
f = calc_equilibrium(rhoTZero, uTZero, c, w)

# sanity check if calculated rho from the lattice is equal to predefined rho at t = 0
assert np.isclose(rhoTZero, get_rho(f), rtol=1e-15, atol=1e-20, equal_nan=False).all(), 'Rho init failed'

####################################################################
############### Start with initial average velocity u ##############
####################################################################

# initial density distribution at t = 0
uTZero = init_u(epsilon, nRows, nCols)
assert (abs(uTZero) < 0.1).all(), 'Limits of u exceeded'
# set rho to all ones
rhoTZero = np.ones((nRows, nCols))

# Calculate lattice equilibrium according to given rho -> equal to f_eq
f = calc_equilibrium(rhoTZero, uTZero, c, w)

# sanity check if calculated u is equal to predefined u at t = 0
# compute j, rho-> == 1, u == uTZero
####################################################################
######################### Scattering ###############################
####################################################################

rhoScatter = rhoTZero.copy()
uScatter = uTZero.copy()
rowPlot = 0
colPlot = 0

# Plotting
showPlot = False
plotDiscret = 50
# Two subplots, the axes array is 1-d
fig1 = plt.figure(figsize=(10, 9))
ax1 = fig1.add_subplot(111)    # The big subplot
ax11 = fig1.add_subplot(211)
ax12 = fig1.add_subplot(212)

# Turn off axis lines and ticks of the big subplot
ax1.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

# Set labels
# ax.set_xlabel('common xlabel')
# ax.set_ylabel('common ylabel')
ax11.set_title('Column of average velocity u')
ax12.set_title('Average velocity u at pi/4')

# storage of u vector points
uStore = []

# time loop
for i in range(timesteps):

	if showPlot:
		if i % plotDiscret == 0:
			ax11.plot(uScatter[:, colPlot, 0], label='t = %s' % i)
			plt.pause(1e-4)
	uStore = np.append(uStore, uScatter[nRows // 4, colPlot, 0])
	# shift distribution f
	f = shift_f(f)
	# get partial current density j
	j = calc_j(c, f)
	# get current density rho
	rhoScatter = get_rho(f)
	# get average velocity
	uScatter = calc_avg_vel(rhoScatter, j)
	# get local equilibrium distributions
	feQ = calc_equilibrium(rhoScatter, uScatter, c, w)
	# update distribution
	f += omega * (feQ - f)

print('Rho after %d timesteps: %s \n' % (timesteps, rhoScatter[0, :]))
t = np.arange(timesteps)
lnU = np.log(uStore)

# get slope of exp. fct via log formula
Ly = nCols
k = 2*np.pi / Ly
point = 20
# compute viscosity(nu) the naive way
nuNoise = (-np.log(uStore[point]) + np.log(uStore[0])) / (k**2 * point)
nuNoiseAll = (-np.log(uStore[t[1::]]) + np.log(uStore[0])) / (k**2 * t[1::])
print('Naive computation of nu = %.3f, mean(nu) = %.3f \n' % (nuNoise, np.mean(nuNoiseAll)))
eFctNaive = uStore[0] * np.exp(-nuNoise * k**2 * t)

# analytical solution of exponential fit
t = np.arange(timesteps).reshape((-1, 1))
phi = np.concatenate((t, np.ones((len(t), 1))), axis=1)
coeffLn = np.linalg.inv(phi.T.dot(phi)).dot(phi.T).dot(lnU)

# line equation
print('Coefficients for linear log(u) fit = %s \n' % coeffLn)
line = coeffLn[0]*t + coeffLn[1]

# get coefficients for exponential equation
A = np.exp(coeffLn[1])
lam = coeffLn[0]
print('A = %.3f, lambda = %.3f for exponential fit \n' % (A, lam))
eFctFit = A * np.exp(lam*t)

# compute viscosity(nu) the exact way
nuExact = (-np.log(eFctFit[point]) + np.log(eFctFit[0])) / (k**2 * point)
nuExactAll = (-np.log(eFctFit[t[1::]]) + np.log(eFctFit[0])).reshape((-1,1)) / (k**2 * t[1::])
print('Exact computation of nu = %.3f, mean(nu) = %.3f \n' % (nuExact, np.mean(nuExactAll)))

if showPlot: # figure blocks -> must be at the end of the code!
	# plot log of u values
	ax12.plot(lnU, label='log(u)')
	# plot linear fit
	ax12.plot(line, 'r--', label='fit of log(u)')
	# plot exp fit
	#ax2.plot(eFct, 'g--', label='fit of $e^\lambda*t$')
	# Legend
	# Shrink current axis by 20%
	box = ax1.get_position()
	ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	# Put a legend to the right of the current axis
	ax11.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	ax12.legend(loc='best')
	# plot u & exponential fit
	fig2, ax2 = plt.subplots(1,1)
	ax2.plot(t, uStore, label='u')
	ax2.plot(t, eFctFit, 'g--', label='$y = A*e^{\lambda*x}$')
	ax2.plot(t, eFctNaive, 'k--', label='$u_0*e^{-\\nu*k^2*t}$', linewidth=0.6)
	ax2.legend(loc='best')

	plt.show() # important to be at the end

