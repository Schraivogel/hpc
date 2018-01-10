import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
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
	rho = np.sum(f, axis=2, dtype=float)
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
	for row in range(nRows):
		for col in range(nCols):
			for ch in range(nCh):
				node = (row, col)

				rhoTmp = rho[node]
				uTmp = u[node].reshape((-1, 1)).T
				cTmp = c[ch].reshape((-1, 1))

				part1 = w[ch] * rhoTmp
				part2 = (1 + 3 * uTmp.dot(cTmp) + 9/2 * uTmp.dot(cTmp) ** 2 - 3/2 * uTmp.dot(uTmp.T))

				fEq[row, col, ch] = part1 * part2
	print(rho)
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
		
		
if __name__ == "__main__":
	####################################################################
	######################## Initialization ############################
	####################################################################

	# lattice dimensions
	nRows = 40
	nCols = 40
	nCh = 9

	# number of timesteps
	timesteps = 200

	# velocity vector for matrix indices starting top left
	c = np.array([[0, 0], [0, 1], [-1, 0], [0, -1], [1, 0], [-1, 1], [-1, -1], [1, -1], [1, 1]])
	# lattice
	f = np.zeros((nRows, nCols, nCh), dtype=float)
	# weights for initial lattice distribution
	w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

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
	print(rhoTZero)
	print(get_rho(f))
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
	plotDiscret = 20
	# Plotting
	# Two subplots, the axes array is 1-d
	fig = plt.figure()
	ax = fig.add_subplot(111)    # The big subplot
	ax1 = fig.add_subplot(311)
	ax2 = fig.add_subplot(312)
	ax3 = fig.add_subplot(313)

	# Turn off axis lines and ticks of the big subplot
	ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

	# Set common labels
	ax.set_xlabel('common xlabel')
	ax.set_ylabel('common ylabel')

	ax1.set_title('Row of density rho')
	ax2.set_title('Column of verage velocity u')
	ax3.set_title('Values of verage velocity u at pi/4')
	# storage of u vector points
	uStore = np.zeros((1,0))

	# time loop
	for t in range(timesteps):

		if t % plotDiscret == 0:
			uStore = np.append(uStore, uScatter[nRows//4, colPlot, 0])
			ax1.plot(rhoScatter[rowPlot, :], label='t = %s' % t)
			print(rhoTZero)
			ax2.plot(uScatter[:, colPlot, 0], label='t = %s' % t)
			plt.draw()
			plt.pause(1e-4)
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

	print('Rho after %f timesteps: %s' % (timesteps, rhoScatter[0, :]))

	ax3.plot(uStore)

	# Legend
	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	# Put a legend to the right of the current axis
	ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.show(ax) # figure blocks -> must be at the end of the code!
	'''
	rho = np.einsum('ijk->jk', f)
	u = np.einsum('ai,ixy->axy',c,f)/rho
	'''
