#!/usr/bin/env python3

import numpy as np
import matplotlib
import sys
import matplotlib.pyplot as plt

plt.switch_backend('Agg')
import matplotlib.cbook
import warnings
import datetime as dt
import os

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


class Flowfield:

    def __init__(self, nRows, nCols, nCh):
        # lattice
        self.lattice = np.zeros((nRows, nCols, nCh), dtype=float)
        # velocity vector for matrix indices starting top left
        self.c = np.array([[0, 0], [1, 0], [0, -1], [-1, 0], [0, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]])


def shift_f(grid, bounceMask, applyBounce=False):
    if applyBounce:
        saveGrid = np.ma.masked_where(bounceMask is True, grid)

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

    if applyBounce:
        # treat corners for ch 5, 6, 7, 8 separately
        if bounceMask[0, 0, 2] and bounceMask[-1, 0, 4]:  # TODO: check every point along line
            # top lid
            # channel 2->4, 5->7, 6->8
            grid[0, :, 4] = saveGrid[0, :, 2]
            grid[0, 1:, 7] = saveGrid[0, 1:, 5]
            grid[0, :-1, 8] = saveGrid[0, :-1, 6]
            # bottom
            # channel 4->2, 7->5, 8->6
            grid[-1, :, 2] = saveGrid[-1, :, 4]
            grid[-1, :-1, 5] = saveGrid[-1, :-1, 7]
            grid[-1, 1:, 6] = saveGrid[-1, 1:, 8]

        if bounceMask[0, 0, 3] and bounceMask[0, -1, 1]:  # TODO: couette flow must still be possible
            # left wall
            # channel 3->1, 7->5, 6->8,
            grid[:, 0, 1] = saveGrid[:, 0, 3]
            grid[1:, 0, 5] = saveGrid[1:, 0, 7]
            grid[:-1, 0, 8] = saveGrid[:-1, 0, 6]
            # right wall
            # channel 1->3, 8->6, 5->7
            grid[:, -1, 3] = saveGrid[:, -1, 1]
            grid[1:, -1, 6] = saveGrid[1:, -1, 8]
            grid[:-1, -1, 7] = saveGrid[:-1, -1, 5]

            # handle corners
            grid[0, 0, 5] = saveGrid[0, 0, 5]
            grid[-1, -1, 5] = saveGrid[-1, -1, 5]
            grid[0, -1, 6] = saveGrid[0, -1, 6]
            grid[-1, 0, 6] = saveGrid[-1, 0, 6]
            grid[0, 0, 7] = saveGrid[0, 0, 7]
            grid[-1, -1, 7] = saveGrid[-1, -1, 7]
            grid[0, -1, 8] = saveGrid[0, -1, 8]
            grid[-1, 0, 8] = saveGrid[-1, 0, 8]

    return grid


def sliding_lid(grid, rho):
    # according to slide 9 in HPC171129Ueb.pdf
    # assume ch 2 and 4 are in equilibrium
    fZero = grid[0, :, 0]
    f1 = grid[0, :, 1]
    f2 = grid[0, :, 2]
    f3 = grid[0, :, 3]
    f5 = grid[0, :, 5]
    f6 = grid[0, :, 6]
    rhoWall = fZero + f1 + f3 + 2 * (f2 + f5 + f6)

    uLid = 0.1
    # substract lid velocity
    ch = 7
    # grid[-1, :, ch] += (6 * w[ch] * rhoWall * c[ch, 0] * uLid).flatten()
    grid[0, :, ch] -= (6 * w[ch] * rhoWall * uLid).flatten()
    # add lid velocity
    ch = 8
    # grid[-1, :, ch] += (6 * w[ch] * rhoWall * c[ch, 0] * uLid).flatten()
    grid[0, :, ch] += (6 * w[ch] * rhoWall * uLid).flatten()

    return grid


def f_init(f, w):
    nCh = len(w)
    for i in range(nCh):
        f[:, :, i] = w[i]
    return f


def get_rho(f, checkMax=False):
    rho = np.sum(f, axis=2)
    if checkMax:
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
    cu2 = cu ** 2
    u2 = np.einsum('abc, abc -> ab', u, u)

    for i in range(nCh):
        fEq[:, :, i] = w[i] * rho * (1 + 3 * cu[:, :, i] + 9 / 2 * cu2[:, :, i] - 3 / 2 * u2)

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
    x = (2 * np.pi * np.arange(cols)) / (cols - 1)  # why cols-1 ?
    rho += epsilon * np.sin(x).reshape(1, cols)
    return rho


def init_u(epsilon, rows, cols):  # TODO: make dim variable. current is u_x with y pos
    # set offset plus a sinusoidal variation of the velocities u_x with the position y
    u = np.zeros((rows, cols, 2))
    # y vector with one sinusodial period
    y = (2 * np.pi * np.arange(rows)) / (rows)
    # only set velocities u_x
    u[:, :, 0] = epsilon * np.sin(y).reshape(rows, 1)
    return u


if __name__ == '__main__':

    ####################################################################
    ######################## Initialization ############################
    ####################################################################

    # lattice dimensions
    nRows = 300
    nCols = 300
    nCh = 9

    # bounce back boundary
    bounceMask = np.zeros((nRows, nCols, nCh))
    # set borders to bounce back

    bounceTopBottom = True
    # top lid
    bounceMask[0, :, 2] = bounceTopBottom
    bounceMask[0, :, 5] = bounceTopBottom
    bounceMask[0, :, 6] = bounceTopBottom
    # bottom
    bounceMask[-1, :, 4] = bounceTopBottom
    bounceMask[-1, :, 7] = bounceTopBottom
    bounceMask[-1, :, 8] = bounceTopBottom

    bounceLeftRight = True
    # left wall
    bounceMask[:, 0, 3] = bounceLeftRight
    bounceMask[:, 0, 6] = bounceLeftRight
    bounceMask[:, 0, 7] = bounceLeftRight
    # right wall
    bounceMask[:, -1, 1] = bounceLeftRight
    bounceMask[:, -1, 5] = bounceLeftRight
    bounceMask[:, -1, 8] = bounceLeftRight

    applyBounce = True
    applySlidingLid = True

    # number of timesteps
    timesteps = 1000000

    # lattice
    f = np.zeros((nRows, nCols, nCh), dtype=float)
    # weights for initial lattice distribution
    w = np.array(
        [4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0])
    # velocity vector for matrix indices starting top left
    c = np.array([[0, 0], [1, 0], [0, -1], [-1, 0], [0, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]])

    # initial lattice distribution f
    f = f_init(f, w)

    # attenuation factor
    omega = 0.7  # TODO: compute reynolds number >! 1000 for turbulent flow
    assert 0 < omega <= 1.7, 'Limits of attenuation factor exceeded'

    # initialize shear wave decay factor
    epsilon = 0.01
    assert epsilon <= 0.1, 'Limits of shear wave decay exceeded'

    ####################################################################
    ##################### Start with initial rho #######################
    ####################################################################

    startWithInitRho = False

    if startWithInitRho:
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

    startWithInitVelocity = False

    if startWithInitVelocity:
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
    ##################### Start with sliding lid #######################
    ####################################################################

    startWithSlidingLid = True

    if startWithSlidingLid:
        # initialize velocity only for top lid in x direction
        uTZero = np.zeros((nRows, nCols, 2))
        assert (abs(uTZero) < 0.1).all(), 'Limits of u exceeded'
        # set rho
        rhoTZero = np.ones((nRows, nCols))

        # Calculate lattice equilibrium according to given rho -> equal to f_eq
        f = calc_equilibrium(rhoTZero, uTZero, c, w)

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
    ax1 = fig1.add_subplot(111)  # The big subplot
    ax11 = fig1.add_subplot(211)
    ax12 = fig1.add_subplot(212)
    # Turn off axis lines and ticks of the big subplot
    ax1.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    # Set labels
    # ax.set_xlabel('common xlabel')
    # ax.set_ylabel('common ylabel')
    ax11.set_title('Column of average velocity u')
    ax12.set_title('Average velocity u at pi/4')

    fig2 = plt.figure(2)

    # storage of u vector points
    uStore = []
    # for streamplot
    x = np.arange(nRows)
    y = np.arange(nCols)
    X, Y = np.meshgrid(x, y)

    # time loop
    for i in range(timesteps):

        if showPlot:
            if i % plotDiscret == 0:
                if startWithInitVelocity:
                    ax11.plot(uScatter[:, colPlot, 0], label='t = %s' % i)
                else:
                    plt.close(fig1)
                # plot velocity streamfield
                fig2.clf()
                # plt.quiver(Y, X, uScatter[:,:,0].T, uScatter[:,:,1].T, color='b')
                plt.streamplot(X, Y, uScatter[:, :, 0], uScatter[:, :, 1], color='b')
                plt.ylim(len(Y), 0)
                plt.pause(1e-6)
        if (i + 1) % 100 == 0:
            print("\rTime {}/{}".format(i + 1, timesteps), end="")
            sys.stdout.flush()
        uStore = np.append(uStore, uScatter[nRows // 4, colPlot, 0])

        # shift distribution f
        f = shift_f(f, bounceMask, applyBounce)
        # slide lid
        if applySlidingLid:
            f = sliding_lid(f, rhoScatter)
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

    date = str(dt.datetime.now())[:10]
    np.save(os.environ['HOME'] + '/results/hpc/X_' + date, X)
    np.save(os.environ['HOME'] + '/results/hpc/Y_' + date, Y)
    np.save(os.environ['HOME'] + '/results/hpc/uScatter_' + date, uScatter)

    if showPlot:
        fig2.show()

    calcViscosity = False  # TODO: add viscosiy(nu) = (1/omega - 0.5) / 3 -> results matchs sufficient

    if calcViscosity:
        print('Rho after %d timesteps: %s \n' % (timesteps, rhoScatter[0, :]))
        t = np.arange(timesteps)
        lnU = np.log(uStore)

        # get slope of exp. fct via log formula
        Ly = nCols
        k = 2 * np.pi / Ly
        point = nRows // 10
        # compute viscosity(nu) the naive way
        nuNoise = (-np.log(uStore[point]) + np.log(uStore[0])) / (k ** 2 * point)
        nuNoiseAll = (-np.log(uStore[t[1::]]) + np.log(uStore[0])) / (k ** 2 * t[1::])
        print('Naive computation of nu = %.3f, mean(nu) = %.3f \n' % (nuNoise, np.mean(nuNoiseAll)))
        eFctNaive = uStore[0] * np.exp(-nuNoise * k ** 2 * t)

        # analytical solution of exponential fit
        t = np.arange(timesteps).reshape((-1, 1))
        phi = np.concatenate((t, np.ones((len(t), 1))), axis=1)
        coeffLn = np.linalg.inv(phi.T.dot(phi)).dot(phi.T).dot(lnU)

        # line equation
        print('Coefficients for linear log(u) fit = %s \n' % coeffLn)
        line = coeffLn[0] * t + coeffLn[1]

        # get coefficients for exponential equation
        A = np.exp(coeffLn[1])
        lam = coeffLn[0]
        print('A = %.3f, lambda = %.3f for exponential fit \n' % (A, lam))
        eFctFit = A * np.exp(lam * t)

        # compute viscosity(nu) the exact way
        nuExact = (-np.log(eFctFit[point]) + np.log(eFctFit[0])) / (k ** 2 * point)
        print('Exact computation of nu = %.3f \n' % nuExact)

        if showPlot:  # figure blocks -> must be at the end of the code!
            # plot log of u values
            ax12.plot(lnU, label='log(u)')
            # plot linear fit
            ax12.plot(line, 'r--', label='fit of log(u)')
            # plot exp fit
            # ax2.plot(eFct, 'g--', label='fit of $e^\lambda*t$')
            # Legend
            # Shrink current axis by 20%
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax11.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax12.legend(loc='best')
            # plot u & exponential fit
            fig2, ax2 = plt.subplots(1, 1)
            ax2.set_title('Average velocity u')
            ax2.plot(t, uStore, label='u')
            ax2.plot(t, eFctFit, 'g--', label='$y = A*e^{\lambda*x}$')
            ax2.plot(t, eFctNaive, 'k--', label='$u_0*e^{-\\nu*k^2*t}$', linewidth=0.6)
            ax2.legend(loc='best')

            plt.show()  # important to be at the end
