# coding: utf-8

import numpy as np
import warnings

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
                part2 = (1 + 3 * uTmp.dot(cTmp) +
                         9 / 2 * uTmp.dot(cTmp) ** 2 -
                         3 / 2 * uTmp.dot(uTmp.T))

                fEq[row, col, ch] = part1 * part2

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
