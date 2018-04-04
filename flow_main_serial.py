#!/usr/bin/env python3

import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt

plt.switch_backend('Agg')
import matplotlib.cbook
import warnings
from mpi4py import MPI
import os

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

parser = argparse.ArgumentParser()
parser.add_argument('-timesteps', type=int, default=1000, help='Number of timesteps')
parser.add_argument('-grid_size', type=int, default=576, help='Size of lattice(grid_size x grid_size)')
parser.add_argument('-vel_lid', type=float, default=0.1, help='Sliding lid velocity')
a = parser.parse_args()


class Flowfield:

    def __init__(self, nRows, nCols, nCh):
        # lattice
        self.lattice = np.zeros((nRows, nCols, nCh), dtype=float)
        # velocity vector for matrix indices starting top left as [x, y]
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
        if bounceMask[0, 0, 2] and bounceMask[-1, 0, 4]: # TODO: check every point along line
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

        if bounceMask[0, 0, 3] and bounceMask[0, -1, 1]: # TODO: couette flow must still be possible
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
    #grid[-1, :, ch] += (6 * w[ch] * rhoWall * c[ch, 0] * uLid).flatten()
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
    if (rho.flatten() < 0).any():
        warnings.warn('Negative occupation / rho. Save u and terminate program...')
        return rho, 1
    return rho, 0


def calc_j(c, f):
    j = f.dot(c)
    return j


def calc_avg_vel(rho, j):
    u = j / rho.reshape(rho.shape[0], rho.shape[1], 1)
    return u


def calc_equilibrium(rho, u, c, w):
    nRows = rho.shape[0]
    nCols = rho.shape[1]
    nCh = len(w)

    fEq = np.zeros((nRows, nCols, nCh))

    cu = np.einsum('ab, cdb -> cda', c, u)
    cu2 = cu ** 2
    u2 = np.einsum('abc, abc -> ab', u, u)

    for i in range(nCh):
        fEq[:, :, i] = w[i] * rho * (1 + 3 * cu[:, :, i] + 9.0 / 2.0 * cu2[:, :, i] - 3.0 / 2.0 * u2)

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


def save_u(u_scatter):
    ux_name = os.environ['HOME'] + '/results/hpc/ux_serial_ts_' + str(a.timesteps) + '.npy'
    uy_name = os.environ['HOME'] + '/results/hpc/uy_serial_ts_' + str(a.timesteps) + '.npy'

    np.save(ux_name, u_scatter[:, :, 0])
    np.save(uy_name, u_scatter[:, :, 1])
    print('Saved results')


if __name__ == '__main__':

    ####################################################################
    ######################## Lattice init ##############################
    ####################################################################

    num_ch = 9
    applyBounce = True
    applySlidingLid = True

    # lattice
    f = np.zeros((a.grid_size, a.grid_size, num_ch), dtype=float)
    # weights for initial lattice distribution
    w = np.array(
        [4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0])
    # velocity vector for matrix indices starting top left as [x, y]
    c = np.array([[0, 0], [1, 0], [0, -1], [-1, 0], [0, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]])

    # initial lattice distribution f
    f = f_init(f, w)

    # attenuation factor
    omega = 0.4
    assert 0 < omega <= 1.7, 'Limits of attenuation factor exceeded'

    kin_visco = 1/3 * (1/omega - 1/2)
    reynolds = (a.grid_size * a.vel_lid) / kin_visco
    if reynolds <= 1000: warnings.warn('Reynolds number must be bigger than 1000. No turbulent flow possible')

    # initialize shear wave decay factor
    epsilon = 0.01
    assert epsilon <= 0.1, 'Limits of shear wave decay exceeded'

    # bounce back boundary
    bounceMask = np.zeros((a.grid_size, a.grid_size, num_ch))
    # set borders to bounce back

    bounceTopBottom = True
    # top lid
    bounceMask[1, :, 2] = bounceTopBottom
    bounceMask[1, :, 5] = bounceTopBottom
    bounceMask[1, :, 6] = bounceTopBottom
    # bottom
    bounceMask[-2, :, 4] = bounceTopBottom
    bounceMask[-2, :, 7] = bounceTopBottom
    bounceMask[-2, :, 8] = bounceTopBottom

    bounceLeftRight = True
    # left wall
    bounceMask[:, 1, 3] = bounceLeftRight
    bounceMask[:, 1, 6] = bounceLeftRight
    bounceMask[:, 1, 7] = bounceLeftRight
    # right wall
    bounceMask[:, -2, 1] = bounceLeftRight
    bounceMask[:, -2, 5] = bounceLeftRight
    bounceMask[:, -2, 8] = bounceLeftRight

    ####################################################################
    ##################### Start with sliding lid #######################
    ####################################################################

    startWithSlidingLid = True

    if startWithSlidingLid:
        # initialize velocity only for top lid in x direction
        uTZero = np.zeros((a.grid_size, a.grid_size, 2))
        assert (abs(uTZero) < 0.1).all(), 'Limits of u exceeded'
        # set rho
        rhoTZero = np.ones((a.grid_size, a.grid_size))

        # Calculate lattice equilibrium according to given rho -> equal to f_eq
        f = calc_equilibrium(rhoTZero, uTZero, c, w)

    rhoScatter = rhoTZero.copy()
    uScatter = uTZero.copy()

    # time loop
    for i in range(a.timesteps):
        if (i + 1) % 1000 == 0:
            print("\rTime {}/{}".format(i + 1, a.timesteps), end="")
            sys.stdout.flush()
        # shift distribution f
        f = shift_f(f, bounceMask, applyBounce)
        # slide lid
        if applySlidingLid:
            f = sliding_lid(f, rhoScatter)
        # get partial current density j
        j = calc_j(c, f)
        # get current density rho
        rhoScatter, rho_exit = get_rho(f)
        if rho_exit:
            save_u(uScatter)
            raise ValueError('Negative rho')
        # get average velocity
        uScatter = calc_avg_vel(rhoScatter, j)
        # get local equilibrium distributions
        feQ = calc_equilibrium(rhoScatter, uScatter, c, w)
        # update distribution
        f += omega * (feQ - f)

    save_u(uScatter)


