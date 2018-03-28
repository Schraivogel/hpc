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
parser.add_argument('-num_procs', type=int, default=4, help='Number of processes')
parser.add_argument('-vel_lid', type=float, default=0.1, help='Sliding lid velocity')
a = parser.parse_args()


class Flowfield:

    def __init__(self, nRows, nCols, nCh):
        # lattice
        self.lattice = np.zeros((nRows, nCols, nCh), dtype=float)
        # velocity vector for matrix indices starting top left as [x, y]
        self.c = np.array([[0, 0], [1, 0], [0, -1], [-1, 0], [0, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]])


def shift_f(grid, bounceMask, borders, applyBounce=False):
    # f defined as [y, x, q] starting top left
    # borders as left, right, down, up
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

        # left wall
        # channel 3->1, 7->5, 6->8,
        if borders[0]:
            c = 1
            grid[:, c, 1] = saveGrid[:, c, 3]
            grid[1:, c, 5] = saveGrid[1:, c, 7]
            grid[:-1, c, 8] = saveGrid[:-1, c, 6]

        # right wall
        # channel 1->3, 8->6, 5->7
        if borders[1]:
            grid[:, -1, 3] = saveGrid[:, -1, 1]
            grid[1:, -1, 6] = saveGrid[1:, -1, 8]
            grid[:-1, -1, 7] = saveGrid[:-1, -1, 5]

        # bottom
        # channel 4->2, 7->5, 8->6
        if borders[2]:
            r = -2
            grid[r, :, 2] = saveGrid[r, :, 4]
            grid[r, :-1, 5] = saveGrid[r, :-1, 7]
            grid[r, 1:, 6] = saveGrid[r, 1:, 8]

        # top lid
        # channel 2->4, 5->7, 6->8
        if borders[3]:
            r = 1
            grid[r, :, 4] = saveGrid[r, :, 2]
            grid[r, 1:, 7] = saveGrid[r, 1:, 5]
            grid[r, :-1, 8] = saveGrid[r, :-1, 6]

        # handle corners
        grid[1, 1, 5] = saveGrid[1, 1, 5]
        grid[-2, -2, 5] = saveGrid[-2, -2, 5]

        grid[1, -2, 6] = saveGrid[1, -2, 6]
        grid[-2, 1, 6] = saveGrid[-2, 1, 6]

        grid[1, 1, 7] = saveGrid[1, 1, 7]
        grid[-2, -2, 7] = saveGrid[-2, -2, 7]

        grid[1, -2, 8] = saveGrid[1, -2, 8]
        grid[-2, 1, 8] = saveGrid[-2, 1, 8]

    return grid


def sliding_lid(grid, rho, borders):
    # according to slide 9 in HPC171129Ueb.pdf
    # assume ch 2 and 4 are in equilibrium
    # take care of bounce back
    # borders as left, right, down, up

    if borders[3]:
        r = 1
        fZero = grid[r, :, 0]
        f1 = grid[r, :, 1]
        f2 = grid[r, :, 2]
        f3 = grid[r, :, 3]
        f5 = grid[r, :, 5]
        f6 = grid[r, :, 6]
        rhoWall = fZero + f1 + f3 + 2 * (f2 + f5 + f6)

        # substract lid velocity
        ch = 7
        # grid[-1, :, ch] += (6 * w[ch] * rhoWall * c[ch, 0] * uLid).flatten()
        grid[r, :, ch] -= (6 * w[ch] * rhoWall * a.vel_lid).flatten()
        # add lid velocity
        ch = 8
        # grid[-1, :, ch] += (6 * w[ch] * rhoWall * c[ch, 0] * uLid).flatten()
        grid[r, :, ch] += (6 * w[ch] * rhoWall * a.vel_lid).flatten()

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


def save_mpiio(comm, fn, g_kl):
    """
    Write a global two-dimensional array to a single file in the npy format
    using MPI I/O: https://docs.scipy.org/doc/numpy/neps/npy-format.html

    Arrays written with this function can be read with numpy.load.

    Parameters
    ----------
    comm
        MPI communicator.
    fn : str
        File name.
    g_kl : array_like
        Portion of the array on this MPI processes. This needs to be a
        two-dimensional array.
    """
    from numpy.lib.format import dtype_to_descr, magic
    magic_str = magic(1, 0)

    local_nx, local_ny = g_kl.shape
    nx = np.empty_like(local_nx)
    ny = np.empty_like(local_ny)

    commx = comm.Sub((True, False))
    commy = comm.Sub((False, True))
    commx.Allreduce(np.asarray(local_nx), nx)
    commy.Allreduce(np.asarray(local_ny), ny)

    arr_dict_str = str({'descr': dtype_to_descr(g_kl.dtype),
                        'fortran_order': False,
                        'shape': (np.asscalar(nx), np.asscalar(ny))})
    while (len(arr_dict_str) + len(magic_str) + 2) % 16 != 15:
        arr_dict_str += ' '
    arr_dict_str += '\n'
    header_len = len(arr_dict_str) + len(magic_str) + 2

    offsetx = np.zeros_like(local_nx)
    commx.Exscan(np.asarray(ny * local_nx), offsetx)
    offsety = np.zeros_like(local_ny)
    commy.Exscan(np.asarray(local_ny), offsety)

    file = MPI.File.Open(comm, fn, MPI.MODE_CREATE | MPI.MODE_WRONLY)
    if comm.Get_rank() == 0:
        file.Write(magic_str)
        file.Write(np.int16(len(arr_dict_str)))
        file.Write(arr_dict_str.encode('latin-1'))
    mpitype = MPI._typedict[g_kl.dtype.char]
    filetype = mpitype.Create_vector(g_kl.shape[0], g_kl.shape[1], ny)
    filetype.Commit()
    file.Set_view(header_len + (offsety + offsetx) * mpitype.Get_size(),
                  filetype=filetype)
    file.Write_all(g_kl.copy())
    filetype.Free()
    file.Close()


#  Note that these matrices need to be two-dimensional, i.e. you will need to write the individual cartesian
#  components of a velocity vector to separate files. To write the velocities from your code, just execute:

# save_mpiio(comm, 'ux.npy', velocities[0])
# save_mpiio(comm, 'uy.npy', velocities[1])


def mpi_communicate(comm, cartcomm, f):
    # communicate cells between carts

    leftSrc, leftDst = cartcomm.Shift(0, -1)
    rightSrc, rightDst = cartcomm.Shift(0, 1)
    upSrc, upDst = cartcomm.Shift(1, -1)
    downSrc, downDst = cartcomm.Shift(1, 1)

    # left
    recvbuf = f[:, -1, :].copy()  # TODO: init once
    comm.Sendrecv(sendbuf=f[:, 1, :].copy(), dest=leftDst,
                  recvbuf=recvbuf, source=leftSrc)
    f[:, -1, :] = recvbuf

    # right
    recvbuf = f[:, 0, :].copy()  # TODO: init once
    comm.Sendrecv(sendbuf=f[:, -2, :].copy(), dest=rightDst,
                  recvbuf=recvbuf, source=rightSrc)
    f[:, 0, :] = recvbuf

    # down
    recvbuf = f[0, :, :].copy()  # TODO: init once
    comm.Sendrecv(sendbuf=f[-2, :, :].copy(), dest=downDst,
                  recvbuf=recvbuf, source=downSrc)
    f[0, :, :] = recvbuf

    # up
    recvbuf = f[-1, :, :].copy()  # TODO: init once
    comm.Sendrecv(sendbuf=f[1, :, :].copy(), dest=upDst,
                  recvbuf=recvbuf, source=upSrc)
    f[-1, :, :] = recvbuf

    """
    rank = comm.Get_rank()
    if leftDst != -1:
        print('Left shift from {} to {}'.format(cartcomm.Get_coords(rank), cartcomm.Get_coords(leftDst)))
    if rightDst != -1:
        print('Right shift from {} to {}'.format(cartcomm.Get_coords(rank), cartcomm.Get_coords(rightDst)))
    if downDst != -1:
        print('Down shift from {} to {}'.format(cartcomm.Get_coords(rank), cartcomm.Get_coords(downDst)))
    if upDst != -1:
        print('Up shift from {} to {}'.format(cartcomm.Get_coords(rank), cartcomm.Get_coords(upDst)))
    sys.stdout.flush()
    """

    return f


def get_borders(rank, cartcomm):
    # check if on edge of full lattice

    leftSrc, leftDst = cartcomm.Shift(0, -1)
    rightSrc, rightDst = cartcomm.Shift(0, 1)
    upSrc, upDst = cartcomm.Shift(1, -1)
    downSrc, downDst = cartcomm.Shift(1, 1)

    # left, right, down, up edge
    borders = np.zeros((4, 1), dtype=bool)
    if leftDst < 0:
        borders[0] = True
    if rightDst < 0:
        borders[1] = True
    if downDst < 0:
        borders[2] = True
    if upDst < 0:
        borders[3] = True

    return borders


def save_u(u_scatter, cartcomm, size):
    ux_name = os.environ['HOME'] + '/results/hpc/ux_np_' + str(size) + '_ts_' + str(a.timesteps) + '.npy'
    uy_name = os.environ['HOME'] + '/results/hpc/uy_np_' + str(size) + '_ts_' + str(a.timesteps) + '.npy'

    # save method expects x as first axis -> [x, y]
    save_mpiio(cartcomm, ux_name, u_scatter[1:-1, 1:-1, 0].T)
    save_mpiio(cartcomm, uy_name, u_scatter[1:-1, 1:-1, 1].T)
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

    ####################################################################
    ######################### comm init ###############################
    ####################################################################
    comm = MPI.COMM_WORLD

    size = comm.Get_size()
    rank = comm.Get_rank()
    print('rank = {}/{}'.format(rank, size))

    # periodic communication disabled
    factor = np.sqrt(size)
    assert (factor * factor) == size, 'Square root of number of processes must be an integer'
    factor = int(factor)
    dims = (factor, factor)

    cartcomm = comm.Create_cart(dims, periods=(False, False), reorder=(False, False))

    # get edges according to cartesian cart position
    borders = get_borders(rank, cartcomm)
    # add ghost cells
    localRows = a.grid_size // dims[0] + 2
    localCols = a.grid_size // dims[1] + 2

    # bounce back boundary
    bounceMask = np.zeros((localRows, localCols, num_ch))
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
        uTZero = np.zeros((localRows, localCols, 2))
        assert (abs(uTZero) < 0.1).all(), 'Limits of u exceeded'
        # set rho
        rhoTZero = np.ones((localRows, localCols))

        # Calculate lattice equilibrium according to given rho -> equal to f_eq
        f = calc_equilibrium(rhoTZero, uTZero, c, w)

    rhoScatter = rhoTZero.copy()
    uScatter = uTZero.copy()

    # time loop
    for i in range(a.timesteps):
        # mpi communication
        f = mpi_communicate(comm, cartcomm, f)
        if (i + 1) % 1000 == 0:
            print("\rTime {}/{}".format(i + 1, a.timesteps), end="")
            sys.stdout.flush()
        # shift distribution f
        f = shift_f(f, bounceMask, borders, applyBounce)
        # slide lid
        if applySlidingLid:
            f = sliding_lid(f, rhoScatter, borders)
        # get partial current density j
        j = calc_j(c, f)
        # get current density rho
        rhoScatter, rho_exit = get_rho(f)
        if rho_exit:
            save_u(uScatter, cartcomm, size)
            raise ValueError('Negative rho')
        # get average velocity
        uScatter = calc_avg_vel(rhoScatter, j)
        # get local equilibrium distributions
        feQ = calc_equilibrium(rhoScatter, uScatter, c, w)
        # update distribution
        f += omega * (feQ - f)

    save_u(uScatter, cartcomm, size)
    print('Finished: Rank {}'.format(rank))
    sys.stdout.flush()


