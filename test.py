import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from threading import Timer
import warnings
from mpi4py import MPI

import numpy as np
import matplotlib
import sys

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cbook
import warnings
from mpi4py import MPI

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

    arr_dict_str = str({ 'descr': dtype_to_descr(g_kl.dtype),
                         'fortran_order': False,
                         'shape': (np.asscalar(nx), np.asscalar(ny)) })
    while (len(arr_dict_str) + len(magic_str) + 2) % 16 != 15:
        arr_dict_str += ' '
    arr_dict_str += '\n'
    header_len = len(arr_dict_str) + len(magic_str) + 2

    offsetx = np.zeros_like(local_nx)
    commx.Exscan(np.asarray(ny*local_nx), offsetx)
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
    file.Set_view(header_len + (offsety+offsetx)*mpitype.Get_size(),
                  filetype=filetype)
    file.Write_all(g_kl.copy())
    filetype.Free()
    file.Close()

#  Note that these matrices need to be two-dimensional, i.e. you will need to write the individual cartesian
#  components of a velocity vector to separate files. To write the velocities from your code, just execute:

#save_mpiio(comm, 'ux.npy', velocities[0])
#save_mpiio(comm, 'uy.npy', velocities[1])


def mpi_communicate(comm, cartcomm, f):

    leftSrc, leftDst = cartcomm.Shift(0, -1)
    rightSrc, rightDst = cartcomm.Shift(0, 1)
    upSrc, upDst = cartcomm.Shift(1, -1)
    downSrc, downDst = cartcomm.Shift(1, 1)

    # right
    recvbuf = f[:, 0, :].copy() # TODO: init once
    comm.Sendrecv(sendbuf=f[:, -1, :].copy(), dest=leftDst,
                  recvbuf=recvbuf, source=leftSrc)
    f[:, 0, :] = recvbuf

    # left
    recvbuf = f[:, -1, :].copy()  # TODO: init once
    comm.Sendrecv(sendbuf=f[:, 0, :].copy(), dest=rightDst,
                  recvbuf=recvbuf, source=rightDst)
    f[:, -1, :] = recvbuf

    # down
    recvbuf = f[0, :, :].copy()  # TODO: init once
    comm.Sendrecv(sendbuf=f[-1, :, :].copy(), dest=upDst,
                  recvbuf=recvbuf, source=upSrc)
    f[0, :, :] = recvbuf

    # up
    recvbuf = f[-1, :, :].copy()  # TODO: init once
    comm.Sendrecv(sendbuf=f[0, :, :].copy(), dest=downDst,
                  recvbuf=recvbuf, source=downSrc)
    f[-1, :, :] = recvbuf

    return f


if __name__ == '__main__':

    ####################################################################
    ######################## Initialization ############################
    ####################################################################

    # lattice dimensions
    nRows = 200
    nCols = 200
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
    timesteps = 1000

    # lattice
    f = np.zeros((nRows, nCols, nCh), dtype=float)
    # weights for initial lattice distribution
    w = np.array([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0])
    # velocity vector for matrix indices starting top left
    c = np.array([[0, 0], [1, 0], [0, -1], [-1, 0], [0, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]])

    # initial lattice distribution f
    f = f_init(f, w)

    # attenuation factor
    omega = 0.7 # TODO: compute reynolds number >! 1000 for turbulent flow
    assert 0 < omega <= 1.7, 'Limits of attenuation factor exceeded'

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

    dimSquare = 2
    dims = (dimSquare, dimSquare)
    # periodic communication disabled
    cartcomm = comm.Create_cart(dims, periods=False, reorder=False)

    localSplit = nRows // dimSquare

    ####################################################################
    ##################### Start with sliding lid #######################
    ####################################################################

    startWithSlidingLid = True

    if startWithSlidingLid:
        localGhost = localSplit + 2
        # initialize velocity only for top lid in x direction
        uTZero = np.zeros((localGhost, localGhost, 2))
        assert (abs(uTZero) < 0.1).all(), 'Limits of u exceeded'
        # set rho
        rhoTZero = np.ones((localGhost, localGhost))

        # Calculate lattice equilibrium according to given rho -> equal to f_eq
        f = calc_equilibrium(rhoTZero, uTZero, c, w)

    rhoScatter = rhoTZero.copy()
    uScatter = uTZero.copy()

    # time loop
    for i in range(timesteps):

        if (i + 1) % 50 == 0:
            print("\rTime {}/{}".format(i + 1, timesteps), end="")
            sys.stdout.flush()
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
        # mpi communication
        f = mpi_communicate(comm, cartcomm, f)

    save_mpiio(cartcomm, 'ux.npy', uScatter[1:-1, 1:-1, 0])
    save_mpiio(cartcomm, 'uy.npy', uScatter[1:-1, 1:-1, 1])


