import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI


def shift_latt(grid, borders):

	saved_latt = np.ma.masked_where(mask is True, latt)
	
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

	#borders = [top, right, bottom, left]
	
	#TOP
	if borders[0] == True:
		grid[1,:,4] = saved_latt[1,:,2]
		grid[1,1:,7] = saved_latt[1,1:,5]
		grid[1,:,8] = saved_latt[1,:,6]
		
	#RIGHT
	if borders[1] == True:
		grid[:,-1,3] = saved_latt[:,-1,1]
		grid[:-1,-1,7] = saved_latt[:-1,-1,5]
		grid[1:,-1,6] = saved_latt[1:,-1,8]
		
	#BOTTOM
	if borders[2] == True:
		grid[-1,:,2] = saved_latt[-1,:,4]
		grid[-1,:-1,5] = saved_latt[-1,:-1,7]
		grid[-1,:,6] = saved_latt[-1,:,8]

	#LEFT
	if borders[3] ==  True:
		grid[:,0,1] = saved_latt[:,0,3]
		grid[1:,0,5] = saved_latt[1:,0,7]
		grid[:-1,0,8] = saved_latt[-1:,0,6]


        # handle corners
	grid[1,1,5] = saved_latt[1,1,5]
	grid[-2,-2,5] = saved_latt[-2,-2,5]

	grid[1,-2,6] = saved_latt[1,-2,6]
	grid[-2,1,6] = saved_latt[-2,1,6]

	grid[1,1,7] = saved_latt[1,1,7]
	grid[-2,-2,7] = saved_latt[-2,-2,7]

	grid[1,-2,8] = saved_latt[1,-2,8]
	grid[-2,1,8] = saved_latt[-2,1,8]

    	return grid 

def sum_occ(mat):
    occ = np.sum(mat, axis=2)
    return occ

def calc_j(c, mat):
    #j = np.dot(mat, c)

    # Alternative
    j1 = np.einsum('abc, cd-> abd', mat, c)

    return j1

def calc_avg_vel(rho, j):
    #u = j / rho.reshape(rows, cols, 1)

    # Alternative
    u = (j.T / rho.T).T

    return u

def calc_equilibrium(rho, u, c, w):
	cu = np.einsum('ab, cdb -> cda', c, u)
	cu2 = cu**2
	u2 = np.einsum('abc, abc -> ab', u, u)
	
	f_eq = np.zeros((rho.shape[0], rho.shape[1], q))
	
	f_eq[:,:,0] = 4.0 / 9.0 * rho * (1 + 3 * cu[:,:,0] + 9.0/2.0 * cu2[:,:,0] - 3.0/2.0 * u2)
	f_eq[:,:,1] = 1.0 / 9.0 * rho * (1 + 3 * cu[:,:,1] + 9.0/2.0 * cu2[:,:,1] - 3.0/2.0 * u2)
	f_eq[:,:,2] = 1.0 / 9.0 * rho * (1 + 3 * cu[:,:,2] + 9.0/2.0 * cu2[:,:,2] - 3.0/2.0 * u2)
	f_eq[:,:,3] = 1.0 / 9.0 * rho * (1 + 3 * cu[:,:,3] + 9.0/2.0 * cu2[:,:,3] - 3.0/2.0 * u2)
	f_eq[:,:,4] = 1.0 / 9.0 * rho * (1 + 3 * cu[:,:,4] + 9.0/2.0 * cu2[:,:,4] - 3.0/2.0 * u2)
	f_eq[:,:,5] = 1.0 / 36.0 * rho * (1 + 3 * cu[:,:,5] + 9.0/2.0 * cu2[:,:,5] - 3.0/2.0 * u2)
	f_eq[:,:,6] = 1.0 / 36.0 * rho * (1 + 3 * cu[:,:,6] + 9.0/2.0 * cu2[:,:,6] - 3.0/2.0 * u2)
	f_eq[:,:,7] = 1.0 / 36.0 * rho * (1 + 3 * cu[:,:,7] + 9.0/2.0 * cu2[:,:,7] - 3.0/2.0 * u2)
	f_eq[:,:,8] = 1.0 / 36.0 * rho * (1 + 3 * cu[:,:,8] + 9.0/2.0 * cu2[:,:,8] - 3.0/2.0 * u2)
	
	return f_eq

def addSlidingLidVelocity(grid, borders):
	if borders[0] == True:
		f0 = grid[1,:,0]
		f1 = grid[1,:,1]
		f2 = grid[1,:,2]
		f3 = grid[1,:,3]
		f5 = grid[1,:,5]
		f6 = grid[1,:,6]
	
		rhoWall = f1 + f3 + f0 + 2 * (f6 + f2 + f5)
		vLid = 0.1
	
		ch = 7
		#grid[0,:,ch] -= 2 * w[ch] * rhoWall * (cu[0,:,ch] / cS**2)
		grid[1,:,ch] -= 6 * w[ch] * rhoWall * vLid
	
		ch = 8
		#grid[0,:,ch] -= 2 * w[ch] * rhoWall * (cu[0,:,ch] / cS**2)
		grid[1,:,ch] += 6 * w[ch] * rhoWall * vLid

	return grid

def mask_vortex(grid):

	# Top row
	mask[1,:,2] = True
	mask[1,:,5] = True
	mask[1,:,6] = True
	
	# Bottom row
	mask[-2,:,4] = True
	mask[-2,:,7] = True
	mask[-2,:,8] = True
	
	# Left column
	mask[:,1,3] = True
	mask[:,1,6] = True
	mask[:,1,7] = True
	
	# Right column
	mask[:,-2,1] = True
	mask[:,-2,5] = True
	mask[:,-2,8] = True

	return mask  

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
    mpitype = MPI.__TypeDict__[g_kl.dtype.char]
    filetype = mpitype.Create_vector(g_kl.shape[0], g_kl.shape[1], ny)
    filetype.Commit()
    file.Set_view(header_len + (offsety+offsetx)*mpitype.Get_size(),
                  filetype=filetype)
    file.Write_all(g_kl.copy())
    filetype.Free()
    file.Close()


##########################################################
rows = 200
cols = 200
q = 9
timesteps = 10000


##############################

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print('Rank {}/{} is alive.'.format(rank, size))

dims = (2,2)
# Create CPU Cluster with given dims
cartcomm = comm.Create_cart(dims, periods=(False,False), reorder=(False,False))

localRows = rows // dims[0]
localCols = cols // dims[1]

#print(localRows)
#print(localCols)

############ calculation not necessary
#startRows = rank * localRows
#endRows = (rank+1) * localRows

startCols = rank * localCols
endCols = (rank+1) * localCols


##############################

doBounce = True

c = np.array([[0,0], [0,1], [-1,0], [0,-1], [1,0], [-1,1], [-1,-1], [1,-1], [1,1]])

latt = np.zeros((localRows, localCols, q))

#print(latt.shape)

# initial lattice occupation
latt[:,:,0] = 4.0 / 9.0
latt[:,:,1] = 1.0 / 9.0
latt[:,:,2] = 1.0 / 9.0
latt[:,:,3] = 1.0 / 9.0
latt[:,:,4] = 1.0 / 9.0
latt[:,:,5] = 1.0 / 36.0
latt[:,:,6] = 1.0 / 36.0
latt[:,:,7] = 1.0 / 36.0
latt[:,:,8] = 1.0 / 36.0

#ch = 3
w = np.array([4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0])

doBounce = True

# add ghost cells
latt = np.zeros((latt.shape[0]+2, latt.shape[1]+2, latt.shape[2]))

# create empty temporary lattice in which the old values (before shifiting) are stored
saved_latt = np.zeros((latt.shape))
# create empty mask lattice
mask = np.zeros((latt.shape))
# create mask for shift_latt function in order to know which channels have to be bounced #back at which position
mask = mask_vortex(latt)

#print(mask.shape)


rho = np.ones((latt.shape[0], latt.shape[1]))
u = np.zeros((latt.shape[0], latt.shape[1], 2))
latt = calc_equilibrium(rho, u, c, w)
omega = 0.3

##########################################


# calculating the borders of each rank

borders = np.zeros(4 ,dtype=bool)

rightSrc, rightDest = cartcomm.Shift(0, 1)
leftSrc, leftDest = cartcomm.Shift(0, -1)
upSrc, upDest = cartcomm.Shift(1, -1)
downSrc, downDest = cartcomm.Shift(1, 1)

#borders = [top, right, bottom, left]
if upDest == -2:
	borders[0] = True

if rightDest == -2:
	borders[1] = True
	
if downDest == -2:
	borders[2] = True
	
if leftDest == -2:
	borders[3] = True

print ("Rank: " + str(rank) + str(borders) + str(cartcomm.Get_coords(rank)))

for t in range(timesteps):
	
	if ((t+1) % 10 == 0):
		print("\rTime {}/{}".format(t + 1, timesteps))
	
	
	#communicate
	
	#left
	recvbuf = latt[:,-1,:].copy()
	comm.Sendrecv(sendbuf=latt[:,1,:].copy(),dest=leftDest,recvbuf=recvbuf,source=leftSrc)
	latt[:, -1, :] = recvbuf
	
	#down
	recvbuf = latt[0,:,:].copy()
	comm.Sendrecv(sendbuf=latt[-2,:,:].copy(),dest=downDest,recvbuf=recvbuf,source=downSrc)
	latt[0,:,:] = recvbuf
	
	#right
	recvbuf = latt[:,0,:].copy()
	comm.Sendrecv(sendbuf=latt[:,-2,:].copy(),dest=rightDest,recvbuf=recvbuf,source=rightSrc)	
	latt[:,0,:] = recvbuf
	
	#up
	recvbuf = latt[-1,:,:].copy()
	comm.Sendrecv(sendbuf=latt[1,:,:].copy(),dest=upDest,recvbuf=recvbuf,source=upSrc)	
	latt[-1,:,:] = recvbuf
	
	# shift and everything
	latt = shift_latt(latt, borders)
	latt = addSlidingLidVelocity(latt, borders)
	j = calc_j(c, latt)
	rho = sum_occ(latt)
	u = calc_avg_vel(rho, j)
	#print(u[0,:,0])
	f_eq = calc_equilibrium(rho, u, c, w)
	latt += omega * (f_eq - latt)
	
ux = u[1:-1,1:-1,1].T
uy = u[1:-1,1:-1,0].T

save_mpiio(cartcomm,"ux_npy", ux)
save_mpiio(cartcomm,"uy_npy", uy)