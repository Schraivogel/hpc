#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

dim = 4
ux = np.load('ux_dim_{}.npy'.format(dim)).T
uy = np.load('uy_dim_{}.npy'.format(dim)).T

x = np.arange(ux.shape[1])
y = np.arange(uy.shape[0])
X, Y = np.meshgrid(x, y)

if __name__ == '__main__':
    plt.close()
    plt.streamplot(X, Y, ux, uy, color='b')
    plt.ylim(len(Y), 0)
    #plt.savefig('mpi_u.pdf')
    plt.show()
