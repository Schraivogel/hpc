#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-timesteps', type=int, default=1000, help='Number of timesteps')
parser.add_argument('-num_procs', type=int, default=4, help='Number of processes')
parser.add_argument('--nemo', action='store_true')
parser.add_argument('--save', action='store_true')
a = parser.parse_args()

if a.save: plt.switch_backend('Agg')

path = os.environ['HOME'] + '/results/hpc/' if a.nemo else 'W:/results/hpc/'
postfix = '_np_{}_ts_{}.npy'.format(a.num_procs, a.timesteps)

ux = np.load(path + 'ux' + postfix).T
uy = np.load(path + 'uy' + postfix).T

x = np.arange(ux.shape[1])
y = np.arange(uy.shape[0])
X, Y = np.meshgrid(x, y)


if __name__ == '__main__':
    print('Plot u*' + postfix)
    plt.close()
    fig, ax = plt.subplots()
    ax.streamplot(X, Y, ux, uy, color='b')
    ax.set_ylim([len(Y), 0])
    ax.set_title('Average velocity $\mathbf{u}(\mathbf{r})$')
    ax.set_xlabel('$L_x$')
    ax.set_ylabel('$L_y$')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)

    if a.save: plt.savefig('mpi_u' + postfix[:-4] + '.pdf')
    plt.show()
