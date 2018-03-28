import numpy as np
import matplotlib.pyplot as plt

savePlot = True

ux = np.load('ux_npy').T
uy = np.load('uy_npy').T

x = np.arange(ux.shape[1])
y = np.arange(uy.shape[0])
X, Y = np.meshgrid(x,y)

print(ux)
print(uy)

plt.figure()
plt.streamplot(X, Y, ux, uy, color='b')
plt.ylim(len(Y), 0)
plt.xlabel('Lattice size [nx]')
plt.ylabel('Lattice size [ny]')
if savePlot == True:
    plt.savefig("Report/Vortex_Streamplot_Parallel" + str(200) + "_" + str(200) + "_" + str(10000) + "omega=0.3_4cores.pdf")
plt.show()