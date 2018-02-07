import numpy as np
import matplotlib.pyplot as plt

savePlot = False

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
if savePlot == True:
    plt.savefig("Vortex_Streamplot_" + str(x) + "_" + str(y) + "_" + str(1) + ".pdf")
plt.show()