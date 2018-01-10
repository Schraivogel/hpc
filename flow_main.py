import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import flow_func
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

if __name__ == "__main__":
    ####################################################################
    ######################## Initialization ############################
    ####################################################################

    # lattice dimensions
    nRows = 40
    nCols = 40
    nCh = 9

    # number of timesteps
    timesteps = 200

    # velocity vector for matrix indices starting top left
    c = np.array([[0, 0], [0, 1], [-1, 0], [0, -1], [1, 0], [-1, 1], [-1, -1], [1, -1], [1, 1]])
    # lattice
    f = np.zeros((nRows, nCols, nCh), dtype=float)
    # weights for initial lattice distribution
    w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

    # initial lattice distribution f
    f = flow_func.f_init(f, w)

    # attenuation factor
    omega = 0.1
    assert 0 < omega < 1.7, 'Limits of attenuation factor exceeded'

    # initialize shear wave decay factor
    epsilon = 0.01
    assert epsilon < 0.1, 'Limits of shear wave decay exceeded'

    ####################################################################
    ##################### Start with initial rho #######################
    ####################################################################

    # set density distribution offset
    rhoOffset = 0.8
    assert 0 < rhoOffset < 1.0, 'Limits of rhoT0 exceeded'
    # initial density distribution at t = 0
    rhoTZero = flow_func.init_rho(epsilon, rhoOffset, nRows, nCols)
    # set u to zero
    uTZero = np.zeros((nRows, nCols, 2))

    # Calculate lattice equilibrium according to given rho -> equal to f_eq
    f = flow_func.calc_equilibrium(rhoTZero, uTZero, c, w)
	print(rhoTZero)
    # sanity check if calculated rho from the lattice is equal to predefined rho at t = 0
    assert np.isclose(rhoTZero, flow_func.get_rho(f), rtol=1e-15, atol=1e-20, equal_nan=False).all(), 'Rho init failed'

    ####################################################################
    ############### Start with initial average velocity u ##############
    ####################################################################

    # initial density distribution at t = 0
    uTZero = flow_func.init_u(epsilon, nRows, nCols)
    assert (abs(uTZero) < 0.1).all(), 'Limits of u exceeded'
    # set rho to all ones
    rhoTZero = np.ones((nRows, nCols))

    # Calculate lattice equilibrium according to given rho -> equal to f_eq
    f = flow_func.calc_equilibrium(rhoTZero, uTZero, c, w)

    # sanity check if calculated u is equal to predefined u at t = 0
    # compute j, rho-> == 1, u == uTZero
    ####################################################################
    ######################### Scattering ###############################
    ####################################################################

    rhoScatter = rhoTZero.copy()
    uScatter = uTZero.copy()
    rowPlot = 0
    colPlot = 0
    plotDiscret = 20
    # Plotting
    # Two subplots, the axes array is 1-d
    fig = plt.figure()
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    # Turn off axis lines and ticks of the big subplot
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    # Set common labels
    ax.set_xlabel('common xlabel')
    ax.set_ylabel('common ylabel')

    ax1.set_title('Row of density rho')
    ax2.set_title('Column of verage velocity u')
    ax3.set_title('Values of verage velocity u at pi/4')
    # storage of u vector points
    uStore = np.zeros((1,0))

    # time loop
    for t in range(timesteps):

        if t % plotDiscret == 0:
            uStore = np.append(uStore, uScatter[nRows//4, colPlot, 0])
            ax1.plot(rhoScatter[rowPlot, :], label='t = %s' % t)
            print(rhoTZero)
            ax2.plot(uScatter[:, colPlot, 0], label='t = %s' % t)
            plt.draw()
            plt.pause(1e-4)
        # shift distribution f
        f = flow_func.shift_f(f)
        # get partial current density j
        j = flow_func.calc_j(c, f)
        # get current density rho
        rhoScatter = flow_func.get_rho(f)
        # get average velocity
        uScatter = flow_func.calc_avg_vel(rhoScatter, j)
        # get local equilibrium distributions
        feQ = flow_func.calc_equilibrium(rhoScatter, uScatter, c, w)
        # update distribution
        f += omega * (feQ - f)

    print('Rho after %f timesteps: %s' % (timesteps, rhoScatter[0, :]))

    ax3.plot(uStore)

    # Legend
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show(ax) # figure blocks -> must be at the end of the code!
    '''
    rho = np.einsum('ijk->jk', f)
    u = np.einsum('ai,ixy->axy',c,f)/rho
    '''
