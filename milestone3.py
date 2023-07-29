import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
from utils import density, velocity, streaming, c, equilibrium_distribution, collision_term, create_sinus_density, create_sinus_velocity


# Collision function with omega = 0.5
collision_func = lambda grid : collision_term(grid, 0.5)



def plot_rho_grid(density_grid, save=False, name='') :
    """
    plots the the density grid
    """
    density_grid = np.moveaxis(density_grid, 0, 1)
    plt.imshow(density_grid, cmap='Blues')
    plt.gca().invert_yaxis()
    plt.title('Grid')
    plt.colorbar()
    if save :
        plt.savefig('./figures/' + name + '.png')
    plt.show()



def animation(create_f, collision=None, frames=200, save=False, name='') :
    """
    animation function
    """
    rho_grid, u_grid = create_f()
    grid = equilibrium_distribution(rho_grid, u_grid)
    density_grid = density(grid)
    density_grid = np.moveaxis(density_grid, 0, 1)
    fig = plt.figure()
    im = plt.imshow(density_grid, animated=True, cmap='Blues')
    plt.gca().invert_yaxis()
    count = 0
    def update_grid(frame) :
        nonlocal grid, count
        count += 1
        if collision is not None :
            grid = streaming(grid, c, collision, test=True)
        else :
            grid = streaming(grid, c, collision=None, test=True)
        frame = density(grid)
        frame = np.moveaxis(frame, 0, 1)
        im.set_array(frame)
        print('frame :', count, "/", frames, end='\r')
        return im,
    animate = FuncAnimation(fig, update_grid, frames=frames)
    fig.colorbar(im)
    if save :
        animate.save('./figures/' + name + '.mp4', writer='ffmpeg')
    plt.show()



def plot_amplitude_velocity() :
    amplitude = []
    # initializing rho_grid with 1 and u_grid using the create_sinus_velocity fct
    rho_grid, u_grid = create_sinus_velocity()
    grid = equilibrium_distribution(rho_grid, u_grid)
    for _ in range(20000) :
        # Getting the amplitude of the velocity
        v = velocity(grid, density(grid))
        v_norms = np.linalg.norm(v, axis=0)
        amplitude.append(np.max(v_norms))
        # Roll and Collision functions
        grid = streaming(grid, c, collision_func)

    amplitude = np.array(amplitude)
    # PLot the amplitude
    plt.plot(amplitude)
    plt.ylabel("amplitude")
    plt.xlabel("time")
    plt.show()


def plot_wave_amplitude() :
    amplitude = []
    # Geth the density and velocity grid
    rho_grid, u_grid = create_sinus_velocity()
    # Creating the probability density grid
    grid = equilibrium_distribution(rho_grid, u_grid)
    for _ in range(20000) :
        # Getting the amplitude
        v = velocity(grid, density(grid))
        v_norms = np.linalg.norm(v, axis=0)
        amplitude.append(np.max(v_norms))
        # Roll and Collision functions
        grid = streaming(grid, c, collision_func)


    steps = [0, 1000, 5000, 10000, 19000]
    amplitude = np.array(amplitude)
    # Taking the a sample of the amplitudes
    new_amplitude = amplitude[steps]
    Ly = 300
    y = np.linspace(0, Ly, 300, endpoint=False)
    # Calculating the velocity on the x-axis for the different amplitudes
    u_x = [amplitude*np.sin((2*np.pi*y)/Ly) for amplitude in new_amplitude]
    # PLotting
    for idx, u in enumerate(u_x) :
        plt.plot(y, u, label='t = {}'.format(steps[idx]))
    plt.legend()
    plt.xlabel("X axis")
    plt.ylabel("Wave Amplitude")
    plt.savefig('figures/wave_amplitude.png')

    plt.show()


def plot_amplitude_density() :
    amplitude_d = []
    # Getting the density and velocity grid 
    rho_grid, u_grid = create_sinus_density()
    grid = equilibrium_distribution(rho_grid, u_grid)
    for _ in range(20000) :
        # Getting the aplitude
        d = density(grid)
        amplitude_d.append(np.max(d))
        # Roll and Collision functions
        grid = streaming(grid, c, collision_func)

    # Getting the peaks of the amplitudes
    peaks, _ = find_peaks(amplitude_d)
    amplitude_d = np.array(amplitude_d)
    # Plotting 
    plt.plot(amplitude_d[peaks])
    plt.ylabel("amplitude")
    plt.xlabel("time")
    plt.show()

def plot_viscosity() :
    # List of 20 different omegas that range between 0 and 2
    omegas = np.linspace(0, 2, 20, endpoint=False)[1:]
    viscosities = []
    for w in omegas :
        collision_func = lambda grid : collision_term(grid, w)
        amplitude_v = []
        rho_grid, u_grid = create_sinus_velocity()
        grid = equilibrium_distribution(rho_grid, u_grid)
        for _ in range(1000) :
            v = velocity(grid, density(grid))
            v_norms = np.linalg.norm(v, axis=0)
            amplitude_v.append(np.max(v_norms))
            # Roll and Collision functions
            grid = streaming(grid, c, collision_func)
        amplitude_v = np.array(amplitude_v)
        # Calculating the viscosity fot that omega
        viscosity = ((-1)*np.log((amplitude_v/amplitude_v[0]))*(u_grid.shape[2]**2))/(((2*np.pi)**2)*1000)
        viscosities.append(viscosity.mean())

    # Theoritical viscosity
    viscosities_theory = [(1/3)*((1/w)-(1/2)) for w in omegas]
    # Plotting
    plt.plot(omegas, viscosities, label="experiment")
    plt.plot(omegas, viscosities_theory, label="theory")
    plt.legend()
    plt.xlabel("omega")
    plt.ylabel("viscosity")
    plt.savefig("figures/viscosity.png")
    plt.show()


if __name__ == '__main__' :
    #animation(create_sinus_density, collision_func)  #rho.mp4 in figures/
    plot_amplitude_velocity()
    plot_amplitude_density()
    plot_wave_amplitude()
    plot_viscosity()
