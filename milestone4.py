import numpy as np
from utils import streaming, c, equilibrium_distribution, density, velocity, collision_term, fixed_boundary_conditions, boundary_conditions
from milestone2 import create_grid_middle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation





def animate(create_f, collision=None, boundary=None, frames=400, save=False, name='') :
    grid = create_f()
    # Calclating the denity gird
    density_grid = density(grid)
    density_grid = np.moveaxis(density_grid, 0, 1)
    fig = plt.figure()
    # Plot the density grid
    im = plt.imshow(density_grid, animated=True, cmap='Blues')
    plt.gca().invert_yaxis()
    count = 0
    def update_grid(frame) :
        nonlocal grid, count
        count += 1
        # Streaming operator
        if collision is not None :
            grid = streaming(grid, c, collision, boundary=boundary, test=True)
        else :
            grid = streaming(grid, c, collision=None, boundary=boundary, test=True)
        frame = density(grid)
        frame = np.moveaxis(frame, 0, 1)
        im.set_array(frame)
        print('frame :', count, "/", frames, end='\r')
        return im,
    # Updating the frames for the number of times desired
    animate = FuncAnimation(fig, update_grid, frames=frames)
    if save :
        animate.save('./figures/' + name + '.mp4', writer='ffmpeg')
    plt.show()

def plot_velocity_by_step() :
    plt.figure()
    collision_func = lambda grid : collision_term(grid, omega=1)
    # Initializing the denisty and velocity grid
    rho_0 = np.ones((100, 100))
    u_0 = np.zeros((2, 100, 100))
    # Initializing the probability density grid
    grid = equilibrium_distribution(rho_0, u_0)
    for i in range(20001) :
        if i in [0, 500, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000] :
            # Calculating the velocity for the x-axis when x = 50
            v = velocity(grid, density(grid))[0, 50, :]
            # Plotting
            plt.plot(v, label=str(i))
        # Doing the steaming and collision and boundary functions
        grid = streaming(grid, c, collision=collision_func, boundary=boundary_conditions, test=True)

    plt.legend()
    plt.xlabel("Y axis")
    plt.ylabel("Velocity")
    plt.show()
    plt.close()


if __name__ == '__main__' :
    collision_func = lambda grid : collision_term(grid, omega=1)
    # animate(create_grid_middle, collision_func, boundary_conditions) # boundary.mp4
    plot_velocity_by_step()
