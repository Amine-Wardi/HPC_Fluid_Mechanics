import numpy as np
from utils import streaming, c, equilibrium_distribution, density, velocity, collision_term, box_sliding_top_boundary
import matplotlib.pyplot as plt





def plot_velocity() :
    plt.figure()
    omega = 1.7
    # Collision fct with omeag = 1.7
    collision_func = lambda grid : collision_term(grid, omega)
    # Initializing the density adn velocity grid 
    rho_0 = np.ones((300, 300))
    u_0 = np.zeros((2, 300, 300))
    # Initializing the pribability density grid
    grid = equilibrium_distribution(rho_0, u_0)
    shape = grid.shape
    x = np.arange(shape[1])
    y = np.arange(shape[2])
    X, Y = np.meshgrid(x, y)
    for i in range(100000) :
        print(i, '/', 100000, end='\r')
        # Calculating the velocity
        u = velocity(grid, density(grid))
        # Streaming with collisiong fct, box sliding lid boundary conditions
        grid = streaming(grid, c, collision=collision_func, boundary=box_sliding_top_boundary, test=True)

    u_x = np.moveaxis(u[0], 0, 1)
    u_y = np.moveaxis(u[1], 0, 1)
    print(u_x.shape, u_y.shape, X.shape, Y.shape)
    plt.streamplot(X, Y, u_x, u_y)

    plt.legend()
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.show()
    plt.close()

if __name__ == '__main__' :
    plot_velocity()
