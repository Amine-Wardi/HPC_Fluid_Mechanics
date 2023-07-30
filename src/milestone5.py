import numpy as np
from utils import streaming, c, equilibrium_distribution, density, velocity, collision_term, fixed_boundary_conditions, pressure_condtions
import matplotlib.pyplot as plt




def plot_velocity() :
    plt.figure()
    omega = 1
    collision_func = lambda grid : collision_term(grid, omega)
    # Initializing the density adn velocity grid 
    rho_0 = np.ones((300, 100))
    u_0 = np.zeros((2, 300, 100))
    # Initializing the pribability density grid
    grid = equilibrium_distribution(rho_0, u_0)
    shape = grid.shape
    # Calculating delta P
    delta_p = (0.03 - 0.3)/shape[1]
    # Calculating the theoritical viscosity
    viscosity = (1/3)*((1/omega)-(1/2))
    for i in range(2001) :
        print(i, '/', 2000, end='\r')
        if i in [0, 500, 600, 700, 900, 1000, 1500, 2000] :
            # Calculating the velocity for the x-axis when x = 50
            v = velocity(grid, density(grid))[0, 150, :]
            # Plotting that velocity
            plt.plot(v, label=str(i))
        # Streaming with collisiong fct, fixed boundary conditions and pressure conditions
        grid = streaming(grid, c, collision=collision_func, boundary=fixed_boundary_conditions, \
                         pressure=pressure_condtions, test=True)

    density_mean = np.mean(density(grid))
    # Theoritical velocity
    theory_velocity = np.array([(1/2)*(delta_p/(9*viscosity*density_mean))*y*(y-shape[2]) for y in range(shape[2])])
    plt.plot(theory_velocity, label="theory")
    plt.legend()
    plt.xlabel("Y axis")
    plt.ylabel("Velocity")
    plt.show()
    plt.close()


if __name__ == '__main__' :
    plot_velocity()






