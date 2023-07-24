import numpy as np
from utils import plot_grid, animation_grid, collision_term






def create_grid_middle():
    """
    creates the probability density grid of shape = (shape[1], shape[2]) with the middle 90 squares having different values
    """
    shape = (9, 300, 300)
    grid = np.zeros(shape)
    for x in range(shape[1]) :
        for y in range(shape[2]) :
            grid[:, x, y] = np.array([1/9 for _ in range(shape[0])])

    for x in range(100, 200) :
        for y in range(150, 250) :
            grid[:, x, y] = np.array([grid[i, x, y]+0.1 for i in range(shape[0])])

    return grid


collision_func = lambda grid : collision_term(grid, 0.5)

if __name__ == '__main__' :
    f = create_grid_middle()
    plot_grid(f)
    # animation_grid(create_grid_middle, collision_func) # For this animation see animation_collision.mp4 in figures/