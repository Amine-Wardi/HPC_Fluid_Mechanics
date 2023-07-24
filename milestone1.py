import numpy as np
from utils import c, streaming, plot_grid, animation_grid



np.random.seed(1234)


def create_f(shape=(9, 300, 300), num=18000) :
    """
    creates the probablity density grid of shape = (shape[1], shape[2]) and with num non-null points
    """
    grid = np.zeros(shape)
    count = 0
    while count < num :
        x = np.random.randint(0, shape[1])
        y = np.random.randint(0, shape[2])
        if abs(np.sum(grid[:, x, y]) - 1) >= 10**-2 :
            grid[:, x, y] = np.random.rand(shape[0])
            grid[:, x, y] = grid[:, x, y]/np.sum(grid[:, x, y])
            count += 1
    return grid


def create_test_grid():
    shape = (9, 30, 30)
    grid = np.zeros(shape)
    for x in range(shape[1]) :
        for y in range(shape[2]) :
            grid[:, x, y] = np.array([1/9 for _ in range(shape[0])])

    grid[:, 15, 15] = np.array([grid[i, 15, 15]+(i+1)/9 for i in range(9)])

    return grid 

if __name__ == '__main__' :
    grid_example = create_f((9, 15, 10), int(10*15*0.2))
    plot_grid(grid_example)
    num = 1
    test_grid = create_test_grid()
    plot_grid(test_grid)
    test_grid = streaming(test_grid, c, test=True)
    plot_grid(test_grid)
    # animation_grid(create_f) #For the this animation see animation.mp4 in figures/