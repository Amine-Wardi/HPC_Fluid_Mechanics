import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpi4py import MPI



#--------------------------------------------------- Milestone 1 ---------------------------------------------------#

# 
c = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
              [0, 0, 1, 0, -1, 1, 1, -1, -1]])

def density(f: np.ndarray) -> np.ndarray :
    """
    :param f: propability density grid (3D NumPy Array)
    :return: 2D NumPy Array where we summed the 0-axis of f
    """
    return f.sum(axis=0)

def velocity(f: np.ndarray, rho: np.ndarray) -> np.ndarray :
    """
    :param f: propability density grid (3D NumPy Array)
    :param rho: density grid, result of density(f)
    :return: 3D NumPy Array that represents the velocity grid
    """
    return (1/rho)*np.einsum("ij, jkl->ikl", c, f)


def plot_grid(grid, save=False, name='') :
    """
    plots the grid, with the option to save it in the firgures directory
    """
    density_grid = density(grid)
    density_grid = np.moveaxis(density_grid, 0, 1)
    plt.imshow(density_grid, cmap='Blues')
    plt.gca().invert_yaxis()
    plt.title('Grid')
    plt.colorbar()
    if save :
        plt.savefig('./figures/' + name + '.png')
    plt.show()


def streaming(grid: np.ndarray, c: np.ndarray, collision=None, boundary=None, pressure=None, test=False) -> np.ndarray :
    """
    :param grid: probablity density grid
    :param c: the directions
    :param collision: collision function
    :param boundary: boundary function
    :param pressure: pressure boudary function
    :param test: boolean to see if we test if the mass is conserved after all these functions
    :return: the grid
    """
    res = np.copy(grid)
    # copy to use in the boundary function
    copy = np.copy(grid)
    if pressure is not None :
        res = pressure(res)
    for i in range(9) :
        res[i, :, :] = np.roll(res[i, :, :], shift=(c[0, i], c[1, i]), axis=(0, 1))
    if boundary is not None :
        res = boundary(res, copy)
    if collision is not None :
        res = collision(res)

    if test :
        try :
            assert np.isclose(grid.sum(), res.sum(), rtol=0.01) 
        except :
            print('Error : Mass is not conserved after streaming')
            raise AssertionError
    return res



def animation_grid(create_f, collision=None, frames=200, save=False, name='') :
    """
    This function was made by using Chat GPT 3
    """
    grid = create_f()
    # Calculating density grid
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
            grid = streaming(grid, c, collision, test=True)
        else :
            grid = streaming(grid, c, collision=None, test=True)
        frame = density(grid)
        frame = np.moveaxis(frame, 0, 1)
        im.set_array(frame)
        print('frame :', count, "/", frames, end='\r')
        return im,
    # Updating the frames for the number of times desired
    animate = FuncAnimation(fig, update_grid, frames=frames)
    fig.colorbar(im)
    if save :
        animate.save('./figures/' + name + '.mp4', writer='ffmpeg')
    plt.show()

#--------------------------------------------------- Milestone 2 ---------------------------------------------------#

w_i = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# function that calculates the density grid
rho = lambda grid : density(grid)

# function that calculates the velocity grid
u = lambda grid : velocity(grid, rho(grid))

def equilibrium_distribution(rho_grid: np.ndarray, u_grid: np.ndarray) -> np.ndarray : 
    """
    :param rho_grid: the density grid
    :param u_grid: the velocity grid
    :return: the probablity density grid
    """
    product = np.einsum('ji, jkl -> ikl', c, u_grid) 
    norm = np.linalg.norm(u_grid, axis=0)**2 
    return w_i[:, np.newaxis, np.newaxis] * rho_grid[np.newaxis, :, :] * (1 + 3 * product + (9/2) * \
             product**2 - (3/2) * norm[np.newaxis, :, :])



def collision_term(grid: np.ndarray, omega: float) -> np.ndarray :
    """
    :param grid: probablity density grid
    :param omega: the relaxation term, it should be between 0 and 2
    :return: the probablity grid after the collision
    """
    return grid + omega*(equilibrium_distribution(rho(grid), u(grid)) - grid) 


#--------------------------------------------------- Milestone 3 ---------------------------------------------------#


def create_sinus_density(shape=(9, 300, 300), epsilon=0.01, rho_0=1) :
    """
    :param shape: the shape of the probality grid we want
    :param epsilon: the initial amplitude
    :pram rho_0: the intial value of the density grid
    :return: the calculated density grid and a velcity gird that is set to 0
    """
    u_grid = np.zeros((2, shape[1], shape[2]))
    Lx = shape[1]
    x = np.linspace(0, Lx, shape[1], endpoint=False)
    # Calculation the density grid
    rho = rho_0 + epsilon * np.sin(2 * np.pi * x / Lx)
    # Getting the right shape for the density grid
    rho_grid = np.tile(rho, (shape[2], 1)).T

    return rho_grid, u_grid


def create_sinus_velocity(shape=(9, 300, 300), epsilon=0.1) :
    """
    :param shape: the shape of the probality grid we want
    :param epsilon: the initial amplitude
    :return: the calculated velocity grid and a density gird that is set to 1
    """
    rho_grid = np.ones((shape[1], shape[2]))
    Ly = shape[2]
    y = np.linspace(0, Ly, shape[2], endpoint=False)
    u = epsilon*np.sin((2*np.pi*y)/Ly)
    # Getting the right shape for u_x and u_y
    u_x = np.broadcast_to(u, (shape[1],) + u.shape)
    u_y = np.zeros((shape[1], shape[2]))
    u_grid = np.empty((2, shape[1], shape[2]))
    u_grid[0] = u_x
    u_grid[1] = u_y
    return rho_grid, u_grid

#--------------------------------------------------- Milestone 4 ---------------------------------------------------#

# Velocity of the wall
u_w = np.array([0.1, 0])

# Top and bottom boundary indices
top_boundary = [5, 2, 6]
bottom_boundary = [7, 4, 8]


def fixed_boundary_conditions(grid: np.ndarray, copy: np.ndarray) -> np.ndarray :
    """
    :param grid: the probability density grid
    :param copy: copy of the grid before the np.roll in the streaming function
    :return: the grid after applying the fixed boundaries rules for the top and bottom wall
    """
    for x in range(grid.shape[1]) :
        # Copy of the top row
        top = np.copy(copy[:, x, -1])
        # Copy of the bottom row
        bottom = np.copy(copy[:, x, 0])

        # Copying the 5, 2, 6 indices of the top row to the 7, 4, 8 indices of the top row
        grid[bottom_boundary, x, -1] = top[top_boundary]
        # Copying the 7, 4, 8 indices of the bottom row to the 5, 2, 6 indices of the bottom row
        grid[top_boundary, x, 0] = bottom[bottom_boundary]

    return grid


def boundary_conditions(grid: np.ndarray, copy: np.ndarray) -> np.ndarray :
    """
    :param grid: the probability density grid
    :param copy: copy of the grid before the np.roll in the streaming function
    :return: the grid after applying the fixed boundary for the bottom wall and a moving top wall
    """
    c_s_square = 1/3
    for x in range(grid.shape[1]) :
        top = np.copy(copy[:, x, -1])
        bottom = np.copy(copy[:, x, 0])
        rho_n = density(grid)
        avg_density = rho_n.mean()
        grid[bottom_boundary, x, -1] = [top[5] - 2*w_i[5]*avg_density*np.dot(c[:, 5], u_w)/c_s_square, \
                                        top[2] - 2*w_i[2]*avg_density*np.dot(c[:, 2], u_w)/c_s_square, \
                                        top[6] - 2*w_i[6]*avg_density*np.dot(c[:, 6], u_w)/c_s_square]

        grid[top_boundary, x, 0] = bottom[bottom_boundary]

    return grid


#--------------------------------------------------- Milestone 5 ---------------------------------------------------#


def pressure_condtions(grid: np.ndarray) -> np.ndarray :
    """
    :param grid: the probabilty density grid
    :return: the grid after the pressure condtions applied
    """
    channels = [1, 5, 8]
    shape = grid.shape
    c_s_squared = 1/3
    p_in = 0.03
    p_out = 0.3
    delta_p = p_out - p_in

    # Calculation density and velocity grids
    rho = density(grid)
    u = velocity(grid, rho)

    # Getting a 3D grid (2, shape[1], shape[2]) of the velcoity grid when x = shape[1]-1, so we are repeating it shape[1] times
    u_N = np.repeat(u[:, -2, :][:, np.newaxis, :], u.shape[1], axis=1)
    # Getting a 3D grid (2, shape[1], shape[2]) of the velcoity grid when x = 1, so we are repeating it shape[1] times
    u_1 = np.repeat(u[:, 1, :][:, np.newaxis, :], u.shape[1], axis=1)

    # Calculating rho_out using p_out
    rho_out = np.full((shape[1], shape[2]), p_out/c_s_squared)
    # Calculating rho_in
    rho_in = np.full((shape[1], shape[2]), (p_out+delta_p)/c_s_squared)

    # Getting the probability density grid using rho and u
    f_eq = equilibrium_distribution(rho, u)

    # Getting the probability density grid using rho_out and u_1
    f_eq_out = equilibrium_distribution(rho_out, u_1)[:, 1, :]
     # Getting the probability density grid using rho_in and u_N
    f_eq_in = equilibrium_distribution(rho_in, u_N)[:, -2, :]

    tmp1 = f_eq_out +  grid[:, 1, :] - f_eq[:, 1, :]
    tmp2 = f_eq_in +  grid[:, -2, :] - f_eq[:, -2, :]

    grid[channels, -1, :] = tmp1[channels, :]
    grid[channels, 0, :] = tmp2[channels, :]

    return grid


#--------------------------------------------------- Milestone 6 ---------------------------------------------------#



right_boundary = [5, 1, 8]
left_boundary = [6, 3, 7]


def box_sliding_top_boundary(grid: np.ndarray, copy: np.ndarray) -> np.ndarray :
    """
    :param grid: the probability density grid
    :param copy: copy of the grid before the np.roll in the streaming function
    :return: the grid after applying the fixed boundary for the bottom, right and left walls and a moving top wall
    """
    c_s_square = 1/3
    rho_n = density(grid)
    avg_density = rho_n.mean()
    # Top and bottom changes
    for x in range(grid.shape[1]) :
        top = np.copy(copy[:, x, -1])
        bottom = np.copy(copy[:, x, 0])

        grid[bottom_boundary, x, -1] = [top[5] - 2*w_i[5]*avg_density*np.dot(c[:, 5], u_w)/c_s_square, \
                                        top[2] - 2*w_i[2]*avg_density*np.dot(c[:, 2], u_w)/c_s_square, \
                                        top[6] - 2*w_i[6]*avg_density*np.dot(c[:, 6], u_w)/c_s_square]

        grid[top_boundary, x, 0] = bottom[bottom_boundary]

    # Right and left changes
    for y in range(grid.shape[2]) :
        right = np.copy(copy[:, -1, y])
        left = np.copy(copy[:, 0, y])

        grid[left_boundary, -1, y] = right[right_boundary]
        grid[right_boundary, 0, y] = left[left_boundary]

    return grid


#--------------------------------------------------- Milestone 7 ---------------------------------------------------#


def top_boundaries(grid: np.ndarray, copy: np.ndarray) -> np.ndarray :
    """
    :param grid: the probability density grid
    :param copy: copy of the grid before the np.roll in the streaming function
    :return: the grid after applying a moving top wall
    """
    c_s_square = 1/3
    rho_n = density(grid)
    avg_density = rho_n.mean()
    for x in range(grid.shape[1]) :
        top = np.copy(copy[:, x, -1])

        grid[bottom_boundary, x, -1] = [top[5] - 2*w_i[5]*avg_density*np.dot(c[:, 5], u_w)/c_s_square, \
                                        top[2] - 2*w_i[2]*avg_density*np.dot(c[:, 2], u_w)/c_s_square, \
                                        top[6] - 2*w_i[6]*avg_density*np.dot(c[:, 6], u_w)/c_s_square]


    return grid


def bottom_boundaries(grid: np.ndarray, copy: np.ndarray) -> np.ndarray :
    """
    :param grid: the probability density grid
    :param copy: copy of the grid before the np.roll in the streaming function
    :return: the grid after applying the fixed boundary for the bottom wall
    """
    for x in range(grid.shape[1]) :
        bottom = np.copy(copy[:, x, 0])
        grid[top_boundary, x, 0] = bottom[bottom_boundary]

    return grid


def left_boundaries(grid: np.ndarray, copy: np.ndarray) -> np.ndarray :
    """
    :param grid: the probability density grid
    :param copy: copy of the grid before the np.roll in the streaming function
    :return: the grid after applying the fixed boundary for the left wall
    """
    for y in range(grid.shape[2]) :
        left = np.copy(copy[:, 0, y])
        grid[right_boundary, 0, y] = left[left_boundary]
    return grid


def right_boundaries(grid: np.ndarray, copy: np.ndarray) -> np.ndarray :
    """
    :param grid: the probability density grid
    :param copy: copy of the grid before the np.roll in the streaming function
    :return: the grid after applying the fixed boundary for the right wall
    """
    for y in range(grid.shape[2]) :
        right = np.copy(copy[:, -1, y])
        grid[left_boundary, -1, y] = right[right_boundary]
    return grid

def left_top_boundary(grid: np.ndarray, copy: np.ndarray) -> np.ndarray :
    """
    :param grid: the probability density grid
    :param copy: copy of the grid before the np.roll in the streaming function
    :return: the grid after applying the fixed boundary for left wall and a moving top wall
    """
    grid = left_boundaries(grid, copy)
    grid = top_boundaries(grid, copy)

    return grid

def left_bottom_boundary(grid: np.ndarray, copy: np.ndarray) -> np.ndarray :
    """
    :param grid: the probability density grid
    :param copy: copy of the grid before the np.roll in the streaming function
    :return: the grid after applying the fixed boundary for the bottom and left walls
    """
    grid = left_boundaries(grid, copy)
    grid = bottom_boundaries(grid, copy)

    return grid

def right_top_boundary(grid: np.ndarray, copy: np.ndarray) -> np.ndarray :
    """
    :param grid: the probability density grid
    :param copy: copy of the grid before the np.roll in the streaming function
    :return: the grid after applying the fixed boundary for the right and a moving top wall
    """
    grid = right_boundaries(grid, copy)
    grid = top_boundaries(grid, copy)

    return grid

def right_bottom_boundary(grid: np.ndarray, copy: np.ndarray) -> np.ndarray :
    """
    :param grid: the probability density grid
    :param copy: copy of the grid before the np.roll in the streaming function
    :return: the grid after applying the fixed boundary for the bottom and right walls
    """
    grid = right_boundaries(grid, copy)
    grid = bottom_boundaries(grid, copy)

    return grid




def communicate(grid: np.ndarray, cartcomm: MPI.Cartcomm, L: list) -> np.ndarray :
    """
    :param grid: the probability density grid
    :param cartcomm: The cartesian communicator
    :param L: List that holds all the source and destination for all the directions
    :return: the probability grid after communication between the processors 
    """

    right_src, right_dst, left_src, left_dst, up_src, up_dst, down_src, down_dst = L
    
    # Send to right which is destination rigth and receive from left which is source right
    recv_buffer = np.copy(grid[:, :, 0])
    send_buffer = np.copy(grid[:, :, -2])
    cartcomm.Sendrecv(send_buffer, up_dst, recvbuf=recv_buffer, source=up_src)
    grid[:, :, 0] = recv_buffer

    # Send to the bottom and receive from the top
    recv_buffer = np.copy(grid[:, :, -1])
    send_buffer = np.copy(grid[:, :, 1])
    cartcomm.Sendrecv(send_buffer, down_dst, recvbuf=recv_buffer, source=down_src)
    grid[:, :, -1] = recv_buffer

    # Send to the left and receive from the right
    recv_buffer = np.copy(grid[:, -1, :])
    send_buffer = np.copy(grid[:, 1, :])
    cartcomm.Sendrecv(send_buffer, left_dst, recvbuf=recv_buffer, source=left_src)
    grid[:, -1, :] = recv_buffer

    # Send to the right and receive from the left
    recv_buffer = np.copy(grid[:, 0, :])
    send_buffer = np.copy(grid[:, -2, :])
    cartcomm.Sendrecv(send_buffer, right_dst, recvbuf=recv_buffer, source=right_src)
    grid[:, 0, :] = recv_buffer

    return grid