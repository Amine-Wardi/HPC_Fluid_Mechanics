import numpy as np
from mpi4py import MPI
from utils import c, streaming, equilibrium_distribution, density, velocity, collision_term, box_sliding_top_boundary, \
                  top_boundaries, bottom_boundaries, right_boundaries, left_boundaries, right_bottom_boundary, right_top_boundary, \
                  left_bottom_boundary, left_top_boundary, save_mpiio, communicate
import argparse
import time





if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    parser.add_argument('-gs', '--grid_size', nargs='+', type=int, default=(300, 300))
    parser.add_argument("-ts", "--time_steps", type=int, default=100000)

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    timesteps = args.time_steps

    # Let us take a square simulation domain that can be divided 
    # by the number of cores and results in a square subdomain 
    NX = args.grid_size[0]
    NY = args.grid_size[1]
    if NX < NY :
        node_shape_x = int(np.floor(np.sqrt(size*NX/NY)))
        node_shape_y = int(np.floor(size/node_shape_x))
        print('We have {} fields in x-direction and {} in y-direction'.format(node_shape_x, node_shape_y))
        print('How do the fractions look like?')
        print('NX/NY={} and node_shape_x/node_shape_y = {}\n'.format(NX/NY, node_shape_x/node_shape_y))
    elif NX > NY :
        node_shape_y = int(np.floor(np.sqrt(size*NY/NX)))
        node_shape_x = int(np.floor(size/node_shape_y))
        print('We have {} fields in x-direction and {} in y-direction'.format(node_shape_x, node_shape_y))
        print('How do the fractions look like?')
        print('NX/NY={} and node_shape_x/node_shape_y = {}\n'.format(NX/NY, node_shape_x/node_shape_y))
    elif NX == NY :
        node_shape_y = int(np.floor(np.sqrt(size)))
        node_shape_x = int(size/node_shape_y)
        if rank == 0 :
            print('In the case of equal size we divide the processes as {} and {}'.format(node_shape_x, node_shape_y))

    
    subgrid_shape_x = NX//node_shape_x+2
    subgrid_shape_y = NY//node_shape_y+2
    # boundary_k=[False,False,False,False] # This is for hard boundaries WHY 4  
    
    # We need a Cartesian communicator
    cartcomm = comm.Create_cart(dims=[node_shape_x, node_shape_y], periods=(False, False), reorder=False)
    rcoords = cartcomm.Get_coords(rank)

    right_src, right_dst = cartcomm.Shift(1, 1)
    left_src, left_dst = cartcomm.Shift(1, -1)
    up_src, up_dst = cartcomm.Shift(0, -1)
    down_src, down_dst = cartcomm.Shift(0, 1)

    L = [right_src, right_dst, left_src, left_dst, up_src, up_dst, down_src, down_dst]

    if rank == 0 : 
        print("Starting")
        start_time = time.time()


    print(f'Rank: {rank} is at coordinates ({rcoords[0]},{rcoords[1]})')


    # Get booleans to see if we are the in the boundaries of the whole grid or not
    # if we are the tight_dst shouldb be equal to -1, that's why we test if it is < 0
    right = right_dst < 0 
    left = left_dst < 0 
    up = up_dst < 0 
    down = down_dst < 0 

    omega = 1.7
    collision_func = lambda grid : collision_term(grid, omega)
    rho_0 = np.ones((subgrid_shape_x, subgrid_shape_y))
    u_0 = np.zeros((2, subgrid_shape_x, subgrid_shape_y))
    grid = equilibrium_distribution(rho_0, u_0)

    for i in range(timesteps) :
        if rank == 0 :
            print("rank :", rank, "timestep =", i, '/', timesteps, end='\r')
        u = velocity(grid, density(grid))

        grid = communicate(grid, cartcomm, L)
        # grid must diveded at least to 4 subgrids.
        if size == 1 :
            grid = streaming(grid, c, collision=collision_func, boundary=box_sliding_top_boundary)
        else :
            if down and left :
                grid = streaming(grid, c, collision=collision_func, boundary=left_bottom_boundary)
            elif down and right : 
                grid = streaming(grid, c, collision=collision_func, boundary=right_bottom_boundary)
            elif down : 
                grid = streaming(grid, c, collision=collision_func, boundary=bottom_boundaries)
            elif up and left :
                grid = streaming(grid, c, collision=collision_func, boundary=left_top_boundary)
            elif up and right :
                grid = streaming(grid, c, collision=collision_func, boundary=right_top_boundary)
            elif up :
                grid = streaming(grid, c, collision=collision_func, boundary=top_boundaries)
            elif left : 
                grid = streaming(grid, c, collision=collision_func, boundary=left_boundaries)
            elif right : 
                grid = streaming(grid, c, collision=collision_func, boundary=right_boundaries)
            else :
                grid = streaming(grid, c, collision=collision_func, boundary=None)


    if rank == 0 :
        end_time = time.time()
        print('{} iterations took {}s'.format(timesteps, end_time - start_time))

    rows, columns = u[0].shape
    vel = u[:, 1:rows-1, 1:columns-1]
    
    if rank == 0 :
        u_full_grid = np.zeros((comm.Get_size(), 2, subgrid_shape_x-2, subgrid_shape_y-2), dtype=np.float64)
    else:
        u_full_grid = None

    comm.Gather(np.ascontiguousarray(vel), u_full_grid, root = 0)
    if rank == 0 :
        with open('u_4.npy', 'wb') as file :
            np.save(file, u_full_grid)
