import numpy as np
import matplotlib.pyplot as plt




if __name__ == '__main__' :
    with open('u_4.npy', 'rb') as file :
        vel = np.load(file)

    vel_0 = vel[0]
    vel_1 = vel[1]
    vel_2 = vel[2]
    vel_3 = vel[3]
    
    u_x0 = vel_0[0]
    u_x1 = vel_1[0]
    u_x2 = vel_2[0]
    u_x3 = vel_3[0]
    u_y0 = vel_0[1]
    u_y1 = vel_1[1]
    u_y2 = vel_2[1]
    u_y3 = vel_3[1]

    u_x = np.zeros((300, 300))
    u_y = np.zeros((300, 300))

    u_x[:150, 150:] = u_x0
    u_x[150:, 150:] = u_x1
    u_x[:150, :150] = u_x2
    u_x[150:, :150] = u_x3
    u_y[:150, 150:] = u_y0
    u_y[150:, 150:] = u_y1
    u_y[:150, :150] = u_y2
    u_y[150:, :150] = u_y3


    u_x = np.moveaxis(u_x, 0, 1)
    u_y = np.moveaxis(u_y, 0, 1)
    x = np.arange(300)
    y = np.arange(300)
    X, Y = np.meshgrid(x, y)


    plt.streamplot(X, Y, u_x, u_y)

    plt.legend()
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.savefig('figures/sliding_lid_parallel.png')
    plt.close()