import numpy as np
import matplotlib.pyplot as plt




if __name__ == '__main__' :
    plt.figure()

    with open('u_4.npy', 'rb') as file :
        u = np.load(file)

    shape = u.shape
    x = np.arange(shape[1])
    y = np.arange(shape[2])
    X, Y = np.meshgrid(x, y)

    u_x = np.moveaxis(u[0], 0, 1)
    u_y = np.moveaxis(u[1], 0, 1)
    print(u_x.shape, u_y.shape, X.shape, Y.shape)
    plt.streamplot(X, Y, u_x, u_y)

    plt.legend()
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.show()
    plt.close()