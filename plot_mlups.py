import matplotlib.pyplot as plt


def calculate_mlups(t) :
        return (10**5*300*300)/(10**6*t)


if __name__ == '__main__' :

        time = {1:2487, 4:596.4154486656189, 9:384.1501386165619, 16:268.5490810871124, 25:248.5106806755066, 
                36:220.8293719291687, 49:182.40020990371704, 69:173.5202353000641, 81:162.81913590431213, 100:145.73559498786926}

        X = []
        MLUPS = []

        for key in time :
                X.append(key)
                MLUPS.append(calculate_mlups(time[key]))

        plt.legend()
        plt.plot(X, MLUPS)
        plt.xlabel("Number of processes")
        plt.ylabel("MLUPS")
        plt.savefig('figures/mlups.png')
