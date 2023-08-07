import matplotlib.pyplot as plt
import numpy as np


class Random:
    def generate_random_z_cond(self, t, w, h):
        x_mesh, y_mesh = np.meshgrid(np.arange(0, w), np.arange(0, h))
        x_mesh, y_mesh = 1.0 * x_mesh, 1.0 * y_mesh
        data = np.sin(x_mesh * 0.02 + np.cos(y_mesh * 0.01 * (np.cos(t * 0.0021) + 2)) * np.cos(t * 0.01) * 3 + np.cos(
            x_mesh * 0.011 * (np.sin(t * 0.00221) + 2)) * np.cos(t * 0.00321) * 3 + 0.01 * y_mesh * np.cos(t * 0.0215))
        data = (5 + data/np.max(data))/5
        data /= np.mean(data)

        return data
    
    def rand():
        return np.random.rand()

    def main(self):
        t = 10000
        w = 200
        h = 200
        mesh = self.generate_random_z_cond(t, w, h)
        print(np.mean(mesh))
        plt.matshow(mesh)
        plt.colorbar()
        plt.show()
        y = mesh[100].squeeze()
        x = np.arange(y.size)
        plt.plot(x,y)
        plt.show()

fRandom = Random()

if __name__ == "__main__":
    r = Random()
    r.main()
