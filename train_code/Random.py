import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class Random:
    def generate_random_z_cond(self, t, w, h):
        x_mesh, y_mesh = torch.meshgrid(
            [torch.arange(0, w), torch.arange(0, h)])
        x_mesh, y_mesh = 1.0 * x_mesh, 1.0 * y_mesh
        data = torch.sin(x_mesh * 0.02 + np.cos(y_mesh * 0.01 * (np.cos(t * 0.0021) + 2)) * np.cos(t * 0.01) * 3 + np.cos(
            x_mesh * 0.011 * (np.sin(t * 0.00221) + 2)) * np.cos(t * 0.00321) * 3 + 0.01 * y_mesh * np.cos(t * 0.0215))
        data = (1 + data/torch.max(data))/2

        return data

    def main(self):
        t = 1000
        w = 200
        h = 200
        mesh = self.generate_random_z_cond(t, w, h)
        plt.matshow(mesh)
        plt.colorbar()
        plt.show()
        y = mesh[100].squeeze()
        x = torch.arange(y.numel())
        plt.plot(x.detach().numpy(), y.detach().numpy())
        plt.show()


if __name__ == "__main__":
    r = Random()
    r.main()
