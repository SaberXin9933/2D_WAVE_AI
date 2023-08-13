import torch
import torch.nn.functional as F
import numpy as np


class Derivatives:
    def __init__(
        self,
        kernel_point_number: int,
        kernel_order: int,
        kernel_delta: int,
        kernel_dtype: torch.dtype,
        device: torch.device = None,
    ):
        self.kernel_point_number = kernel_point_number
        self.kernel_order = kernel_order
        self.kernel_delta = kernel_delta
        self.dtype = kernel_dtype
        self.device = device if device != None else torch.device("cpu")
        self.initConvdKernel()

    def initConvdKernel(self):
        pointNumber = self.kernel_point_number
        order = self.kernel_order
        delta = self.kernel_delta
        self.padding = pointNumber // 2
        self.dy_kernel = (
            self.get_larg_kernel(pointNumber, order, delta)
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(2)
            .to(self.dtype)
            .to(self.device)
        )

        self.dx_kernel = (
            self.get_larg_kernel(pointNumber, order, delta)
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(3)
            .to(self.dtype)
            .to(self.device)
        )

        self.move_top_kernel = (
            torch.Tensor([0, 0, 1])
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(3)
            .to(self.dtype)
            .to(self.device)
        )

        self.move_bottom_kernel = (
            torch.Tensor([1, 0, 0])
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(3)
            .to(self.dtype)
            .to(self.device)
        )

        self.move_left_kernel = (
            torch.Tensor([0, 0, 1])
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(2)
            .to(self.dtype)
            .to(self.device)
        )

        self.move_right_kernel = (
            (torch.Tensor([1, 0, 0]).unsqueeze(0).unsqueeze(1).unsqueeze(2))
            .to(self.dtype)
            .to(self.device)
        )

        self.mean_xy_kernel = (
            (torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).unsqueeze(0).unsqueeze(1))
            .to(self.dtype)
            .to(self.device)
        )

        self.mean_top_kernel = 0.5 * torch.Tensor([1, 1, 0]).unsqueeze(0).unsqueeze(
            1
        ).unsqueeze(3).to(self.dtype).to(self.device)

        self.mean_bottom_kernel = 0.5 * torch.Tensor([0, 1, 1]).unsqueeze(0).unsqueeze(
            1
        ).unsqueeze(3).to(self.dtype).to(self.device)

        self.mean_right_kernel = 0.5 * torch.Tensor([0, 1, 1]).unsqueeze(0).unsqueeze(
            1
        ).unsqueeze(2).to(self.dtype).to(self.device)

        self.mean_left_kernel = 0.5 * torch.Tensor([1, 1, 0]).unsqueeze(0).unsqueeze(
            1
        ).unsqueeze(2).to(self.dtype).to(self.device)

    def get_larg_kernel(self, point_num: int, order: int = 1, delta: int = 1):
        x = [-(point_num - 1) * delta / 2 + i * delta for i in range(point_num)]
        # x = [-1.5,-0.5,0.5,1.5]
        n = len(x)
        _len = len(x)
        A = np.zeros((_len, n))
        X = np.array(x)
        X = np.expand_dims(X, axis=1)
        for i, delta in enumerate(x):
            for j in range(n):
                A[i, j] = (delta**j) / np.math.factorial(j)
        A_inv = np.linalg.inv(A)
        kernel = np.zeros(point_num + 1)
        kernel[1:] = A_inv[order]
        return torch.Tensor(kernel)

    def dx(self, v):
        return F.conv2d(v, self.dx_kernel, padding=(self.padding, 0))

    def dy(self, v):
        return F.conv2d(v, self.dy_kernel, padding=(0, self.padding))

    def move_top(self, v):
        return F.conv2d(v, self.move_top_kernel, padding=(1, 0))

    def move_bottom(self, v):
        return F.conv2d(v, self.move_bottom_kernel, padding=(1, 0))

    def move_left(self, v):
        return F.conv2d(v, self.move_left_kernel, padding=(0, 1))

    def move_right(self, v):
        return F.conv2d(v, self.move_right_kernel, padding=(0, 1))

    def mean_xy(self, v):
        return F.conv2d(v, self.mean_xy_kernel, padding=(1, 1)) / 9

    def mean_top(self, v):
        return F.conv2d(v, self.mean_top_kernel, padding=(1, 0))

    def mean_bottom(self, v):
        return F.conv2d(v, self.mean_bottom_kernel, padding=(1, 0))

    def mean_left(self, v):
        return F.conv2d(v, self.mean_left_kernel, padding=(0, 1))

    def mean_right(self, v):
        return F.conv2d(v, self.mean_right_kernel, padding=(0, 1))


if __name__ == "__main__":
    kernel_point_number: int = 2
    kernel_order: int = 1
    kernel_delta: int = 1
    dtype: torch.device = torch.float32
    device: torch.device = torch.device("cpu")
    d = Derivatives(kernel_point_number, kernel_order, kernel_delta, dtype, device)
    print(d.dx_kernel)
    v = 1.0 * torch.arange(16).reshape(4, 4).unsqueeze(0).unsqueeze(1)
    print(v)
    y = d.dx(v)
    print(y)
