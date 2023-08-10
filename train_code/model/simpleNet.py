import torch
from torch import nn


class Spinn_Wave_Model(nn.Module):
    def __init__(self, hidden_size=64, bilinear=True, residuals=False):
        super(Spinn_Wave_Model, self).__init__()
        self.hidden_size = hidden_size
        self.bilinear = bilinear
        self.residuals = residuals
        self.conv1 = nn.Conv2d(4, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_size, 3, kernel_size=3, padding=1)

    def forward(self, p_old, v_old, field_masks):
        x = torch.cat([p_old, v_old, field_masks], dim=1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        p_new, v_new = 4 * torch.tanh((1 / 4) * (p_old + x[:, 0:1])), 4 * torch.tanh(
            (1 / 4) * (v_old + x[:, 1:3])
        )
        return p_new, v_new


class Conv2d_4layer(nn.Module):
    def __init__(self, hidden_size=64, bilinear=True, residuals=False):
        super(Conv2d_4layer, self).__init__()
        self.hidden_size = hidden_size
        self.bilinear = bilinear
        self.residuals = residuals
        self.conv1 = nn.Conv2d(4, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden_size, 3, kernel_size=3, padding=1)

    def forward(self, p_old, v_old, field_masks):
        x = torch.cat([p_old, v_old, field_masks], dim=1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.conv4(x)
        p_new, v_new = 4 * torch.tanh((1 / 4) * (p_old + x[:, 0:1])), 4 * torch.tanh(
            (1 / 4) * (v_old + x[:, 1:3])
        )
        return p_new, v_new


class Conv2d_6layer(nn.Module):
    def __init__(self, hidden_size=64, bilinear=True, residuals=False):
        super(Conv2d_6layer, self).__init__()
        self.hidden_size = hidden_size
        self.bilinear = bilinear
        self.residuals = residuals
        self.conv1 = nn.Conv2d(4, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(hidden_size, 3, kernel_size=3, padding=1)

    def forward(self, p_old, v_old, field_masks):
        x = torch.cat([p_old, v_old, field_masks], dim=1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.conv5(x)
        p_new, v_new = 4 * torch.tanh((1 / 4) * (p_old + x[:, 0:1])), 4 * torch.tanh(
            (1 / 4) * (v_old + x[:, 1:3])
        )
        return p_new, v_new
