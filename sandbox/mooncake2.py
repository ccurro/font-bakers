#!/bin/python

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512
NUM_BLOCKS = 16
NUM_TRAIN_ITERATIONS = 10000
SEED_LENGTH = 20


class CausalConv1d(nn.Conv1d):
    # From Alex Rogozhnikov:
    # https://github.com/pytorch/pytorch/issues/1333#issuecomment-400338207
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
    ):
        self.__padding = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


def generate_sinewaves(batch_size,
                       num_periods=2,
                       variance=0.1,
                       max_num_samples=200):
    """
    Returns samples of one period of a sinewave with random frequency, phase
    and number of samples.

    Parameters
    ----------
    batch_size : int
        Batch size.
    num_periods : int
        Number of periods of the sinewave to generate before stopping.
    variance : float
        Variance of the AWGN to add to the sinewave.
    max_num_samples : int
        The maximum length of the stopped sinewave.

    Returns
    -------
    sinewave : [batch_size, 1, max_num_samples]
    """
    sinewaves = []
    for i in range(batch_size):
        n = np.random.randint(50, 150)
        f = np.random.uniform(0.8, 1.2)
        phi = np.random.uniform(0.1, 0.5)
        x = np.linspace(0, num_periods / f, n, dtype=np.float32)
        sinewave = np.sin(2 * np.pi * f * (x + phi))
        pad_length = max_num_samples - len(sinewave)
        sinewave = np.pad(
            sinewave, (0, pad_length), "constant", constant_values=0)
        sinewave += variance * np.random.randn(max_num_samples)
        sinewaves.append(sinewave)
    return torch.tensor(np.stack(sinewaves, axis=0)).unsqueeze(1).to(DEVICE)


def negative_log_prob(x, pi, mu, sigma):
    """
    Negative log probability of predictive distribution of x_{k+1} from x_1,
    ..., x_k.

    Parameters
    ----------
    x : [batch_size, num_channels, max_num_samples]
        Sinewave.
    pi, mu, sigma : [batch_size, num_components (+1 for pi), max_num_samples]
        Mixture weights, means and standard deviations of each Gaussian
        component, respectively.

    Returns
    -------
    [batch_size, max_num_samples - 1]
    """
    # Index appropriately to compute nlogp of predicted next value.
    vals = x[..., 1:]
    pi = pi[:, 1:, :-1]
    mu = mu[..., :-1]
    sigma = sigma[..., :-1]
    negative_densities = (0.5 * np.log(2 * np.pi) + torch.log(sigma) -
                          torch.log(pi) + 0.5 * (vals - mu)**2 / sigma**2)
    return negative_densities.sum(dim=1)


class ResidualBlock(nn.Module):
    """
        |-------------------------------------|
        |                                     |
        |             |-- tanh --|            |
    ----|-> dilated_conv         * --- 1x1 -- + -->
                      |-- sigm --|      |
                                        |
                                        |
    ----------------------------------> + -------->
    """

    def __init__(self, num_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.num_channels = num_channels
        self.conv1 = CausalConv1d(
            num_channels,
            2 * num_channels,
            kernel_size=kernel_size,
            dilation=dilation)
        self.conv2 = nn.Conv1d(
            num_channels, num_channels, kernel_size=1, dilation=1)

    def forward(self, x):
        a = self.conv1(x)
        b = torch.tanh(a[:, :self.num_channels, :])
        c = torch.sigmoid(a[:, self.num_channels:, :])

        return self.conv2(b * c) + x


class Mooncake(nn.Module):
    def __init__(
            self,
            in_channels=1,
            max_num_samples=200,
            num_channels=4,
            num_blocks=8,
            kernel_size=2,
            dilations=[1, 2, 4, 8, 16, 32, 64, 128],
            num_components=1,
    ):
        super(Mooncake, self).__init__()

        if len(dilations) != num_blocks:
            msg = ("Number of dilations must be equal to number of residual "
                   "blocks.")
            raise ValueError(msg)

        self.max_num_samples = max_num_samples
        self.num_channels = num_channels
        self.num_blocks = num_blocks

        # Coordconv adds 1 to `in_channels`
        self.conv1 = CausalConv1d(
            in_channels + 1, num_channels, kernel_size=kernel_size, dilation=1)
        self.conv2 = nn.Conv1d(num_channels, 2 * num_channels, kernel_size=1)

        self.blocks = nn.ModuleList([
            ResidualBlock(
                num_channels, kernel_size=kernel_size,
                dilation=dilations[i]).to(DEVICE)
            for i in range(self.num_blocks)
        ])

        self.conv_pi = nn.Conv1d(
            2 * num_channels, num_components + 1, kernel_size=1)
        self.conv_mu = nn.Conv1d(
            2 * num_channels, num_components, kernel_size=1)
        self.conv_sigma = nn.Conv1d(
            2 * num_channels, num_components, kernel_size=1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : [batch_size, num_channels, num_samples]
            Input.

        Returns
        -------
        pi, mu, sigma : [batch_size, num_components (+1 for pi), num_samples]
            Mixture weights, means and standard deviations of each Gaussian
            component, respectively.
        """
        batch_size = x.shape[0]
        num_samples = x.shape[2]
        linspace = torch.tensor(
            np.tile(
                np.linspace(0, num_samples / self.max_num_samples,
                            num_samples),
                [batch_size, 1])).unsqueeze(1).float().to(DEVICE)
        x = torch.cat([x, linspace], dim=1)

        taps = [self.conv1(x)]

        for i in range(self.num_blocks):
            tap = self.blocks[i](taps[i])
            taps.append(tap)

        aggregated_blocks = F.relu(torch.stack(taps).mean(dim=0))
        z = self.conv2(aggregated_blocks)

        pi = F.softmax(self.conv_pi(z), dim=1)
        mu = 2 * torch.tanh(self.conv_mu(z) / 2)
        sigma = F.softplus(self.conv_sigma(z))

        return pi, mu, sigma


if __name__ == "__main__":
    np.random.seed(1618)
    torch.manual_seed(1618)

    dilations = [2**i for i in range(NUM_BLOCKS)]
    mooncake = Mooncake(num_blocks=NUM_BLOCKS, dilations=dilations).to(DEVICE)
    optimizer = optim.Adam(mooncake.parameters(), lr=0.002)
    num_trainable_params = sum(
        p.numel() for p in mooncake.parameters() if p.requires_grad)
    print("# trainable parameters: {}".format(num_trainable_params))

    # Train
    mooncake.train()

    for i in range(NUM_TRAIN_ITERATIONS):
        # Generate one sinewave and make a batch out of it.
        sinewaves = generate_sinewaves(BATCH_SIZE)
        pi, mu, sigma = mooncake(sinewaves)

        # Update
        optimizer.zero_grad()
        nlogp = negative_log_prob(sinewaves, pi, mu, sigma).mean()
        nlogp.backward()
        optimizer.step()

        if i % 10 == 0:
            print("[{}] nlogp: {}".format(i,
                                          nlogp.cpu().detach().numpy().item()))
        del nlogp

        if i % 500 == 499:
            # Infer
            mooncake.eval()
            fig, ax = plt.subplots()

            for c in ['r', 'b', 'g']:
                with torch.no_grad():
                    ground_truth = generate_sinewaves(1)
                    inferred = torch.zeros_like(ground_truth)
                    inferred[..., :SEED_LENGTH] = ground_truth[..., :
                                                               SEED_LENGTH]

                    for j in range(SEED_LENGTH, inferred.shape[-1]):
                        pi, mu, sigma = mooncake(inferred[..., :j])

                        pi = pi[:, 1:, -1]
                        mu = mu[..., -1]
                        sigma = sigma[..., -1]
                        inferred[..., j] = (pi * mu).sum(dim=1)

                ax.plot(
                    ground_truth.cpu().detach().numpy().squeeze(),
                    color=c,
                    linestyle='dashed',
                    alpha=0.3)
                ax.plot(
                    inferred.cpu().detach().numpy().squeeze(),
                    color=c,
                    linestyle='solid')

            plt.savefig("inference_{}.png".format(i))
