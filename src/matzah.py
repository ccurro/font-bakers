import numpy as np
import torch
from torch import nn, optim


def adain(x, w):
    """
    Adaptive instance normalization.

    Parameters
    ----------
    x : [batch_size, num_channels, *]
        Input. 0th dimension must be batch dimension, and 1st dimension must be
        channel dimension. No restrictions on other dimensions.
    w : [batch_size, style_dim]
        Output of style network.

    Returns
    -------
    adain : [batch_size, num_channels, *]
        Same shape as x.

    Notes
    -----
    `style_dim` must be exactly twice `num_channels`.
    """
    batch_size = x.shape[0]
    num_channels = x.shape[1]
    style_dim = w.shape[1]
    if 2 * num_channels != style_dim:
        raise ValueError('style_dim must be exactly twice num_channels.')

    w_mean = w[:, :num_channels]
    w_std = w[:, num_channels:]

    dims = list(range(2, len(x.shape)))  # All dims except the first two
    mean = x.mean(dim=dims)
    std = x.reshape(batch_size, num_channels, -1).std(dim=2)
    x = x.permute(dims + [0, 1])  # Permute dims to use broadcasting.
    adain = w_std * ((x - mean) / std) + w_mean
    adain = adain.permute(dims[-2:] + [0, 1] + dims[:-2])

    return adain


class StyleNetwork(nn.Module):
    def __init__(self, DEVICE, fcSize, numFC, styleDim):
        super(StyleNetwork, self).__init__()
        self.fc1 = nn.Linear(styleDim, fcSize).to(DEVICE)
        self.fcLayers = [
            nn.Linear(fcSize, fcSize).to(DEVICE) for i in range(1, numFC)
        ]
        self.relu = nn.ReLU()

    def forward(self, classVec, styleVec):
        styleVec = self.relu(self.fc1(styleVec))
        for layer in self.fcLayers:
            layer(styleVec)
            styleVec = self.relu(layer(styleVec))
        # concate the classVec and the StyleVec
        style = torch.cat((styleVec, classVec), dim=1)
        return style


class SynthesisNetwork(nn.Module):
    def __init__(self, DEVICE, outputDim, styleDim, numBlocks,
                 channels, kernel):
        super(SynthesisNetwork, self).__init__()
        self.numBlocks = numBlocks
        self.outputDim = outputDim
        self.flatOutputDim = np.prod(outputDim)
        self.DEVICE = DEVICE
        self.channels = channels
        self.noiseScales = [
            torch.randn(1, requires_grad=True).to(DEVICE)
            for _ in range(numBlocks)
        ]
        self.affineTransforms = [
            nn.Linear(styleDim, 2 * channels).to(DEVICE)
            for _ in range(numBlocks)
        ]
        self.adain = adain
        # go to figure out the padding here....
        self.convolutions = [
            nn.Conv3d(channels, channels, kernel, padding=[4, 1, 0]).to(DEVICE)
            for _ in range(numBlocks)
        ]
        self.conv1 = nn.Conv3d(
            outputDim[0], channels, kernel, padding=[4, 1, 0]).to(DEVICE)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(
            channels, outputDim[0], kernel, padding=[4, 1, 0]).to(DEVICE)

    def synthesisBlock(self, noiseDim, generatedCurves, latentStyle, index):
        noiseDim[1] = self.channels
        generatedCurves = generatedCurves \
            + torch.randn(noiseDim).to(self.DEVICE)*self.noiseScales[index]
        affineStyle = self.affineTransforms[index](latentStyle)
        self.adain(generatedCurves, affineStyle)
        generatedCurves = self.relu(self.convolutions[index](generatedCurves))
        return generatedCurves

    def forward(self, batchSize, latentStyle):
        outDim = [batchSize]  + list(self.outputDim)
        generatedCurves = torch.zeros(outDim).to(self.DEVICE)
        # got to do first convolution to get it up to the right channel size
        generatedCurves = self.conv1(generatedCurves)
        for i in range(self.numBlocks):
            generatedCurves = self.synthesisBlock(outDim,generatedCurves, latentStyle,
                                                  i)
        generatedCurves = self.conv2(generatedCurves)
        return generatedCurves


class Matzah(nn.Module):
    def __init__(self, DEVICE, fcSize, numFC, styleDim, outputDim, numBlocks,
                  channels, kernel, numClasses):
        super(Matzah, self).__init__()
        self.Style = StyleNetwork(DEVICE, fcSize, numFC, styleDim)
        self.Synthesis = SynthesisNetwork(DEVICE, outputDim,
                                          fcSize + numClasses, numBlocks,
                                           channels, kernel)

    def forward(self, styleVec, classVec):
        latentStyle = self.Style.forward(styleVec, classVec)
        # now need to set up the other side of the network.
        generatedCurves = self.Synthesis.forward(styleVec.shape[0], latentStyle)
        return generatedCurves


def matzah_optimizer(net):
    '''Returns optimizer and number of epochs, in that order.'''
    return optim.Adam(net.parameters(), lr=0.002)
