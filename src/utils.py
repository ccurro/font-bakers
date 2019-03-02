from datetime import datetime
import numpy as np
import torch
from imageio import imwrite

CHARACTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    't', 'u', 'v', 'w', 'x', 'y', 'z', 'zero', 'one', 'two', 'three', 'four',
    'five', 'six', 'seven', 'eight', 'nine', 'exclam', 'numbersign', 'dollar',
    'percent', 'ampersand', 'asterisk', 'question', 'at'
]


def rasterize(x, resolution=64, sigma=0.01, device='cuda'):
    '''
    Simple rasterization: drop a single Gaussian at every control point.

    Parameters
    ----------
    x : [batch_size, num_control_points, 2]
        Control points of glyphs.
    resolution : int
        Resolution of raster.
    sigma : float
        Standard deviation of Gaussians.
    device : one of 'cuda' or 'cpu'
        PyTorch device to raster on.

    Notes
    -----
    The num_contours and num_beziers dimensions have been collapsed into one
    num_control_points dimension.

    Also, we can pad with sufficiently large coordinates (e.g. 999) to
    indicate that there are no more control points: this places a Gaussian
    off-raster, which minimally affects the raster.
    '''

    # Padding constants chosen by looking at empirical distribution of
    # coordinates of Bezier control point from real fonts
    left_pad = 0.25 * resolution
    right_pad = 1.25 * resolution
    up_pad = 0.8 * resolution
    down_pad = 0.4 * resolution
    mesh_lr = np.linspace(
        -left_pad, resolution + right_pad, num=resolution, endpoint=False)
    mesh_ud = np.linspace(
        -down_pad, resolution + up_pad, num=resolution, endpoint=False)
    XX, YY = np.meshgrid(mesh_lr, mesh_ud)
    YY = np.flip(YY)
    XX_expanded = XX[:, :, np.newaxis]
    YY_expanded = YY[:, :, np.newaxis]
    x_meshgrid = torch.Tensor(XX_expanded / resolution).to(device)
    y_meshgrid = torch.Tensor(YY_expanded / resolution).to(device)

    batch_size = x.size()[0]
    num_samples = x.size()[1]
    x_samples = x[:, :, 0].unsqueeze(1).unsqueeze(1)
    y_samples = x[:, :, 1].unsqueeze(1).unsqueeze(1)

    x_meshgrid_expanded = x_meshgrid.expand(batch_size, resolution, resolution,
                                            num_samples)
    y_meshgrid_expanded = y_meshgrid.expand(batch_size, resolution, resolution,
                                            num_samples)

    raster = torch.exp((-(x_samples - x_meshgrid_expanded)**2 -
                        (y_samples - y_meshgrid_expanded)**2) / (2 * sigma**2))
    raster = raster.sum(dim=3)
    return raster


def save_images(images, size, path):
    """
    Saves images.

    Parameters
    ----------
    images : [batch_size, height, width]
        Images to save.
    size : 2-tuple
        Dimensions of grid layout (i.e. [nrows, ncols] in resulting image).
    path : string
        Path to save images to.

    Returns
    -------
    imwrite
    """
    h, w = images.shape[1], images.shape[2]
    images = images.cpu().detach().numpy()
    img = np.zeros((1, h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = int(idx / size[1])
        img[0, j * h:j * h + h, i * w:i * w + w] = image

    return imwrite(path, img[0])


def save_model(checkpoint_num, epoch, gen, disc, optimizer_gen, optimizer_disc,
               pastry_gen, pastry_disc, path):
    '''
    Saves model checkpoint.

    Parameters
    ----------
    checkpoint_num : int
        Checkpoint number.
    epoch : int
        Epoch number.
    gen : torch.nn.Module
        Generator object.
    disc : torch.nn.Module
        Discriminator object.
    optimizer_gen : torch.optim.Optimizer
        Generator optimizer object.
    optimizer_disc : torch.optim.Optimizer
        Discriminator optimizer object.
    pastry_gen : string
        Pastry name of generator.
    pastry_disc : string
        Pastry name of discriminator.
    path : string
        Path to save checkpoint.

    Returns
    -------
    Saves a checkpoint file under the `checkpoints/` with filename
    state_NUM_GEN_DISC_TIME.tar
    '''
    now = datetime.now().strftime("%m.%d.%H.%M")
    name_end = '_{}_{}_{}_{}'.format(checkpoint_num, pastry_gen, pastry_disc,
                                     now)
    state = {
        'epoch': epoch,
        'gen_state_dict': gen.state_dict(),
        'disc_state_dict': disc.state_dict(),
        'optimizer_gen_state_dict': optimizer_gen.state_dict(),
        'optimizer_disc_state_dict': optimizer_disc.state_dict(),
        'gen_pastry_name': pastry_gen,
        'disc_pastry_name': pastry_disc,
    }
    torch.save(state, path + 'state' + name_end + '.pt')
    return name_end


def load_model(path):
    '''
    Loads model checkpoint.

    Parameters
    ----------
    path : string
        Path to the checkpoint file.

    Returns
    -------
    pytorch model
    '''
    raise NotImplementedError('load_model not implemented yet.')
    # checkpoint = torch.load(path)
    # epoch = checkpoint['epoch']
    # gen = checkpoint['gen']
    # disc = checkpoint['disc']
    # optimizer_gen = checkpoint['optimizer_gen']
    # optimizer_disc = checkpoint['optimizer_disc']
    # return epoch, gen, disc, optimizer_gen, optimizer_disc
