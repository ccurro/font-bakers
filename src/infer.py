from absl import flags, app
import torch
from utils import CHARACTERS, rasterize, save_images, load_model

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint", None, "Path to checkpoint file.")
flags.DEFINE_integer("numfonts", 16,
                     "Number of fonts to generate at inference time.")
flags.DEFINE_integer('styledim', 100,
                     "Dimensionality of the latent style space.")
flags.DEFINE_integer('resolution', 64,
                     "Resolution of rasters at inference time.")
flags.DEFINE_string('device', 'cuda:0', "Device to train and infer on.")


def infer(gen, num_fonts, path, style_dim=100, resolution=64, device='cuda'):
    """
    Runs generator at inference time to generate a batch of fonts.

    Parameters
    ----------
    gen : torch.nn.Module
        PyTorch generator object.
    num_fonts : int
        Number of fonts to generate.
    style_dim : int
        Dimensionality of style space (i.e. output of style network). Defaults
        to 100.
    resolution : int
        Resolution of rasters. Defaults to 64.
    device : one of 'cuda' or 'cpu'
        Device to infer on. Defaults to 'cuda'.

    Returns
    -------
    """
    # Create random dense style vector
    # style_vector = torch.rand([num_fonts * len(CHARACTERS), style_dim], device=FLAGS.device)
    style_vector = torch.rand([16, style_dim], device=FLAGS.device)

    # Create random one-hot character vector
    # char_vector = torch.eye(len(CHARACTERS), device=FLAGS.device).repeat([num_fonts, 1])
    char_vector = torch.zeros([16, len(CHARACTERS)], device=FLAGS.device)
    char_vector[range(16), range(16)] = 1

    # Generate a batch of fake characters
    fake_chars = gen(char_vector, style_vector)
    fake_chars = torch.reshape(fake_chars, [num_fonts, -1, 2])
    rasters = rasterize(fake_chars)

    # save_images(rasters, [num_fonts, len(CHARACTERS)], path + '_raster.png')
    save_images(rasters, [4, 4], path + '_raster.png')
    torch.save(fake_chars, path + '_bezier.pt')


def main(argv):
    if FLAGS.checkpoint is None:
        raise ValueError('No checkpoint file supplied.')

    gen = load_model(FLAGS.checkpoint)
    infer(gen, FLAGS.numfonts, FLAGS.path, FLAGS.styledim, FLAGS.resolution,
          FLAGS.device)


if __name__ == '__main__':
    app.run(main)
