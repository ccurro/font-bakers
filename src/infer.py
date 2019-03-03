from absl import flags, app
import torch
from utils import CHARACTERS, save_images, load_model
from rasterizer.rasterizer import Rasterizer

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint", None, "Path to checkpoint file.")
flags.DEFINE_integer("numfonts", 8,
                     "Number of fonts to generate at inference time.")
flags.DEFINE_integer('styledim', 100,
                     "Dimensionality of the latent style space.")
flags.DEFINE_integer('resolution', 64,
                     "Resolution of rasters at inference time.")
flags.DEFINE_string('device', 'cuda:0', "Device to train and infer on.")


@torch.no_grad()
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
    Saves fonts to specified path.
    """
    raster_list = []
    rasterizer = Rasterizer().to(FLAGS.device)

    for _ in range(num_fonts):
        # Create random dense style vector
        style_vector = torch.rand([len(CHARACTERS), style_dim],
                                  device=FLAGS.device)

        # Create random one-hot character vector
        char_vector = torch.eye(len(CHARACTERS), device=FLAGS.device)

        # Generate a batch of fake characters
        fake_chars = gen(char_vector, style_vector)
        fake_chars = torch.reshape(fake_chars,
                                   [len(CHARACTERS), -1, 3, 2]).detach()

        # Rasterize glyph by glyph to avoid OOM. This _kills_ performance.
        rasters = []
        for fake_char in fake_chars:
            fake_char = fake_char.unsqueeze(0)
            raster = rasterizer(fake_char).detach()
            rasters.append(raster)
        rasters = torch.cat(rasters)
        raster_list.append(rasters)

    raster_list = torch.cat(raster_list)
    save_images(raster_list, [num_fonts, len(CHARACTERS)],
                path + '_raster.png')
    torch.save(fake_chars, path + '_bezier.pt')


def main(argv):
    if FLAGS.checkpoint is None:
        raise ValueError('No checkpoint file supplied.')

    gen = load_model(FLAGS.checkpoint)
    infer(gen, FLAGS.numfonts, FLAGS.path, FLAGS.styledim, FLAGS.resolution,
          FLAGS.device)


if __name__ == '__main__':
    app.run(main)
