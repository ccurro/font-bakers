from absl import flags, app
from glob import glob
import numpy as np
import torch
from torch import nn, autograd
from torch.utils import data
from serialization.fontDataset import Dataset
import pantry
from utils import CHARACTERS, save_model, rasterize
from infer import infer

np.set_printoptions(threshold=np.inf)  # Print full confusion matrix.
FLAGS = flags.FLAGS

flags.DEFINE_string("disc", None, "Pastry name of discriminator and optimizer")
flags.mark_flag_as_required("disc")
flags.DEFINE_string("gen", None, "Pastry name of generator and optimizer")
flags.mark_flag_as_required("gen")
flags.DEFINE_integer("batch", 16, "Batch: indicates batch size")
flags.DEFINE_integer("epochs", 2, "Number of epochs to train for")
flags.DEFINE_integer(
    "fcSize", 128, "Size of the fully connected layers in the style network")
flags.DEFINE_integer("numFC", 3, "Number of fc layers in the style network")
flags.DEFINE_integer("styleDim", 100, "the output dimension for the style net")
flags.DEFINE_integer("numBlocks", 3, "number of conv blocks in synthesis net")
flags.DEFINE_integer("channels", 32,
                     "number of channels in each of the conv blocks")
flags.DEFINE_list("outputDim", [20, 30, 2, 2],
                  "output dimension of produced glyph")
flags.DEFINE_list("kernelDim", [9, 3, 1],
                  "Dimension of kernel in synthesis net block")
flags.DEFINE_integer("numClasses", 70, "number of different glyphs")
flags.DEFINE_float("clip", 3.0, "Value to clip norm of gradients at")

DATA_PATH = '/flour/noCapsnoRepeatsSingleExampleProtos/'
DIMENSIONS = (20, 30, 3, 2)
LAMBDA_GP = 10

CLASS_INDEX = {label: x for x, label in enumerate(CHARACTERS)}
FONT_FILES = glob(DATA_PATH + '*')  # List of paths to protobuf files


def compute_gradients(discriminator, real_samples, fake_samples):
    """
    Calculates the gradient penalty loss for WGAN-GP.

    Parameters
    ----------
    discriminator : torch.nn.Module
        Discriminator PyTorch module.

    real_samples : [N, resolution, resolution]
        Raster of real glyph. Zeroth dimension must be the batch dimension.

    fake_samples : [N, resolution, resolution]
        Raster of fake glyph. Zeroth dimension must be the batch dimension.

    Returns
    -------
    gradient_deviation : float
        Gradient deviation from 1. Must take mean of square to obtain gradient
        penalty.
    """

    # Random weight term for interpolation between real and fake samples
    alpha = torch.randint(2, [real_samples.size(0), 1, 1]).type(
        torch.cuda.FloatTensor).to(FLAGS.device)
    # Get random interpolation between real and fake samples
    interpolates = (
        alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    d_interpolates = discriminator(interpolates, raster_input=True)
    fake = autograd.Variable(
        torch.ones([real_samples.shape[0], 70]),
        requires_grad=False).type(torch.cuda.FloatTensor).to(FLAGS.device)

    # Get gradient with respect to the interpolations.
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    return gradients.view(gradients.size(0), -1)


def main(argv):
    print('Starting...')

    trainset = Dataset(FONT_FILES, DIMENSIONS)
    trainloader = data.DataLoader(
        trainset,
        batch_size=FLAGS.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True)

    disc = pantry.disc[FLAGS.disc](FLAGS.device).to(FLAGS.device)
    kernelDim = [int(x) for x in FLAGS.kernelDim]
    outputDim = [int(x) for x in FLAGS.outputDim]
    gen = pantry.gen[FLAGS.
                     gen](  # All these flags are specifically for matzah.
                         FLAGS.device,
                         fcSize=FLAGS.fcSize,
                         numFC=FLAGS.numFC,
                         styleDim=FLAGS.styleDim,
                         outputDim=outputDim,
                         numBlocks=FLAGS.numBlocks,
                         channels=FLAGS.channels,
                         kernel=kernelDim,
                         numClasses=FLAGS.numClasses).to(FLAGS.device)
    optimizer_disc = pantry.optimsDisc[FLAGS.disc](disc)
    optimizer_gen = pantry.optimsGen[FLAGS.gen](gen)
    num_epochs = FLAGS.epochs

    for epoch in range(num_epochs):
        for i, (real_chars, _) in enumerate(trainloader):
            # -------------------
            # Train Discriminator
            # -------------------

            # Zero the gradients for the generator
            optimizer_disc.zero_grad()

            # Create random dense style vector
            style_vector = torch.rand([FLAGS.batch, FLAGS.styledim],
                                      device=FLAGS.device)

            # Create random one-hot character vector
            char_vector = torch.zeros(
                [FLAGS.batch, len(CHARACTERS)], device=FLAGS.device)
            onehot = np.random.randint(
                low=0, high=len(CHARACTERS), size=FLAGS.batch)
            char_vector[range(FLAGS.batch), onehot] = 1

            # Convert real characters and generate a batch of fake characters
            real_chars = real_chars.float().to(FLAGS.device)
            fake_chars = gen(char_vector, style_vector)

            real_validity = disc(real_chars)  # Real characters
            fake_validity = disc(fake_chars)  # Fake characters

            # Gradient penalty. Calculate glyph by glyph to avoid OOM
            all_gradients = []
            for real_char, fake_char in zip(real_chars, fake_chars):
                real_char = real_char.view(1, -1, 2)
                real_char_raster = rasterize(real_char, device=FLAGS.device)
                real_char_raster = real_char_raster.unsqueeze(1)

                points = torch.cat(
                    [curve for contour in fake_char for curve in contour])
                points = points.unsqueeze(0)
                fake_char_raster = rasterize(points, device=FLAGS.device)
                fake_char_raster = fake_char_raster.unsqueeze(1)

                gradients = compute_gradients(disc, real_char_raster, fake_char_raster)
                all_gradients.append(gradients)

            gradient_penalty = ((torch.cat(all_gradients, dim=0).norm(dim=1) - 1)**2).mean()

            # Discriminator loss
            loss_disc = -torch.mean(real_validity) + torch.mean(
                fake_validity) + LAMBDA_GP * gradient_penalty

            # Update discriminator with clipped gradients
            loss_disc.backward()
            nn.utils.clip_grad_norm_(
                disc.parameters(), FLAGS.clip, norm_type=2)
            optimizer_disc.step()

            # Print statistics and delete loss
            print('[%d, %5d] loss Discriminator: %.3f' % (epoch + 1, i + 1,
                                                          loss_disc.item()))
            del loss_disc

            # ---------------
            # Train Generator
            # ---------------

            # Zero the gradients for the generator
            optimizer_gen.zero_grad()

            # Generate a batch of fake characters
            fake_chars = gen(char_vector, style_vector)

            # Generator loss
            fake_validity = disc(fake_chars)
            loss_gen = -torch.mean(fake_validity)

            # Update generator with clipped gradients
            loss_gen.backward()
            nn.utils.clip_grad_norm_(gen.parameters(), FLAGS.clip, norm_type=2)
            optimizer_gen.step()

            # Print statistics and delete loss
            print('[%d, %5d] loss Generator: %.3f' % (epoch + 1, i + 1,
                                                      loss_gen.item()))
            del loss_gen

            # --------------------
            # Infer From Generator
            # --------------------
            if i % 2000 == 1999:  # Infer every 2000 backprops.
                name_end = save_model(  # save_model returns filename
                    epoch,
                    epoch,
                    gen,
                    disc,
                    optimizer_gen,
                    optimizer_disc,
                    FLAGS.gen,
                    FLAGS.disc,
                    path='../output/checkpoints/')
                gen.eval()
                infer(
                    gen,
                    num_fonts=FLAGS.numfonts,
                    path='../output/fonts/font{}'.format(name_end),
                    style_dim=FLAGS.styledim,
                    resolution=FLAGS.resolution,
                    device=FLAGS.device)
                gen.train()

    print('Finished.')


if __name__ == '__main__':
    app.run(main)
