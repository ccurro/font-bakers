from absl import flags, app
from glob import glob
import numpy as np
import torch
from torch import nn, autograd
from torch.utils import data
from serialization.fontDataset import Dataset
import pantry
from utils import CHARACTERS, save_model
from infer import infer

np.set_printoptions(threshold=np.inf)  # Print full confusion matrix.
FLAGS = flags.FLAGS

flags.DEFINE_string("disc", None, "Pastry name of discriminator and optimizer")
flags.mark_flag_as_required("disc")
flags.DEFINE_string("gen", None, "Pastry name of generator and optimizer")
flags.mark_flag_as_required("gen")
flags.DEFINE_integer("batch", 16, "Batch: indicates batch size")
flags.DEFINE_integer("epochs", 2, "Number of epochs to train for")
flags.DEFINE_float("clip", 3.0, "Value to clip norm of gradients at")

DATA_PATH = '/flour/noCapsnoRepeatsSingleExampleProtos/'
DIMENSIONS = (20, 30, 3, 2)
LAMBDA_GP = 10

CLASS_INDEX = {label: x for x, label in enumerate(CHARACTERS)}
FONT_FILES = glob(DATA_PATH + '*')  # List of paths to protobuf files


def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """
    Calculates the gradient penalty loss for WGAN-GP.

    Parameters
    ----------
    discriminator : torch.nn.Module
        Discriminator PyTorch module.

    real_samples : [N, _, _, _, _]
        True data. Zeroth dimension must be the batch dimension.

    fake_samples : [N, _, _, _, _]
        Generated samples. Zeroth dimension must be the batch dimension.

    Returns
    -------
    gradient_penalty : float
        Gradient penalty to enforce 1-Lipschitz condition.
    """

    # Random weight term for interpolation between real and fake samples
    alpha = torch.randint(2, [real_samples.size(0), 1, 1, 1, 1]).type(
        torch.cuda.FloatTensor).to(FLAGS.device)
    # Get random interpolation between real and fake samples
    interpolates = (
        alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    d_interpolates = discriminator(interpolates)
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

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    return gradient_penalty


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
    gen = pantry.gen[FLAGS.
                     gen](  # All these flags are specifically for matzah.
                         FLAGS.device,
                         fcSize=128,
                         numFC=3,
                         styleDim=100,
                         outputDim=(16, 20, 30, 3, 2),
                         numBlocks=3,
                         startDim=(16, 30, 30, 3, 2),
                         channels=32,
                         kernel=(9, 3, 1),
                         numClasses=70).to(FLAGS.device)
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

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(
                disc, real_chars, fake_chars)

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
                infer(
                    gen,
                    num_fonts=FLAGS.numfonts,
                    path='../output/fonts/font{}'.format(name_end),
                    style_dim=FLAGS.styledim,
                    resolution=FLAGS.resolution,
                    device=FLAGS.device)

    print('Finished.')


if __name__ == '__main__':
    app.run(main)
