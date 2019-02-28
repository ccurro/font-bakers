from absl import flags, app
from glob import glob
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch import autograd
from serialization.fontDataset import Dataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pantry
from fridge import model_saver as saver

np.set_printoptions(threshold=np.inf)  # Print full confusion matrix.
FLAGS = flags.FLAGS

flags.DEFINE_string("disc", None, "Pastry name of discriminator and optimizer")
flags.mark_flag_as_required("disc")
flags.DEFINE_string("gen", None, "Pastry name of generator and optimizer")
flags.mark_flag_as_required("gen")
flags.DEFINE_integer("styledim", 100, "Dimension of the latent style space")
flags.DEFINE_integer("batch", 16, "Batch: indicates batch size")
flags.DEFINE_integer("epochs", 2, "Number of epochs to train for")

DATA_PATH = '/flour/noCapsnoRepeatsSingleExampleProtos/'
DIMENSIONS = (20, 30, 3, 2)
TRAIN_TEST_SPLIT = 0.8
LAMBDA_GP = 10

CHARACTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    't', 'u', 'v', 'w', 'x', 'y', 'z', 'zero', 'one', 'two', 'three', 'four',
    'five', 'six', 'seven', 'eight', 'nine', 'exclam', 'numbersign', 'dollar',
    'percent', 'ampersand', 'asterisk', 'question', 'at'
]
CLASS_INDEX = {label: x for x, label in enumerate(CHARACTERS)}

FONT_FILES = glob(DATA_PATH + '*')  # List of paths to protobuf files
SPLIT = int(TRAIN_TEST_SPLIT * len(FONT_FILES))
FONT_FILES_TRAIN = FONT_FILES[:SPLIT]
FONT_FILES_VAL = FONT_FILES[SPLIT:]

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


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
    alpha = torch.randint(2, [real_samples.size(0), 1, 1, 1, 1])

    # Get random interpolation between real and fake samples
    interpolations = (
        alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    d_interpolates = discriminator(interpolates)
    fake = autograd.Variable(
        torch.ones([real_samples.shape[0], 1]), requires_grad=False)

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


def validate(net, epoch_num):
    print('\tValidating...')

    valset = Dataset(FONT_FILES_VAL, DIMENSIONS)
    valloader = data.DataLoader(
        valset,
        batch_size=FLAGS.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True)

    correct = 0
    total = 0
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for font, labels in valloader:
            font = font.float().to(DEVICE)
            labels = [CLASS_INDEX[label] for label in labels]
            labels = torch.tensor(labels).to(DEVICE)

            outputs = net(font)
            _, predicted = torch.max(outputs.data, dim=1)
            total += len(labels)
            correct += (predicted == labels).sum().item()
            pred_labels.append(predicted.cpu())
            true_labels.append(labels.cpu())

    pred_labels = torch.cat(pred_labels)
    true_labels = torch.cat(true_labels)

    confusion = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=[18, 12])
    plt.imshow(confusion)
    plt.colorbar()
    tick_marks = np.arange(len(CHARACTERS))
    plt.xticks(tick_marks, CHARACTERS, rotation=45)
    plt.yticks(tick_marks, CHARACTERS)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_{}.png'.format(epoch_num))

    print('\tAccuracy on validation fonts: %d %%' % (100 * correct / total))
    print(confusion)
    print('\n\n\n')


def main(argv):
    print('Starting...')

    trainset = Dataset(FONT_FILES_TRAIN, DIMENSIONS)
    trainloader = data.DataLoader(
        trainset,
        batch_size=FLAGS.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True)

    disc = pantry.disc[FLAGS.disc](DEVICE).to(DEVICE)
    gen = pantry.gen[FLAGS.gen](DEVICE, (16, 20, 30, 3, 2)).to(DEVICE)
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
                                      device=DEVICE)

            # Create random one-hot character vector
            char_vector = torch.zeros(
                [FLAGS.batch, len(CHARACTERS)], device=DEVICE)
            onehot = np.random.randint(
                low=0, high=len(CHARACTERS), size=FLAGS.batch)
            char_vector[range(FLAGS.batch), onehot] = 1

            # Convert real characters and generate a batch of fake characters
            real_chars = real_chars.float().to(DEVICE)
            fake_chars = gen(char_vector, style_vector)

            real_validity = disc(real_chars)  # Real characters
            fake_validity = disc(fake_chars)  # Fake characters

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(
                disc, real_chars, fake_chars)

            # Discriminator loss
            loss_disc = -torch.mean(real_validity) + torch.mean(
                fake_validity) + LAMBDA_GP * gradient_penalty

            loss_disc.backward()
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

            loss_gen.backward()
            optimizer_gen.step()

            # Print statistics and delete loss
            print('[%d, %5d] loss Generator: %.3f' % (epoch + 1, i + 1,
                                                      loss_gen.item()))
            del loss_gen

        # Validate at the end of every epoch.
        validate(net, epoch)

    print('Finished.')


if __name__ == '__main__':
    app.run(main)
