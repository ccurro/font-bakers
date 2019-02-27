from absl import flags, app
from glob import glob
import numpy as np
import torch
from torch import nn
from torch.utils import data
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

    criterion_disc = nn.CrossEntropyLoss()
    criterion_disc = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (font, labels) in enumerate(trainloader):
            font = font.float().to(DEVICE)
            labels = [CLASS_INDEX[label] for label in labels]
            labels = torch.tensor(labels).to(DEVICE)

            style_vec = torch.rand((FLAGS.batch, FLAGS.styledim),
                                   device=DEVICE)
            char_vec = torch.randint(
                0, len(CHARACTERS) - 1, [1, FLAGS.batch],
                dtype=torch.float32).to(DEVICE)

            # add index for fake character
            fake_vec = (len(CHARACTERS) + 1) * torch.ones(FLAGS.batch,
                                                          1).to(DEVICE)

            # Combine char vector from real and fake charcters
            labels_combined = torch.cat(
                [labels.float(), fake_vec.squeeze().float()], 0)

            # Generate characters
            chars_generated = gen(char_vec, style_vec)

            # Zero the parameter gradients for the discriminator
            optimizer_disc.zero_grad()

            # Make predictions
            prediction_real = disc(font)
            prediction_fake = disc(chars_generated)
            prediction_combined = torch.cat((prediction_real, prediction_fake),
                                            dim=0)

            # Train the discriminator
            loss_disc = criterion_disc(prediction_combined, labels_combined)
            loss_disc.backward()
            optimizer_disc.step()

            # Zero the parameter gradients for the generator
            optimizer_gen.zero_grad()

            # Train the generator
            loss_gen = criterion_disc(prediction_fake, char_vec)
            loss_gen.backward()
            optimizer_gen.step()

            # Print statistics
            lossd_ = loss_disc.item()
            print('[%d, %5d] loss Discriminator: %.3f' % (epoch + 1, i + 1,
                                                          lossd_))
            lossg_ = loss_gen.item()
            print(
                '[%d, %5d] loss Generator: %.3f' % (epoch + 1, i + 1, lossg_))

            del lossd_
            del lossg_

        # Validate at the end of every epoch.
        validate(net, epoch)

    print('Finished.')


if __name__ == '__main__':
    app.run(main)
