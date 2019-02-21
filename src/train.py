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

np.set_printoptions(threshold=np.inf)  # Print full confusion matrix.
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "pastry", None,
    "Pastry: indicates network, optimizer and number of epochs")
flags.mark_flag_as_required("pastry")

TRAIN_PATH_1 = '/zooper1/fontbakers/data/noCapsnoRepeatsT1/'
TRAIN_PATH_2 = '/zooper1/fontbakers/data/noCapsnoRepeatsT2/'
VAL_PATH = '/zooper1/fontbakers/data/noCapsnoRepeatsVal/'

DIMENSIONS = (20, 30, 3, 2)

CHARACTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    't', 'u', 'v', 'w', 'x', 'y', 'z', 'zero', 'one', 'two', 'three', 'four',
    'five', 'six', 'seven', 'eight', 'nine', 'exclam', 'numbersign', 'dollar',
    'percent', 'ampersand', 'asterisk', 'question', 'at'
]
CLASS_INDEX = {label: x for x, label in enumerate(CHARACTERS)}

# List of paths to protobuf files
FONT_FILES_TRAIN_1 = glob(TRAIN_PATH_1 + '*')
FONT_FILES_TRAIN_2 = glob(TRAIN_PATH_2 + '*')
FONT_FILES_VAL = glob(VAL_PATH + '*')

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def validate(net, epoch_num):
    print('\tValidating...')

    valset = Dataset(FONT_FILES_VAL, DIMENSIONS)
    valloader = data.DataLoader(
        valset, batch_size=16, shuffle=True, num_workers=4)

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

    net = pantry.nets[FLAGS.pastry](device=DEVICE).to(DEVICE)
    optimizer, num_epochs = pantry.optims[FLAGS.pastry](net)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for path in [TRAIN_PATH_1, TRAIN_PATH_2]:
            trainset = Dataset(path, DIMENSIONS)
            trainloader = data.DataLoader(
                trainset, batch_size=16, shuffle=True, num_workers=4)

            running_loss = 0.0

            for i, (font, labels) in enumerate(trainloader):
                font = font.float().to(DEVICE)
                labels = [CLASS_INDEX[label] for label in labels]
                labels = torch.tensor(labels).to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward, backward, optimize
                prediction = net(font)
                loss = criterion(prediction, labels)
                loss.backward()
                optimizer.step()

                # Print statistics
                loss_ = loss.item()
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss_))

            print('\tFinished with shard.')

        # Validate at the end of every epoch.
        validate(net, epoch)

    print('Finished.')


if __name__ == '__main__':
    app.run(main)
