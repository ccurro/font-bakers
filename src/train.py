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

DATA_PATH = '/zooper1/fontbakers/data/singleExampleProtos/'
DIMENSIONS = (20, 30, 3, 2)
TRAIN_TEST_SPLIT = 0.8


def validate(epoch_num):
    print('\tValidating...')

    valset = Dataset(font_files_val, DIMENSIONS)
    valloader = data.DataLoader(
        valset, batch_size=16, shuffle=True, num_workers=4)

    correct = 0
    total = 0
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for font, labels in valloader:
            font = font.float().to(device)
            labels = [class_index[label] for label in labels]
            labels = torch.tensor(labels).to(device)

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
    classes = class_index.keys()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_{}.png'.format(epoch_num))

    print('\tAccuracy on validation fonts: %d %%' % (100 * correct / total))
    print(confusion)
    print('\n\n\n')


def main(argv):
    print('Starting...')

    characters = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b',
        'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'zero', 'one', 'two',
        'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'exclam',
        'numbersign', 'dollar', 'percent', 'ampersand', 'asterisk', 'question',
        'at'
    ]
    class_index = {label: x for x, label in enumerate(characters)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    font_files = glob(DATA_PATH + '*')  # List of paths to protobuf files
    split = int(TRAIN_TEST_SPLIT * len(font_files))
    font_files_train = font_files[:split]
    font_files_val = font_files[split:]

    trainset = Dataset(font_files_train, DIMENSIONS)
    trainloader = data.DataLoader(
        trainset, batch_size=16, shuffle=True, num_workers=4)

    net = pantry.nets[FLAGS.pastry](device=device).to(device)
    optimizer, num_epochs = pantry.optims[FLAGS.pastry](net)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (font, labels) in enumerate(trainloader):
            font = font.float().to(device)
            labels = [class_index[label] for label in labels]
            labels = torch.tensor(labels).to(device)

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

        # Validate at the end of every epoch.
        validate(epoch)

    print('Finished.')


if __name__ == '__main__':
    app.run(main)
