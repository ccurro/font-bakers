from absl import app, flags
import numpy as np
import torch
import matplotlib.pyplot as plt
from rasterizer import Rasterizer

FLAGS = flags.FLAGS
flags.DEFINE_string('file', None, 'Path to .out file containing Python dict.')
flags.DEFINE_string(
    'display', 'lowercase',
    ('If "lowercase", displays lowercase letters. If "all", displays all '
     'characters. Otherwise, interpret input as a list of characters to '
     'display (e.g. "abc" will display the first three lowercase letters.'))

LOWERCASE_LETTERS = list("abcdefghijklmnopqrstuvwxyz")
CHARACTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e',
    'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z', 'one', 'two', 'three', 'four', 'five', 'six',
    'seven', 'eight', 'nine', 'space', 'exclam', 'quotedbl', 'numbersign',
    'dollar', 'percent', 'ampersand', 'quotesingle', 'parenleft', 'parenright',
    'asterisk', 'plus', 'comma', 'hyphen', 'period', 'slash', 'colon',
    'semicolon', 'less', 'equal', 'greater', 'question', 'at', 'bracketleft',
    'bracketright', 'asciicircum', 'underscore', 'grave', 'braceleft', 'bar',
    'braceright', 'asciitilde'
]


def main(argv):
    if FLAGS.file is None:
        raise ValueError('--file flag not supplied.')

    use_cuda = torch.cuda.is_available() and not FLAGS.disable_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Using device "{}"'.format(device))

    # Upon reading in, control_points is a dict with keys: ints (indexing over
    # the number of closed curves in the glyph), values: lists of 2-tuples
    with open(FLAGS.file, 'rb') as f:
        control_points = eval(f.read())

    rasterizer = Rasterizer(
        resolution=FLAGS.resolution,
        steps=FLAGS.steps,
        sigma=FLAGS.sigma,
        method=FLAGS.method,
    )
    rasterizer.to(device)

    rasters = []

    if FLAGS.display == 'lowercase':
        characters_to_display = LOWERCASE_LETTERS
    elif FLAGS.display == 'all':
        # FIXME remove this exception
        raise ValueError(
            'Please don\'t set `--char all` until the rasterizer is faster. -GH'
        )
        characters_to_display = CHARACTERS
    else:
        characters_to_display = list(FLAGS.display)

    for character in characters_to_display:
        try:
            # Index
            character_control_points = control_points[character]

            # Flatten
            character_control_points = [
                bezier for closed_curve in character_control_points.values()
                for bezier in closed_curve
            ]

            # Rasterize
            character_control_points = torch.autograd.Variable(
                torch.Tensor(np.array(character_control_points) / 512),
                requires_grad=True,
            )
            raster = rasterizer.forward(character_control_points)
            rasters.append(raster.data.cpu().numpy())
        except KeyError:
            print(
                'Character {} not found in .out file; appending blank raster...'
                .format(character))
            blank_raster = np.zeros([FLAGS.resolution, FLAGS.resolution])
            rasters.append(blank_raster)

    # Plot
    if FLAGS.display == 'lowercase':
        nrowscols = [2, 13]
    elif FLAGS.display == 'all':
        nrowscols = [10, 10]
    else:
        nrowscols = [1, len(FLAGS.display)]

    plt.axis('off')
    fig, axarr = plt.subplots(
        nrows=nrowscols[0],
        ncols=nrowscols[1],
        sharex=True,
        sharey=True,
    )
    axarr = axarr.flatten()

    for idx, raster in enumerate(rasters):
        axarr[idx].matshow(raster)

    plt.show()


if __name__ == '__main__':
    app.run(main)