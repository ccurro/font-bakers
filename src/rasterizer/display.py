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

    use_cuda = torch.cuda.is_available() and not FLAGS.disablecuda
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
        use_cuda=use_cuda,
    )
    rasterizer.to(device)

    if FLAGS.display == 'lowercase':
        characters_to_display = LOWERCASE_LETTERS
    elif FLAGS.display == 'all':
        characters_to_display = CHARACTERS
    else:
        characters_to_display = list(FLAGS.display)

    # FIXME this for-try-except block be removed once Ostap finishes data
    # serialization. We would just deserialize an entire font.
    max_beziers = 80  # FIXME this number is temporary, pending Jonny.
    off_raster_point = (9999, 9999)  # Any coordinate that is off the raster
    batched_control_points = []
    for character in characters_to_display:
        try:
            # Index
            character_control_points = control_points[character]

            # Flatten
            character_control_points = [
                bezier for closed_curve in character_control_points.values()
                for bezier in closed_curve
            ]

            pad_to = max_beziers - len(character_control_points)
            character_control_points += pad_to * [3 * [off_raster_point]]
            batched_control_points.append(character_control_points)
        except KeyError:
            print(
                'Character {} not found in .out file; appending blank raster...'
                .format(character))
            batched_control_points.append(50 * [3 * [off_raster_point]])

    batched_control_points = torch.Tensor(batched_control_points)
    rasters = rasterizer.forward(batched_control_points)
    rasters = rasters.data.cpu().numpy()

    # Plot
    if FLAGS.display == 'lowercase':
        nrowscols = [2, 13]
    elif FLAGS.display == 'all':
        nrowscols = [10, 10]
    else:
        nrowscols = [1, len(FLAGS.display)]

    fig, axarr = plt.subplots(
        nrows=nrowscols[0],
        ncols=nrowscols[1],
        sharex=True,
        sharey=True,
        figsize=(15,4)
    )
    try:
        axarr = axarr.flatten()
    except AttributeError:
        # There is only one character to display, and `axarr` is just an AxesSubplot
        axarr = [axarr]

    for idx, raster in enumerate(rasters):
        axarr[idx].matshow(raster)
        axarr[idx].axis("off")
        axarr[idx].set_xticklabels([])
        axarr[idx].set_yticklabels([])
        axarr[idx].set_aspect('equal')

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()


if __name__ == '__main__':
    app.run(main)
