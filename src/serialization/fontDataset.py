import torch
from torch.utils import data
import font_pb2
from os.path import join
import numpy as np
import glob


class Dataset(data.Dataset):
    def __init__(self, proto_dir, dimensions):
        '''
        Parameters
        ----------
        proto_dir : directory of where all the glyph protos are located

        dimensions: the dimensions desired for each of the glyphs given in
            (num_contours, num_bezier_curves, 3,2)
        '''
        self.proto_dir = proto_dir
        self.dimensions = dimensions
        self.protobufs = glob.glob('{}/*'.format(proto_dir))

    def __len__(self):
        return len(self.protobufs)

    def __getitem__(self, index):
        proto = self.protobufs[index]
        glyph_proto = font_pb2.glyph()
        glyph_proto.ParseFromString(open(proto, 'rb').read())

        # we can call a little function here to reshape and also add padding
        # or alternativly it can be called on the other end
        return roll_pad_reshape(glyph_proto, self.dimensions)


def roll_pad_reshape(glyph_proto, dimensions):
    '''
        Parameters
        ----------
        glyph_proto : the proto object containing all the glyphs for that shard
        dimensions: the output dimensions we want to come out, The assumption is
        that this follows (contours,curves,points=3,coordinates=2)
        Returns:
        -------
        An numpy array with the dimensions specefied. Plus an additional
        dimension of the number of examples in that shard. 
        '''
    glyphs = glyph_proto.glyph
    dimensions = (len(glyphs), ) + dimensions
    reshaped_glyph = np.full(dimensions, 999.)
    for j, glyph_array in enumerate(glyphs):
        bezier_points = glyph_array.bezier_points
        contour_locations = glyph_array.contour_locations
        start = 0
        num_contours = min(len(contour_locations), dimensions[1])
        for i in range(num_contours):
            contour = contour_locations[i]
            roll_num = np.random.randint(20)
            # isolate the points for this contour
            end = contour * 3 * 2 + start  # conversion from curves to points
            current_points = bezier_points[start:end]
            # fold it into the correct shape and pop it in.
            folded_values = np.reshape(current_points, (1, 1, contour, 3, 2))
            curves_dim = min(contour, dimensions[2])
            reshaped_glyph[j, i, :curves_dim, :, :] = folded_values[
                0, 0, :curves_dim, :, :]
            start = end
    return reshaped_glyph, glyph_array.glyph_name
