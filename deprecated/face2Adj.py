#!/usr/bin/env python3
from freetype import *
from freetype.ft_enums.ft_curve_tags import FT_CURVE_TAG
import numpy as np
import matplotlib.pyplot as plt
import bezier
import sys
import string
import glob
import pickle
import pygsp

chars = string.ascii_letters + string.digits


def dist(i, j):
    return np.linalg.norm(i - j, 2)


def outline2adj(outline):
    CONIC = 0
    ON = 1
    CUBIC = 2

    points = np.array(outline.points, dtype=np.double).transpose()
    N = points.shape[1]
    trans = np.array([[np.amin(points[0, :])], [np.amin(points[1, :])]])
    max_dist = 0
    for ii in range(N):
        d = dist(0, points[:, ii])
        if d > max_dist:
            max_dist = d

    points = (points - trans) / max_dist
    # adjacency matrix, aij = 1 if points are part of the same curve
    # aii = 1 iff point is on curve
    # first two channels are meshgrid coordinates
    # third channel is adj matrix
    A = np.zeros((N, N, 3))
    for ii in range(N):
        A[ii, :, 0] = np.full(N, points[0, ii])
        A[ii, :, 1] = np.full(N, points[1, ii])
    tags = outline.tags
    curves = []

    j = 0  # pt indices
    # iterate over the number of contours
    for i in range(outline.n_contours):
        # starting point is just past the end of the previous contour
        if i == 0:
            start = 0
        else:
            start = outline.contours[i - 1] + 1
        j = start
        end = outline.contours[i]

        # iterate over all points in contour
        while j <= end:
            node_indices = []
            on_count = 0
            conic_count = 0
            cubic_count = 0
            while on_count < 2 and j <= end:
                # last point is first point
                if j == start and FT_CURVE_TAG(
                        tags[start]) != ON and FT_CURVE_TAG(tags[end]) == ON:
                    node_indices.append(end)
                    on_count += 1

                # adding node_indices points
                if FT_CURVE_TAG(tags[j]) == ON:
                    node_indices.append(j)
                    on_count += 1
                if FT_CURVE_TAG(tags[j]) == CONIC:
                    node_indices.append(j)
                    conic_count += 1
                if FT_CURVE_TAG(tags[j]) == CUBIC:
                    node_indices.append(j)
                    cubic_count += 1

                # first point is last point
                if j == end and FT_CURVE_TAG(
                        tags[start]) == ON and FT_CURVE_TAG(tags[end]) != ON:
                    node_indices.append(start)
                    on_count += 1
                # first and last point are off --> first and last point are the midpoint
                # add to the count to get out the loop, check again once outside
                if (j == start or j == end) and FT_CURVE_TAG(
                        tags[start]) != ON and FT_CURVE_TAG(tags[end]) != ON:
                    on_count += 1
                # allows end point of one curve to be start point of next curve
                if on_count < 2:
                    j += 1
            if j > end:
                break

            # filling in Adjacency matrix
            # all nodes in a curve are connected to each other
            for ii in node_indices:
                for jj in node_indices:
                    if ii != jj:
                        A[ii, jj, 2] = 1
            # diagonal element =1 means on curve point
            A[node_indices[0], node_indices[0], 2] = 1
            A[node_indices[-1], node_indices[-1], 2] = 1

            # nodes numpy array (2,num_pts)
            nodes = points[:, node_indices]

            if (j == start or j == end) and FT_CURVE_TAG(
                    tags[start]) != ON and FT_CURVE_TAG(tags[end]) != ON:
                midpoint = np.reshape((points[:, start] + points[:, end]) / 2,
                                      (2, 1))
                if j == start:
                    nodes = np.insert(nodes, 0, midpoint, axis=1)
                if j == end:
                    nodes = np.concatenate((nodes, midpoint), axis=1)
            if cubic_count == 1:
                print(
                    'too few cubic points for node_indices',
                    nodes,
                    file=sys.stderr)
            if conic_count > 1:
                k = 1
                mps = []
                r = 1
                while r < conic_count:
                    r += 1
                    midpoint = np.reshape((nodes[:, k] + nodes[:, k + 1]) / 2,
                                          (2, 1))
                    midpoint = np.repeat(midpoint, 2, axis=1)
                    nodes = np.insert(
                        nodes, k + 1, midpoint.transpose(), axis=1)
                    mps.append(k + 2)
                    k += 3

                split_nodes = np.split(nodes, mps, axis=1)
                curves.extend(split_nodes)

            # appending complete bezier curve
            else:
                curves.append(nodes)

        if FT_CURVE_TAG(tags[start]) == ON and FT_CURVE_TAG(tags[end]) == ON:
            nodes = points[:, (start, end)]
            curves.append(nodes)

    # CONVERTING CURVE LIST TO 4xMx2 NUMPY ARRAY,
    # WHERE M IS THE NUMBER OF CURVES IN THE GLYPH
    np_glyph = np.empty((4, len(curves), 2), dtype=np.double)
    for i in range(len(curves)):
        # elevate all curvers to 3rd order (4 pts)
        while curves[i].shape[1] < 4:
            c = bezier.Curve.from_nodes(curves[i])
            curves[i] = c.elevate().nodes
        np_glyph[:, i, :] = curves[i].transpose()

    return A


def plot_glyph(np_glyph, ax, animate=False, annotate=False, points=None):
    for i in range(np_glyph.shape[1]):
        curve = bezier.Curve.from_nodes(np_glyph[:, i, :].transpose())
        _ = curve.plot(num_pts=256, ax=ax)
        if animate is True:
            plt.draw()
            plt.pause(0.01)
    if annotate is True and points is not None:
        for i, pt in enumerate(points):
            ax.annotate(i, pt)


def face2adj(fn, w, h):
    face = Face(fn)
    face.set_char_size(48 * 64)
    face_adjs = []
    for char in chars:
        face.load_char(char)
        outline = face.glyph.outline
        A = outline2adj(outline)
        if A is -1:
            print('too many nodes, skipping')
            return -1
        face_adjs.append(A)
    return face_adjs


print('let\'s begin!')
img_size = 32
bez_dir = '../bez'
img_dir = '../img' + str(img_size)
adj_dir = './adj'
filenames = glob.glob('../fonts/*.*tf')
'''
for fn in filenames:
	print(fn)
	try:
		adjs = face2adj(fn,img_size,img_size)
	except:
		print('irregular, skipping')
		continue
	if adjs is -1:
		continue
	# save data
	fn = fn.replace(' ','')
	fn = os.path.basename(fn)
	outfile_adjs = open(adj_dir+'/adj_'+fn,'wb')
	pickle.dump(adjs,outfile_adjs)
	outfile_adjs.close()
print('done!')
'''

i = np.random.randint(len(filenames))
fn = filenames[i]
fn = fn.replace(' ', '')
fn = os.path.basename(fn)
infile_bez = open(bez_dir + '/bez_' + fn, 'rb')
infile_img = open(img_dir + '/img' + str(img_size) + '_' + fn, 'rb')
infile_adj = open(adj_dir + '/adj_' + fn, 'rb')
adj = pickle.load(infile_adj)
bez = pickle.load(infile_bez)
img = pickle.load(infile_img)
infile_adj.close()
infile_bez.close()
infile_img.close()

j = np.random.randint(len(chars))
A = adj[j]
G = pygsp.graphs.Graph(A[:, :, 2])
coords = A[:, 0, :2]
G.set_coordinates(coords)

fig, [ax1, ax2, ax3, ax4] = plt.subplots(1, 4)
plot_glyph(
    bez[j],
    ax1,
)
ax2.imshow(img[j])
G.plot(ax=ax3)
ax4.imshow(A)
plt.show()
