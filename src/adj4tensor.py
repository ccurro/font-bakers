#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pygsp
import glob
import string
import pickle
import sys
import h5py

chars = string.ascii_letters + string.digits
used_chars = chars
NODE_LIM = 48
'''
print('using chars %s with at most %d nodes'%(used_chars,NODE_LIM))

filenames = glob.glob('./adj/*')

valid_glyphs = []
labels = []
A_list = []

for fn in filenames:
	print(fn)
	infile = open(fn,'rb')
	adjs = pickle.load(infile)
	A_list.extend(adjs)
	labels.extend(range(len(chars)))
	infile.close()

count = np.zeros(len(used_chars))
for i, A in enumerate(A_list):
	j = i%(len(chars))
	if j==0:
		print('\n')
	if A.shape[0]<=NODE_LIM and (not np.isnan(A).any()) and np.amax(A)!=0:
		count[j]+=1
		valid_glyphs.append(i)
		print(used_chars[j],end='')

L = len(valid_glyphs)
print('\nnum valid glyphs:',L)
print(count/L)
'''

filename = 'adj_dset.hdf5'
'''
f = h5py.File(filename,'w')
dset_imgs   = f.create_dataset('imgs', (L,NODE_LIM,NODE_LIM,3), chunks=True, 
	maxshape=(None,NODE_LIM,NODE_LIM,3))
dset_labels = f.create_dataset('labels', (L,), dtype='i8', chunks=True)
	
k = 0
for i,val in enumerate(valid_glyphs):
	print(i,end='')
	# pad adj matrix to node_lim
	n = A_list[val].shape[0]
	A = np.zeros((NODE_LIM,NODE_LIM,3))
	A[:n,:n,:] = A_list[val]/np.amax(A_list[val])
	dset_imgs[k] = A
	# make onehot label
	dset_labels[k] = val%(len(chars))
	k+=1
	sys.stdout.flush()
	# f.flush()
f.close()
'''

f = h5py.File(filename, 'r')
Adj_tensor = f['imgs']
print(Adj_tensor.shape)
labels = f['labels']
print('nanlocs', np.argwhere(np.isnan(Adj_tensor)))


def plotAdj(A, ax):
    G = pygsp.graphs.Graph(A[:, :, 2])
    coords = A[:, 0, :2]
    G.set_coordinates(coords)
    G.plot(ax=ax)


L = labels.shape[0]

for i in range(30, 50):
    A = Adj_tensor[i, :, :, :]
    j = int(labels[i])
    print('char=', used_chars[j])
    fig, [ax1, ax2] = plt.subplots(1, 2)
    plotAdj(A, ax1)
    ax2.imshow(A)
plt.show()
