import tensorflow as tf
import numpy as np
import scipy.io as sio
from six.moves import xrange
import time
import h5py
import pickle

import argparse
from utils import *
# import model
import openmesh as om
# from vispy.plot import Fig
# import vispy.io as io


# with tf.Graph().as_default():
#     a = tf.constant([[1,2],[3,4],[5,6]],name='a')
#     b = tf.constant([[[1,2],[3,4],[5,6]],[[1,2],[3,4],[7,8]]])

#     dataset = tf.data.Dataset.from_tensor_slices((a, a))
#     dataset = dataset.shuffle(buffer_size=1000).batch(2).repeat(10)
#     print(dataset.output_shapes)
#     print(np.shape(dataset))
#     # b = tf.tile(a,[2,3])
#     print(np.shape(a))
#     print(np.shape(b))
#     sess = tf.Session()
#     bb,cc,dd,ff = sess.run([[a,a],[b,b],[a,b],dataset])
#     print(bb)
#     print(cc)
#     print(dd)
#     print(ff)
# EPOCHS = 10
# x, y = tf.placeholder(tf.float32, shape=[None,2]), tf.placeholder(tf.float32, shape=[None,1])
# dataset = tf.data.Dataset.from_tensor_slices((x, y))
# train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
# test_data = (np.array([[1,2]]), np.array([[0]]))
# print(train_data)
# print(test_data)
# iter = dataset.make_initializable_iterator()
# features, labels = iter.get_next()
# with tf.Session() as sess:
# #     initialise iterator with train data
#     sess.run(iter.initializer, feed_dict={ x: train_data[0], y: train_data[1]})
#     for _ in range(5):
#         print(sess.run([features, labels]))
# #     switch to test data
#     sess.run(iter.initializer, feed_dict={ x: test_data[0], y: test_data[1]})
#     print(sess.run([features, labels]))

#     sess.run(iter.initializer, feed_dict={ x: train_data[0], y: train_data[1]})
#     for _ in range(5):
#         print(sess.run([features, labels]))
# #     switch to test data
#     sess.run(iter.initializer, feed_dict={ x: test_data[0], y: test_data[1]})
#     print(sess.run([features, labels]))
# timecurrent = time.strftime('%m%d%H%M%S', time.localtime(time.time())) + '_' + str(random.randint(1000,9999))
# parser = argparse.ArgumentParser()

# # you must set this parameter
# parser.add_argument('--output_dir', default='./result'+timecurrent, type = str)

# # --------------------------not include in file name-------------------------------------
# parser.add_argument('--model', default = 'scape', type = str) # training model name
# parser.add_argument('--lz_d', nargs='+', default = [128, 32], type=int) # dimension of latent space
# parser.add_argument('--l0', default = 10.0, type = float) # 1.0  generation loss
# # parser.add_argument('--l1', default = 0.5, type = float) # 0.5  distance loss
# parser.add_argument('--l2', default = 10.0, type = float) # 0.5 	weight loss
# parser.add_argument('--l3', default = 1.0, type = float) # 0.005 Kl loss
# parser.add_argument('--l4', default = 0.00, type = float) # 0.005 l2 loss
# # parser.add_argument('--l5', default = 0.2, type = float) # 0.005 region loss
# parser.add_argument('--joint', default = 0, type = int) # jointly training
# parser.add_argument('--bin', default = 0, type = int) # use binary to train net
# parser.add_argument('--trcet', default = 1.0, type = float) # training percent of the dataset
# parser.add_argument('--layer', default = 2, type = int) # the number of mesh convolution layer
# parser.add_argument('--activ', default = '', type = str) # activate function of structure net : every layer
# parser.add_argument('--lr', default = 0.001, type = float) # learning rate need adjust if the loss is too large in the early training
# parser.add_argument('--fcvae', default = 0, type = int)
# parser.add_argument('--batch_size', default = 64, type = int)
# parser.add_argument('--K', default=3, type = int)
# parser.add_argument('--gcnn', default=0, type = int)
# parser.add_argument('--iddatfile', default = 'id.dat', type = str)
# parser.add_argument('--autoencoder', default='vae', type = str)
# parser.add_argument('--numvae', default = 2, type = int)
# parser.add_argument('--finaldim', default = 9, type = int)
# parser.add_argument('--filename', default = 'horse.obj', type = str)
# parser.add_argument('--featurefile', default = 'hips_vaefeature.mat', type = str)
# parser.add_argument('--controlid', nargs='+', default = [0], type=int)
# parser.add_argument('--epoch_deform', default=10000, type = int)
# parser.add_argument('--epoch_structure', default=100000, type = int)


# args = parser.parse_args()

# print(type(args.epoch_structure))

def load(sess, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
    # saver = self.saver

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    # import the inspect_checkpoint library
    from tensorflow.python.tools import inspect_checkpoint as chkp

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

        # saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        # print all tensors in checkpoint file
        chkp.print_tensors_in_checkpoint_file(os.path.join(checkpoint_dir, ckpt_name), tensor_name='', all_tensors=False,all_tensor_names=True)
        # chkp._count_total_params

        # if not ckpt_name.find('best') == -1:
        #     counter = 0
        # else:
        #     counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))

        # print(" [*] Success to read {}".format(ckpt_name))
        # return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        # return False, 0  # model = convMESH()

# inifile2args(args, ininame='0425171950.ini')
with tf.Session() as sess:
    load(sess, '')



# Reinitializable iterator to switch between Datasets
# EPOCHS = 10
# # making fake data using numpy
# train_data = (np.random.sample((3,2)), np.random.sample((3,1)))
# test_data = (np.random.sample((10,2)), np.random.sample((10,1)))
# print(train_data)
# print('aa')
# print(test_data)

# # create two datasets, one for training and one for test
# train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
# test_dataset = tf.data.Dataset.from_tensor_slices(test_data)

# train_dataset = train_dataset.shuffle(buffer_size=10000).batch(3)
# # train_dataset = train_dataset.shuffle(buffer_size=10000)
# # train_dataset = train_dataset.batch(self.batch_size)

# # create a iterator of the correct shape and type
# iter_train = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
# iter_test = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)
# features_train, labels_train = iter_train.get_next()
# features_test, labels_test = iter_test.get_next()
# # create the initialisation operations
# train_init_op = iter_train.make_initializer(train_dataset)
# test_init_op = iter_test.make_initializer(test_dataset)
# with tf.Session() as sess:
#     sess.run(train_init_op) # switch to train dataset
#     sess.run(test_init_op) # switch to val dataset
#     for _ in range(EPOCHS):
#         print('train')
#         print(sess.run([features_train, labels_train]))
#         print('test')
#         print(sess.run([features_test, labels_test]))



# parser = argparse.ArgumentParser()

# parser.add_argument('--hiddendim', default = 50, type = int)
# parser.add_argument('-f', '--featurefile', default = 'scapefeature.mat', type = str)
# parser.add_argument('-n', '--neighbourfile', default = 'scapeneighbour.mat', type = str)
# parser.add_argument('--neighbourvariable', default = 'neighbour', type = str)
# # parser.add_argument('-d', '--distancefile', default = 'scapedistance.mat', type = str)
# # parser.add_argument('--distancevariable', default = 'distance', type = str)
# parser.add_argument('--l1', default = 0.5, type = float)
# parser.add_argument('--l2', default = 0.5, type = float)
# parser.add_argument('--lr', default = 0.001, type = float)
# parser.add_argument('--finaldim', default = 9, type = int)
# parser.add_argument('-l', '--layers', default = 1, type = int)
# # parser.add_argument('-m', '--maxepoch', default=2000, type = str)
# #parser.add_argument('--modelfile', default = 'convmesh-modelbest', type = str)
# parser.add_argument('--output_dir', default='./result', type = str)

# args = parser.parse_args()
# argpaser2file(args,args.output_dir+'/example_test.ini')

# hidden_dim = args.hiddendim
# featurefile = args.featurefile
# neighbourfile = args.neighbourfile
# neighbourvariable = args.neighbourvariable
# # distancefile = args.distancefile
# # distancevariable = args.distancevariable
# lambda1 = args.l1
# lambda2 = args.l2
# lr = args.lr
# finaldim = args.finaldim
# layers = args.layers
# modelfile = args.output_dir+'/convmesh-modelbest'
# # maxepoch = args.maxepoch

# feature, logrmin, logrmax, smin, smax, pointnum = load_data(featurefile)

# neighbour, degrees, maxdegree = load_neighbour(neighbourfile, neighbourvariable, pointnum)

# # geodesic_weight = load_geodesic_weight(distancefile, distancevariable, pointnum)

# model = model.convMESH(pointnum, neighbour, degrees, maxdegree, hidden_dim, finaldim, layers, lambda1, lambda2, lr)

# model.individual_dimension(modelfile, feature, logrmin, logrmax, smin, smax)

# mesh1 = readmesh('chairbacksub.obj')


# point_array1 = mesh1.points()
# # point_array2 = mesh2.points()
# face = mesh1.face_vertex_indices()

# fig = Fig()
# ax_left = fig[0, 0]

# image=ax_left.mesh(vertices=point_array1, faces=face, vertex_colors=None, face_colors=None, color=(0.5, 0.5, 1.0), fname=None, meshdata=None)
# print(image)
# io.write_png("wonderful.png",image)
# C = np.ones((np.shape(point_array2)[0], 4))
# C[:,0:3] = point_array2
# align_deforma_v = (np.dot(T, C.T).T)[:,0:3]

# savemesh(mesh2, 'align.obj', align_deforma_v)

# from numpy import load
# from vispy.geometry import MeshData
# from vispy.io.image import write_png
# from vispy.plot import Fig

# # download https://dl.dropboxusercontent.com/u/66601/fsaverage.npz
# # surf = load('fsaverage.npz')

# meshdata = MeshData(vertices=point_array1, faces=face)

# fig = Fig()
# plt = fig[0, 0]
# SKIN_COLOR = 0.94, 0.82, 0.81, 1.

# plt.mesh(meshdata=meshdata, color=SKIN_COLOR)

# plt.view.camera.center = (35, -18, 15)
# plt.view.camera.scale_factor = 128
# plt.view.camera.elevation = 0
# plt.view.camera.azimuth = 90

# img = fig.render()
# write_png('rendered_default.png', img)