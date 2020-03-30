import tensorflow as tf
import numpy as np
import scipy.io as sio
from six.moves import xrange
import time
import h5py
import pickle
import argparse
from utils import *
import model_stacknewvae as modelvae
import os
from config import *
import icp
import random


# vae
timecurrent = time.strftime('%m%d%H%M', time.localtime(time.time())) + '_' + str(random.randint(1000,9999))
test = False
parser = argparse.ArgumentParser()

parser.add_argument('--model', default = 'scape', type = str) # training model name
parser.add_argument('--l0', default = 10.0, type = float) # 1.0  generation loss
# parser.add_argument('--l1', default = 0.5, type = float) # 0.5  distance loss
parser.add_argument('--l2', default = 10.0, type = float) # 0.5 	weight loss
parser.add_argument('--l3', default = 1.0, type = float) # 0.005 Kl loss
parser.add_argument('--l4', default = 0.001, type = float) # 0.005 l2 loss
# parser.add_argument('--l5', default = 0.2, type = float) # 0.005 region loss
parser.add_argument('--joint', default = 0, type = int) # jointly training
parser.add_argument('--bin', default = 0, type = int) # use binary to train net
parser.add_argument('--trcet', default = 1.0, type = float) # training percent of the dataset

# --------------------------not include in file name-------------------------------------
parser.add_argument('--lz_d', nargs='+', default = [128, 32], type=int) # dimension of latent space
parser.add_argument('--layer', default = 2, type = int) # the number of mesh convolution layer
parser.add_argument('--activ', default = '', type = str) # activate function of structure net : every layer
parser.add_argument('--lr', default = 0.001, type = float) # learning rate need adjust if the loss is too large in the early training
parser.add_argument('--fcvae', default = 0, type = int)
parser.add_argument('--batch_size', default = 64, type = int)
parser.add_argument('--K', default=3, type = int)
parser.add_argument('--gcnn', default=0, type = int)
parser.add_argument('--iddatfile', default = 'id.dat', type = str)
parser.add_argument('--autoencoder', default='vae', type = str)
parser.add_argument('--numvae', default = 2, type = int)
parser.add_argument('--finaldim', default = 9, type = int)
parser.add_argument('--filename', default = 'horse.obj', type = str)
parser.add_argument('--featurefile', default = 'hips_vaefeature.mat', type = str)
parser.add_argument('--controlid', nargs='+', default = [0], type=int)
parser.add_argument('--epoch_deform', default=15000, type = int)
parser.add_argument('--epoch_structure', default=100000, type = int)
parser.add_argument('--output_dir', default='./result'+timecurrent, type = str)

args = parser.parse_args()

if not args.autoencoder.find('tan') == -1:
	args.l3 = 0.00

args.featurefile = args.model + '_vaefeature.mat'
args.filename = args.model + '.obj'
args.iddatfile = args.model + '_vaeid.dat'
theList = ['K','gcnn','nvae','fcvae','activ','lr','layer','lz_d']
if not args.output_dir.find('./result'+timecurrent) == -1:
	a = './' + timecurrent + "-".join(["{}_{}".format(k,args.__dict__[k]) for k in sorted(args.__dict__.keys()) if len(k) < 6 and k not in theList])
	a = a.replace(' ','')
	a = a.replace('[','')
	a = a.replace(']','')
	a = a.replace(',','--')
	args.output_dir = a.replace(',','--')

#args.output_dir = './1219002003K_3-fcvae_0-gcnn_0-l0_1000.0-l3_1.0-l4_0.1-layer_2-lr_0.001-lz_d_128-model_chairbacksub'

# parafile = args.output_dir + '/checkpoint/convmesh-modelbest'# + str(args.maxepoch)

# if not os.path.exists(parafile + '.index'):
# 	training = True
# else:
# 	training = False

# if test:
	# recentfolder = traversalDir_FirstDir('./')
	# args.output_dir = './' + recentfolder[1]
	# print(args.output_dir)

datainfo = Config(args)

print(args.output_dir)
print(args.lz_d)
model = modelvae.convMESH(datainfo)


# print(safe_b64encode(str(para)))
# print(safe_b64decode(safe_b64encode(str(para))))


# hidden_dim = args.hiddendim
# featurefile = args.featurefile
# neighbourfile = args.neighbourfile
# neighbourvariable = args.neighbourvariable
# distancefile = args.distancefile
# distancevariable = args.distancevariable
# lambda0 = args.l0
# lambda1 = args.l1
# lambda2 = args.l2
# lambda3 = args.l3
# lr = args.lr
# finaldim = args.finaldim
# layers = args.layers
# maxepoch = args.maxepoch

# feature, logrmin, logrmax, smin, smax, pointnum = load_data(featurefile)

# neighbour, degrees, maxdegree = load_neighbour(neighbourfile, neighbourvariable, pointnum)

# geodesic_weight = load_geodesic_weight(distancefile, distancevariable, pointnum)



# #model.train(feature, geodesic_weight, maxepoch)
if not test:
	model.train_scvae()
# # model_stack.train_vae_inorder(datainfo.geodesic_weight)
#model.individual_dimension_vae(args.output_dir + '/convmesh-model-20000', datainfo)
# recover_data1(model.base)
#model.output_res_feature(args.output_dir + '/convmesh-model-20000', datainfo)
# parafile = args.output_dir + '/convmesh-model-' + str(max(args.epoch_deform, args.epoch_structure))
with tf.device('/cpu:0'):
	model.recover_mesh(datainfo)
	model.random_gen(datainfo)
	model.interpolate1(datainfo, [3, 4])
# model.interpolate1(args.output_dir + '/convmesh-model-2000', datainfo, [2,3])
# model.recover_mesh(args.output_dir + '/convmesh-model-4000', datainfo)
# model.recover_mesh(args.output_dir + '/convmesh-model-6000', datainfo)
# model.recover_mesh(args.output_dir + '/convmesh-model-8000', datainfo)

# model.recover_mesh(args.output_dir + '/convmesh-modelbest', datainfo)
# model.individual_dimension(args.output_dir + '/convmesh-modelbest', datainfo)

# model.get_res_feature(args.output_dir + '/convmesh-modelbest', datainfo)



# with tf.Graph().as_default():
#     a = tf.constant([[1,2],[3,4],[5,6]],name='a')
#     b = tf.constant([[[1,2],[3,4],[5,6]],[[1,2],[3,4],[7,8]]])
#     # b = tf.tile(a,[2,3])
#     print(np.shape(a))
#     print(np.shape(b))
#     sess = tf.Session()
#     print(sess.run(b-a))

# mesh1 = readmesh('scape.obj')
# mesh2 = readmesh('4.obj')
# v1=np.zeros((2500, 3)).astype('float32')
# v2=np.zeros((2500, 3)).astype('float32')
# point_array1 = mesh1.points()
# point_array2 = mesh2.points()

# T,R,t = icp.best_fit_transform(point_array2, point_array1)
# C = np.ones((np.shape(point_array2)[0], 4))
# C[:,0:3] = point_array2
# align_deforma_v = (np.dot(T, C.T).T)[:,0:3]

# savemesh(mesh2, 'align.obj', align_deforma_v)



