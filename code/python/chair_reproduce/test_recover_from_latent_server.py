import numpy as np

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
timecurrent = time.strftime('%m%d%H%M%S', time.localtime(time.time())) + '_' + str(random.randint(1000,9999))
test = True
parser = argparse.ArgumentParser()

# you must set this parameter
parser.add_argument('--output_dir', default='./result'+timecurrent, type = str)

# --------------------------not include in file name-------------------------------------
parser.add_argument('--model', default = 'scape', type = str) # training model name
parser.add_argument('--lz_d', nargs='+', default = [128, 32], type=int) # dimension of latent space
parser.add_argument('--l0', default = 10.0, type = float) # 1.0  generation loss
# parser.add_argument('--l1', default = 0.5, type = float) # 0.5  distance loss
parser.add_argument('--l2', default = 10.0, type = float) # 0.5 	weight loss
parser.add_argument('--l3', default = 1.0, type = float) # 0.005 Kl loss
parser.add_argument('--l4', default = 0.00, type = float) # 0.005 l2 loss
# parser.add_argument('--l5', default = 0.2, type = float) # 0.005 region loss
parser.add_argument('--joint', default = 0, type = int) # jointly training
parser.add_argument('--bin', default = 0, type = int) # use binary to train net
parser.add_argument('--trcet', default = 1.0, type = float) # training percent of the dataset
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
parser.add_argument('--epoch_deform', default=10000, type = int)
parser.add_argument('--epoch_structure', default=100000, type = int)

# inter
parser.add_argument('--beginid', default='3', type = str)
parser.add_argument('--endid', default='4', type = str)
parser.add_argument('--interids', nargs='+', default=['3'], type = str)

args = parser.parse_args()

if not args.autoencoder.find('tan') == -1:
	args.l3 = 0.00

args.featurefile = args.model + '_vaefeature.mat'
args.filename = args.model + '.obj'
args.iddatfile = args.model + '_vaeid.dat'

ininame = getFileName(args.output_dir, '.ini')
if len(ininame)>1:
    x = int(input('Please select a number:'))
else:
    x = 0
args = inifile2args(args, os.path.join(args.output_dir, ininame[x]))
[print('{}: {}'.format(x,k)) for x,k in vars(args).items()]
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

datainfo = Config(args, False)
datainfo.featurefile = args.featurefile
print(args.output_dir)
print(args.lz_d)
model = modelvae.convMESH(datainfo)

#------------------------------------------------------------get feature by id--------------------------------------------------------------------------------------
# for interid in args.interids:
# 	f, sf = model.get_features_by_id(datainfo, interid)
#------------------------------------------------------------get feature by id--------------------------------------------------------------------------------------

# model.recover_mesh(datainfo)
# model.random_gen(datainfo)
# model.interpolate1(datainfo, [args.beginid, args.endid])

import zmq
import sys
import zlib
import pickle
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://127.0.0.1:7720")

with tf.Session(config = model.config) as sess:
    # Read checkpoint
    tf.global_variables_initializer().run()
    _success, epoch = model.load(sess, model.checkpoint_dir_structure)
    if not _success:
        raise Exception("抛出一个异常")
    path = model.checkpoint_dir_structure +'/../interpolation_recover'+str(epoch)
    if not os.path.isdir(path):
        os.makedirs(path)

    while True:
        try:
            print("wait for client ...")
            message = socket.recv()

            pickle_c = zlib.decompress(message)
            code = pickle.loads(pickle_c)
            
            print('start inference...')
            result = model.recover_from_latent(sess, datainfo, [code])
            print('finished!')

            pickle_result = pickle.dumps(result)
            compressed_result = zlib.compress(pickle_result)

            socket.send(compressed_result)
        except Exception as e:
            print('異常:',e)
            sys.exit()

