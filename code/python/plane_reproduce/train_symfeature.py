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


# vae
timecurrent = time.strftime('%m%d%H%M%S', time.localtime(time.time()))
test = False
parser = argparse.ArgumentParser()

parser.add_argument('--model', default = 'chair', type = str)

parser.add_argument('--lz_d', nargs='+', default = [16], type=int)
parser.add_argument('--l0', default = 1000.0, type = float) # 1.0  generation loss
parser.add_argument('--l3', default = 1.0, type = float) # 0.005 Kl loss
parser.add_argument('--l4', default = 0.0, type = float) # 0.005 l2 loss
parser.add_argument('--lr', default = 0.001, type = float) # learning rate need adjust if the loss is too large in the early training
parser.add_argument('--batch_size', default = 10000, type = int)

parser.add_argument('--iddatfile', default = 'id.dat', type = str)
parser.add_argument('--autoencoder', default='vae', type = str)
parser.add_argument('--finaldim', default = 5, type = int)
parser.add_argument('--featurefile', default = 'hips_vaefeature.mat', type = str)
parser.add_argument('--maxepoch', default=500000, type = int)
parser.add_argument('--output_dir', default='./result'+timecurrent, type = str)

# parser.add_argument('--K', default=3, type = int)
# parser.add_argument('--gcnn', default=0, type = int)
# parser.add_argument('--layer', default = 3, type = int)
parser.add_argument('--fcvae', default = 1, type = int)
parser.add_argument('--numvae', default = 1, type = int)
parser.add_argument('--filename', default = 'horse.obj', type = str)
parser.add_argument('--controlid', nargs='+', default = [0], type=int)

args = parser.parse_args()

if not args.autoencoder.find('tan') == -1:
	args.l3 = 0.00

if not args.output_dir.find('./result'+timecurrent) == -1:
	a = './' + timecurrent + "-".join(["{}_{}".format(k,args.__dict__[k]) for k in sorted(args.__dict__.keys()) if len(k) < 6])
	a = a.replace(' ','')
	a = a.replace('[','')
	a = a.replace(']','')
	a = a.replace(',','--')
	args.output_dir = a.replace(',','--')

# args.output_dir = './1217154045l0_1000.0-l3_1.0-l4_0.0-lr_0.001-lz_d_16-model_chair'

args.featurefile = args.model + '_symfeature.mat'
args.iddatfile = args.model + '_symid.dat'

# if test:
	# recentfolder = traversalDir_FirstDir('./')
	# args.output_dir = './' + recentfolder[1]
	# print(args.output_dir)

datainfo = Config(args, symmetry = True)
argpaser2file(args, args.output_dir+'/'+timecurrent+'.ini')

print(args.output_dir)
print(args.lz_d)
model = modelvae.convMESH(datainfo, symmetry = True)


# print(safe_b64encode(str(para)))
# print(safe_b64decode(safe_b64encode(str(para))))

if not test:
	model.train_total_vae()

parafile = args.output_dir + '/convmesh-model-' + str(args.maxepoch)
model.recover_mesh(parafile, datainfo)
model.random_gen(parafile, datainfo)
model.interpolate1(parafile, datainfo, [469, 477])






