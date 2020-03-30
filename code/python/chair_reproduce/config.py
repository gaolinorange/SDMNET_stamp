import os, time
import sys
import numpy as np
import tensorflow as tf
from utils import *
from numpy.linalg import pinv
import glob
import shutil
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))


class Config():
    def __init__(self, args, istraining = True):
        self.fcvae = bool(int(args.fcvae)) # symmetry
        self.changenet = bool(int(args.bin))
        # self.hidden_dim = [4, 8, 16, 32, 64]
        if istraining:
            self.hidden_dim = args.lz_d
        else:
            self.hidden_dim = [int(x) for x in eval(args.lz_d)]
        self.batch_size = int(args.batch_size)
        self.featurefile = args.featurefile
        self.jointly = bool(int(args.joint))
        self.train_percent = float(args.trcet)
        # self.neighbourfile = args.neighbourfile
        # self.distancefile = args.distancefile

        self.lambda0 = float(args.l0)
        # self.lambda1 = args.l1
        self.lambda2 = float(args.l2)
        self.lambda3 = float(args.l3)
        self.lambda4 = float(args.l4)
        # self.lambda5 = args.l5
        self.lr = float(args.lr)

        self.K = int(args.K)
        self.graph_conv = bool(int(args.gcnn))
        self.layers = int(args.layer)
        self.activate = args.activ

        self.finaldim = int(args.finaldim)
        self.epoch_deform = int(args.epoch_deform)
        self.epoch_structure = int(args.epoch_structure)
        self.output_dir = args.output_dir
        self.use_ae_or_vae = args.autoencoder
        if istraining:
            self.control_idx = args.controlid
        else:
            self.control_idx = [int(x) for x in eval(args.controlid)]

        self.num_vaes = int(args.numvae)
        self.hidden_dim = self.hidden_dim[0:self.num_vaes]
        print(type(self.hidden_dim))
        assert(self.num_vaes == len(self.hidden_dim))
        # self.latent_dim = [c//2 for c in self.hidden_dim]

        # if not symmetry:
        #     self.feature, self.neighbour, self.degrees, self.logrmin, self.logrmax, self.smin, self.smax, self.modelnum, self.pointnum, self.maxdegree, \
        #     self.L1, self.cotw1, self.laplacian, self.reconmatrix, self.vdiff, self.all_vertex = load_datanew(self.featurefile, self.graph_conv)
        #     self.vertex_dim = 9
        # else:
        #     self.feature, self.pointnum, self.vertex_dim, self.modelnum, self.sym_featuremin, self.sym_featuremax = load_data_sym(self.featurefile)
        #     self.neighbour = []
        #     self.degrees = []
        #     self.logrmin = []
        #     self.logrmax = []
        #     self.smin = []
        #     self.smax = []
        #     self.maxdegree = []
        #     self.L1, self.cotw1, self.laplacian, self.reconmatrix, self.vdiff, self.all_vertex = [], [], [], [], [], []
        #     self.mesh

        if args.model == 'chair':
            self.part_name = ['armrest_1', 'armrest_2', 'back', 'leg1_1', 'leg1_2', 'leg2_1', 'leg2_2', 'seat']
            # part_names = {'armrest_1', 'armrest_2', 'leg1_1', 'leg1_2', 'leg2_1', 'leg2_2', 'seat'};
            # part_names = {'back', 'seat'};
        elif args.model == 'knife':
            self.part_name = ['part1', 'part2']
        elif args.model == 'guitar':
            self.part_name = ['part1', 'part2', 'part3']
        elif args.model == 'skateboard':
            self.part_name = ['surface', 'bearing1', 'bearing2', 'wheel1_1', 'wheel1_2', 'wheel2_1', 'wheel2_2']
        elif args.model == 'cup':
            self.part_name = ['part1', 'part2']
        elif args.model == 'car':
            self.part_name = ['body', 'left_front_wheel', 'right_front_wheel', 'left_behind_wheel', 'right_behind_wheel']
        elif args.model == 'plane':
            self.part_name = ['body', 'left_wing', 'right_wing', 'left_tail', 'right_tail', 'upper_tail', 'down_tail', 'front_landing_gear', 'left_landing_gear', 'right_landing_gear', 'left_engine_1', 'right_engine_1', 'left_engine_2', 'right_engine_2']
        elif args.model == 'table':
            self.part_name = ['surface', 'leg1_1', 'leg1_2', 'leg2_1', 'leg2_2', 'leg3_1', 'leg3_2', 'leg4_1', 'leg4_2']
        else:
            raise Exception("抛出一个异常")

        self.feature, self.neighbour, self.degrees, self.logrmin, self.logrmax, self.smin, self.smax, self.modelnum, self.pointnum, self.maxdegree, \
        self.L1, self.cotw1, self.laplacian, self.reconmatrix, self.vdiff, self.all_vertex, self.symmetry_feature, self.boxmin, self.boxmax, self.modelname, \
        self.part_name = load_datanew(self.featurefile, self.changenet, self.graph_conv)
        self.vertex_dim = 9


        if os.path.exists(args.filename):
            self.mesh = readmesh(args.filename)
        else:
            print('the basemesh is not exist')
            exit()

        L = self.laplacian
        print(self.pointnum)

        for i in self.control_idx:
            temp = np.zeros((1, self.pointnum))
            temp[0, i] = 1
            L = np.concatenate((L,temp))

        # self.deform_reconmatrix = np.dot(pinv(np.dot(L.transpose(), L)), L.transpose())
        self.deform_reconmatrix = self.reconmatrix # if don't need align the shape,please use this line, otherwise, comment this line and uncomment the former line
        # self.deform_reconmatrix = []


        timecurrent = time.strftime('%m%d%H%M%S', time.localtime(time.time()))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.output_dir+'/code'+timecurrent):
            os.makedirs(self.output_dir+'/code'+timecurrent)
        [os.system('cp '+file +' %s' % self.output_dir + '/code'+timecurrent+'/\n') for file in glob.glob(r'./code/*.py')]

        if os.path.exists(os.path.join(self.output_dir, 'log.txt')):
            self.flog = open(os.path.join(self.output_dir, 'log.txt'), 'a')
            printout(self.flog, 'add '+timecurrent)
        else:
            self.flog = open(os.path.join(self.output_dir, 'log.txt'), 'w')

        self.iddat_name = args.iddatfile
        self.train_id, self.valid_id = spilt_dataset(len(self.feature), self.train_percent, self.iddat_name)
        if not istraining:
            printout(self.flog, 'add test')
        else:
            argpaser2file(args, args.output_dir+'/'+timecurrent+'.ini')



