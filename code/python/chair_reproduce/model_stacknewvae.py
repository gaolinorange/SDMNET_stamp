import tensorflow as tf
import numpy as np
import time,os,random
import scipy.io as sio
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline
import scipy.interpolate as interpolate
import icp
import shutil
xrange = range

from utils import *
from render import *

timeline_use = False
tensorboard = False
train_inorder = False
advance_api = True

def g():
    raise Exception("抛出一个异常")

class convMESH():

    VAE = 'SCVAE'
    # config.log_device_placement = True  # 是否打印设备分配日志
    # config.allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    no_opt = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.ON_2,
                            do_common_subexpression_elimination=True,
                            do_function_inlining=True,
                            do_constant_folding=True)
    gpu_opt = tf.GPUOptions(allocator_type = 'BFC', allow_growth = True)
    config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=no_opt),
                        allow_soft_placement=False, gpu_options = gpu_opt )

    def __init__(self, datainfo):
        self.activate = datainfo.activate
        self.change_net = datainfo.changenet
        self.union = datainfo.jointly                                                     # training jointly or separately
        self.interval = 50
        self.modelname = datainfo.modelname
        fcvae = datainfo.fcvae
        self.featurefile = datainfo.featurefile
        self.part_name = datainfo.part_name
        self.part_num = len(datainfo.part_name)
        self.part_dim = 2*self.part_num+9
        self.cube_point_num = np.shape(datainfo.mesh.points())[0]
        self.cube_vertex_dim = 9
        # self.max_degree = sel

        self.batch_size = datainfo.batch_size
        self.pointnum = datainfo.pointnum                                        # symmetry feature
        self.vertex_dim = datainfo.vertex_dim                                    # symmetry feature
        self.hiddendim = datainfo.hidden_dim                                     # symmetry feature
        # self.latent_dim = datainfo.latent_dim
        self.maxdegree = datainfo.maxdegree
        self.finaldim = datainfo.finaldim                                        # symmetry feature
        self.layers = datainfo.layers
        self.lambda0 = datainfo.lambda0                                          # symmetry feature
        # self.lambda1 = datainfo.lambda1
        self.lambda2 = datainfo.lambda2
        self.lambda3 = datainfo.lambda3                                          # symmetry feature
        self.lambda4 = datainfo.lambda4                                          # symmetry feature
        # self.lambda5 = datainfo.lambda5
        self.lr = datainfo.lr                                                    # symmetry feature
        self.decay_step = 5000
        self.decay_rate = 0.8
        self.maxepoch_deform = datainfo.epoch_deform                                       # symmetry feature
        self.maxepoch_structure = datainfo.epoch_structure
        self.K = datainfo.K
        self.graph_conv = datainfo.graph_conv

        self.num_vaes = datainfo.num_vaes
        # self.dmin = datainfo.dmin
        # self.dmax = datainfo.dmax
        self.reconm = datainfo.reconmatrix
        self.w_vdiff = datainfo.vdiff
        self.control_id = datainfo.control_idx
        self.all_vertex = datainfo.all_vertex
        self.deform_reconmatrix_holder = datainfo.deform_reconmatrix
        self.mesh = datainfo.mesh

        self.neighbourmat = datainfo.neighbour
        self.degrees = datainfo.degrees
        self.feature = datainfo.feature                                          # symmetry feature
        self.symmetry_feature = datainfo.symmetry_feature
        # self.geodesic_weight = datainfo.geodesic_weight
        self.cot_w = datainfo.cotw1
        self.L1_ = datainfo.L1 # graph conv

        self.outputdir = datainfo.output_dir                                     # symmetry feature
        self.flog = datainfo.flog                                                # symmetry feature
        self.iddat_name = datainfo.iddat_name                                    # symmetry feature
        self.train_percent = datainfo.train_percent
        self.train_id = datainfo.train_id
        self.valid_id = datainfo.valid_id

        if not datainfo.use_ae_or_vae.find('tan') == -1:
            self.ae = True
            self.lambda3 = 0.0
            datainfo.lambda3 = 0.0
        else:
            self.ae = False

        # tf.set_random_seed(1)

        self.inputs_feature = tf.placeholder(tf.float32, [None, self.part_num, self.cube_point_num, self.cube_vertex_dim], name = 'input_mesh')
        self.inputs_symmetry = tf.placeholder(tf.float32, [None, self.part_num, self.part_dim], name = 'input_mesh_sym')

        if advance_api:
            # self.caseid = tf.placeholder(tf.int32, shape=(), name = 'input_id')
            self.handle = tf.placeholder(tf.string, shape=[])
            #train
            dataset_train = tf.data.Dataset.from_tensor_slices((self.inputs_feature, self.inputs_symmetry))
            dataset_valid = tf.data.Dataset.from_tensor_slices((self.inputs_feature, self.inputs_symmetry))
            dataset_app = tf.data.Dataset.from_tensor_slices((self.inputs_feature, self.inputs_symmetry))
            # dataset = dataset.map(...)
            dataset_train = dataset_train.shuffle(buffer_size = 10000).batch(self.batch_size)
            dataset_valid = dataset_valid.shuffle(buffer_size = 10000).batch(self.batch_size)
            dataset_app = dataset_app.batch(len(self.feature))

            iterator = tf.data.Iterator.from_string_handle(self.handle, dataset_train.output_types, dataset_train.output_shapes)
            inputs_feature, inputs_symmetry = iterator.get_next()
            self.train_iterator = dataset_train.make_initializable_iterator()
            self.valid_iterator = dataset_valid.make_initializable_iterator()
            self.app_iterator = dataset_app.make_initializable_iterator()

            # def data_flow(id):
            #     if id == 0:
            #         dataset = dataset.repeat(self.maxepoch_deform+self.maxepoch_structure).shuffle(buffer_size=10000).batch(self.batch_size)
            #     elif id == 1:
            #         dataset = dataset.shuffle(buffer_size = 10000).batch(self.batch_size)
            #     else:
            #         dataset = dataset
            #     iterator = dataset.make_initializable_iterator()
            #     inputs_feature, inputs_symmetry = self.iterator_train.get_next()
            #     return iterator, inputs_feature, inputs_symmetry

            # Case_0 = (tf.equal(self.caseid, 0), lambda: data_flow(0, dataset)) # train
            # Case_1 = (tf.equal(self.caseid, 1), lambda: data_flow(1, dataset)) # valid
            # Case_2 = (tf.equal(self.caseid, 2), lambda: data_flow(2, dataset)) # app

            # Case_List = [Case_0, Case_1, Case_2]
            # self.iterator, inputs_feature, inputs_symmetry = tf.case(pred_fn_pairs = Case_List, default = g)

            # self.iterator = dataset.make_initializable_iterator()

            # iterator_train = dataset_train.make_one_shot_iterator()

            # inputs_feature, inputs_symmetry = self.iterator_train.get_next()

        else:
            inputs_feature = self.inputs_feature
            inputs_symmetry = self.inputs_symmetry

        self.nb = tf.constant(self.neighbourmat, dtype = 'int32', shape=[self.cube_point_num, self.maxdegree], name='nb_relation')
        self.degrees = tf.constant(self.degrees, dtype = 'float32', shape=[self.cube_point_num, 1], name = 'degrees')
        self.cw = tf.constant(self.cot_w, dtype = 'float32', shape=[self.cube_point_num, self.maxdegree, 1], name = 'cw')

        self.feature2point_pre(datainfo)

        self.initial_vae()
        # self.optimizer_vae = []
        variables_vae = []
        print(self.part_num)
        for i in range(self.part_num):
            output_vae = self.build_vae_block(inputs_feature[:,i,:,:], self.hiddendim[0], name = 'vae_block'+self.part_name[i])
            # global_step_all = tf.Variable(0, trainable = False, name = 'golbal_step'+str(i))
            # learning_rate_deform = tf.train.exponential_decay(self.lr, global_step_all, 1000, self.decay_rate, staircase = True)
            # learning_rate_deform = tf.maximum(learning_rate_deform, 0.0000001)
            # self.optimizer_vae.append(tf.train.AdamOptimizer(learning_rate_deform, name='Adam_vae_block'+self.part_name[i]))
            variables_vae.append(slim.get_variables(scope='vae_block'+self.part_name[i]))
            # print(tf.global_variables(scope='vae_block'+self.part_name[i]))
            # [n.name for n in tf.get_default_graph().as_graph_def().node]
            self.post_output_vae(output_vae)

        if self.union:
            # hidden_code = tf.reshape(tf.concat(tf.expand_dims(self.encode, axis = 2), [tf.shape(self.encode)[1],tf.shape(self.encode)[0],tf.shape(self.encode)[2]]))
            hidden_code = tf.transpose(self.encode, perm = [1, 0, 2])
            print(np.shape(hidden_code))

            structure_feature = tf.reshape(tf.concat([hidden_code, inputs_symmetry], axis = 2), [tf.shape(self.encode)[1], -1])
        else:
            # hidden_code = tf.reshape(tf.concat(tf.expand_dims(self.test_encode, axis = 2), [tf.shape(self.test_encode)[1],tf.shape(self.test_encode)[0],tf.shape(self.test_encode)[2]]))
            # print(np.shape(tf.expand_dims(self.test_encode, axis = 2)))
            hidden_code = tf.transpose(self.test_encode, perm = [1, 0, 2])
            print(np.shape(hidden_code))
            structure_feature = tf.reshape(tf.concat([hidden_code, inputs_symmetry], axis = 2), [tf.shape(self.test_encode)[1], -1])

        output_vae = self.build_vae_block_for_symmetry(structure_feature, self.hiddendim[1], name = 'vae_block_structure')
        # global_step_all = tf.Variable(0, trainable = False, name = 'golbal_step_struc')
        # learning_rate_structure = tf.train.exponential_decay(self.lr, global_step_all, 3000, self.decay_rate, staircase = True)
        # learning_rate_structure = tf.maximum(learning_rate_structure, 0.0000001)
        # self.optimizer_vae.append(tf.train.AdamOptimizer(learning_rate_structure, name='vae_block_structure'))
        self.post_output_vae(output_vae)
        variables_vae.append(slim.get_variables(scope="vae_block_structure"))

        # self.traintotalvae_loss = self.generation_loss[0]
        # self.validtotalvae_loss = self.test_loss[0]
        # self.traintotalvae_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(self.inputs - self.train_stack_generate, 2.0), [1,2]))
        # self.validtotalvae_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(self.inputs - self.valid_stack_generate, 2.0), [1,2]))
        train_variables_vae_all = []
        for id in range(self.part_num + 1):
            variables = variables_vae[id]
            train_variables_vae = []
            for v in variables:
                if v in tf.trainable_variables():
                    train_variables_vae.append(v)

            train_variables_vae_all.append(train_variables_vae)

        print(np.expand_dims(tf.trainable_variables(), axis = 0))
        # print(np.expand_dims(tf.global_variables(), axis = 0))
        self.get_total_loss()
        self.train_op = []
        if self.union:
            global_step_all = tf.Variable(0, trainable = False, name = 'global_step_all')
            learning_rate_deform = tf.train.exponential_decay(self.lr, global_step_all, 3000, self.decay_rate, staircase = True)
            learning_rate_deform = tf.maximum(learning_rate_deform, 0.0000001)
            optimizer_vae = tf.train.AdamOptimizer(learning_rate_deform, name='Adam_vae_block_all')
            self.total_trainop = tf.contrib.training.create_train_op(tf.reduce_sum(self.loss_vae), optimizer_vae, global_step = global_step_all,
                                                                variables_to_train=tf.trainable_variables(),
                                                                summarize_gradients=False)
        else:
            for id in range(self.part_num + 1):
                if id == self.part_num:
                    name = 'struc'
                else:
                    name = self.part_name[id]
                global_step_all = tf.Variable(0, trainable = False, name = 'global_step_'+name)
                if id == self.part_num:
                    learning_rate_deform = tf.train.exponential_decay(self.lr, global_step_all, 3000, self.decay_rate, staircase = True)
                else:
                    learning_rate_deform = tf.train.exponential_decay(self.lr, global_step_all, 1000, self.decay_rate, staircase = True)

                learning_rate_deform = tf.maximum(learning_rate_deform, 0.0000001)
                optimizer_vae = tf.train.AdamOptimizer(learning_rate_deform, name='Adam_vae_block'+name)
                trainop = tf.contrib.training.create_train_op(self.loss_vae[id], optimizer_vae, global_step = global_step_all,
                                                                variables_to_train=train_variables_vae_all[id],
                                                                summarize_gradients=False)
                self.train_op.append(trainop)

        # print('all_Optimizer:', optimizer_vae)
        # print('all_global_step:', global_step_all)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.total_loss, global_step = self.global_step_all)

        self.checkpoint_dir_defrom = os.path.join(self.outputdir, 'deform_ckt')
        self.checkpoint_dir_structure = os.path.join(self.outputdir, 'structure_ckt')

        if tensorboard:
            lr_all = tf.summary.scalar('all_learning_rate', learning_rate_structure)
            self.train_summary_var.append([lr_all])
            self.get_var_summary1()
        else:
            self.train_summary = tf.constant(0)
            self.valid_summary = tf.constant(0)

        self.saver = tf.train.Saver(max_to_keep = 10)
        # self.print_netinfo()

    def initial_vae(self):
        self.encode = []
        self.decode = []
        self.encode_std = []
        self.test_encode = []
        self.test_decode = []
        self.test_encode_std = []
        self.embedding_inputs = []
        self.embedding_decode = []
        self.generation_loss = []
        self.distance_norm = []
        self.weights_norm = []
        self.kl_diver = []
        self.l2_loss = []
        self.test_loss = []
        self.testkl_diver = []
        self.train_summary_var = []
        self.valid_summary_var = []
        self.loss_vae = []
        self.op_vae = []
        self.valid_summary_one = []
        self.train_summary_one = []
        self.laplacian_check = []
        self.region = []

    def post_output_vae(self, output_vae):
        #encode, decode, test_encode, test_decode, embedding_inputs, embedding_decode, generation_loss, l2_loss, test_generation_loss, 
        #train_summary_var, valid_summary_var, total_loss
        if self.ae:
            self.encode.append(output_vae[0])
            self.decode.append(output_vae[1])
            # self.encode_std.append(output_vae[0])
            self.test_encode.append(output_vae[2])
            self.test_decode.append(output_vae[3])
            # self.test_encode_std.append(output_vae[0])
            self.embedding_inputs.append(output_vae[4])
            self.embedding_decode.append(output_vae[5])

            self.generation_loss.append(output_vae[6])
            # self.distance_norm.append(output_vae[7])
            self.weights_norm.append(output_vae[7])
            # self.kl_diver.append(output_vae[0])
            self.l2_loss.append(output_vae[8])

            self.test_loss.append(output_vae[9])
            # self.testkl_diver.append(output_vae[0])

            self.train_summary_var.append(output_vae[10])
            self.valid_summary_var.append(output_vae[11])

            self.loss_vae.append(output_vae[12])
            # self.op_vae.append(output_vae[14])
            if tensorboard:
                self.valid_summary_one.append(tf.summary.merge(output_vae[11]))
                self.train_summary_one.append(tf.summary.merge(output_vae[10]))
            else:
                self.valid_summary_one.append(tf.constant(0))
                self.train_summary_one.append(tf.constant(0))
            # self.laplacian_check.append(output_vae[15])
            # self.region.append(output_vae[15])

        else:
            #encode, decode, encode_std, test_encode, test_decode, test_encode_std, embedding_inputs, embedding_decode,
            #generation_loss, kl_diver, l2_loss, test_generation_loss, testkl_diver, train_summary_var, valid_summary_var, total_loss
            print(len(output_vae))
            print(output_vae[16])

            self.encode.append(output_vae[0])
            self.decode.append(output_vae[1])
            self.encode_std.append(output_vae[2])

            self.test_encode.append(output_vae[3])
            self.test_decode.append(output_vae[4])
            self.test_encode_std.append(output_vae[5])

            self.embedding_inputs.append(output_vae[6])
            self.embedding_decode.append(output_vae[7])

            self.generation_loss.append(output_vae[8])
            # self.distance_norm.append(output_vae[9])
            self.weights_norm.append(output_vae[9])
            self.kl_diver.append(output_vae[10])
            self.l2_loss.append(output_vae[11])

            self.test_loss.append(output_vae[12])
            self.testkl_diver.append(output_vae[13])

            self.train_summary_var.append(output_vae[14])
            self.valid_summary_var.append(output_vae[15])

            self.loss_vae.append(output_vae[16])
            # self.op_vae.append(output_vae[18])
            if tensorboard:
                self.valid_summary_one.append(tf.summary.merge(output_vae[15]))
                self.train_summary_one.append(tf.summary.merge(output_vae[14]))
            else:
                self.valid_summary_one.append(tf.constant(0))
                self.train_summary_one.append(tf.constant(0))
            # self.laplacian_check.append(output_vae[19])
            # self.region.append(output_vae[19])

    def get_total_loss(self):

        # self.total_loss = tf.reduce_sum(tf.tuple(self.loss_vae))# + self.traintotalvae_loss # self.generation_loss + self.weights_norm + self.distance_norm + self.kl_diver + self.l2_loss
        # self.total_generation_loss = tf.reduce_sum(tf.tuple(self.generation_loss))
        # self.total_weights_norm = tf.reduce_sum(tf.tuple(self.weights_norm))
        # self.total_distance_norm = tf.reduce_sum(tf.tuple(self.distance_norm))
        # self.total_l2_loss = tf.reduce_sum(tf.tuple(self.l2_loss))

        # self.total_test_loss = tf.reduce_sum(tf.tuple(self.test_loss))# + self.validtotalvae_loss
        # if not self.ae:
        #     self.total_kl_loss = tf.reduce_sum(tf.tuple(self.kl_diver))
        #     self.total_testkl_loss = tf.reduce_sum(tf.tuple(self.testkl_diver))
        self.total_loss = tf.reduce_sum(self.loss_vae)# + self.traintotalvae_loss # self.generation_loss + self.weights_norm + self.distance_norm + self.kl_diver + self.l2_loss
        self.total_generation_loss = tf.reduce_sum(self.generation_loss)
        self.total_weights_norm = tf.reduce_sum(self.weights_norm)
        # self.total_distance_norm = tf.reduce_sum(tf.tuple(self.distance_norm))
        self.total_l2_loss = tf.reduce_sum(self.l2_loss)

        self.total_test_loss = tf.reduce_sum(self.test_loss)# + self.validtotalvae_loss
        if not self.ae:
            self.total_kl_loss = tf.reduce_sum(self.kl_diver)
            self.total_testkl_loss = tf.reduce_sum(self.testkl_diver)

    def get_var_summary(self):
        t_vars = tf.trainable_variables()
        fc_vars = [tf.summary.histogram(var.name, var) for var in t_vars if 'fclayer' in var.name]
        # fc_vars = [tf.summary.scalar('fclayer_' + str(i), var) for var, i in zip(t_vars, range(len(t_vars))) if 'fclayer' in var.name]

        loss_allvae = tf.summary.scalar('traintotalvae_loss', self.traintotalvae_loss)
        loss_all = tf.summary.scalar('total_loss', self.total_loss)
        loss_allgeneration = tf.summary.scalar('total_generation_loss', self.total_generation_loss)
        # loss_allweights = tf.summary.scalar('total_weights_norm', self.total_weights_norm)
        # loss_alldistance = tf.summary.scalar('total_distance_norm', self.total_distance_norm)
        loss_alll2 = tf.summary.scalar('total_l2_loss', self.total_l2_loss)

        # self.train_summary_var.append([loss_allvae, loss_all, loss_allgeneration, loss_allweights, loss_alldistance, loss_alll2]+fc_vars)
        self.train_summary_var.append([loss_allvae, loss_all, loss_allgeneration, loss_alll2]+fc_vars)
        loss_allvaevalid = tf.summary.scalar('validtotalvae_loss', self.validtotalvae_loss)
        loss_allvalid = tf.summary.scalar('vaildtotal_loss', self.total_test_loss)
        self.valid_summary_var.append([loss_allvaevalid, loss_allvalid])

        if not self.ae:
            loss_allkl = tf.summary.scalar('total_kl_loss', self.total_kl_loss)
            self.train_summary_var.append([loss_allkl])
            validloss_allkl = tf.summary.scalar('validtotal_testkl_loss', self.total_testkl_loss)
            self.valid_summary_var.append([validloss_allkl])

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        # trainable_variables = set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        valid_summary_var = list(np.reshape(self.valid_summary_var, [-1]))

        train_summary_var = [x for x in summaries if x not in valid_summary_var]

        self.valid_summary = tf.summary.merge(valid_summary_var)
        self.train_summary = tf.summary.merge(train_summary_var)

        self.merge_summary = tf.summary.merge_all()

        # self.valid_summary = tf.summary.merge(list(np.reshape(self.valid_summary_var, [-1])))

    def get_var_summary1(self):
        t_vars = tf.trainable_variables()
        fc_vars = [tf.summary.histogram(var.name, var) for var in t_vars if 'fclayer' in var.name]
        # fc_vars = [tf.summary.scalar('fclayer_' + str(i), var) for var, i in zip(t_vars, range(len(t_vars))) if 'fclayer' in var.name]

        total_loss = tf.summary.tensor_summary('total_loss', self.loss_vae)
        generation_loss = tf.summary.tensor_summary('generation_loss', self.generation_loss)
        l2_loss = tf.summary.tensor_summary('l2_loss', self.generation_loss)
        test_loss = tf.summary.tensor_summary('test_loss', self.generation_loss)

        self.train_summary_var.append([total_loss, generation_loss, l2_loss]+fc_vars)
        self.valid_summary_var.append([test_loss])

        if not self.ae:
            kl_loss = tf.summary.tensor_summary('kl_loss', self.kl_diver)
            testkl_loss = tf.summary.tensor_summary('testkl_loss', self.testkl_diver)
            self.train_summary_var.append([kl_loss])
            self.valid_summary_var.append([testkl_loss])

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        # trainable_variables = set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        valid_summary_var = list(np.reshape(self.valid_summary_var, [-1]))

        train_summary_var = [x for x in summaries if x not in valid_summary_var]

        self.valid_summary = tf.summary.merge(valid_summary_var)
        self.train_summary = tf.summary.merge(train_summary_var)

        self.merge_summary = tf.summary.merge_all()

        # self.valid_summary = tf.summary.merge(list(np.reshape(self.valid_summary_var, [-1])))

    def print_netinfo(self):
        # summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        # # trainable_variables = set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        # valid_summary_var = list(np.reshape(self.valid_summary_var, [-1]))

        # train_summary_var = [x for x in summaries if x not in valid_summary_var]

        # variables_encoder = slim.get_variables(scope="vae_block0")

        # variables_encoder = slim.get_variables(scope="*/fclayer")
        # variables_encoder = slim.get_variables(scope="*/fclayer_std")
        # print(np.expand_dims(variables_encoder, axis=0))
        # t_vars = tf.trainable_variables()
        # self.d_vars = [var for var in t_vars if 'd_' in var.name]
        # a = set(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='vae_block0'))
        # reg_losses = [var for var in a if 'batch_norm' not in var.name]
        # print(np.shape(self.decode))
        # print(np.shape(self.test_decode))

        # print(np.expand_dims(valid_summary_var, axis=0))

        # print(np.expand_dims(train_summary_var, axis=0))

        # print(np.expand_dims(summaries, axis=0))

        # [print(x.name) for x in a]
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        [print(x) for x in summaries]

        # print(np.expand_dims(tf.trainable_variables(), axis=0))

    def turnresult(self, decode_sym, input_sym, inputs_structure):
        decode = tf.reshape(decode_sym, [tf.shape(decode_sym)[0], self.part_num, self.hiddendim[0]+self.part_dim+2*self.part_num +2+4])
        decode_float = decode[:,:,:(self.hiddendim[0]+3)]
        decode_binary = decode[:,:,(self.hiddendim[0]+3):]
        decode_binary_pred = tf.reshape(decode_binary, [tf.shape(decode_sym)[0], self.part_num, 2*self.part_num+2+4, 2])
        decode_binary = tf.argmax(decode_binary_pred, 3)
        decode_real = tf.concat([decode_float, tf.cast(decode_binary,tf.float32)], axis=2)
        decode_real = tf.reshape(decode_real, [tf.shape(decode_sym)[0], -1])

        # binary loss
        input_sym = tf.reshape(inputs_structure, [tf.shape(inputs_structure)[0], self.part_num, -1])
        # per_instance_seg_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(decode_binary_pred,[-1,2]), labels=tf.reshape(tf.cast(input_sym[:,:,-(2*self.part_num+2):],tf.int32),[-1])), axis=1)) *50
        per_instance_seg_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decode_binary_pred, labels=tf.cast(input_sym[:,:,-(2*self.part_num+2+4):],tf.int32)), axis=[1,2])) #*50
        # hidden and center and symmetry plane loss
        # inputs_structure = tf.reshape(inputs_structure, [tf.shape(inputs_structure)[0], self.part_num, self.hiddendim[0]+self.part_dim])
        inputs_float = input_sym[:,:,:(self.hiddendim[0]+3)]
        hidden_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(decode_float-inputs_float, 2.0), [1,2]))

        generation_loss = hidden_loss+per_instance_seg_loss

        return decode_real, generation_loss

    def build_vae_block_for_symmetry(self, inputs, hiddendim, name = 'vae_block'):
        with tf.variable_scope(name) as scope:
            embedding_inputs = tf.placeholder(tf.float32, [None, hiddendim], name = 'embedding_inputs')

            object_stddev = tf.constant(np.concatenate((np.array([1, 1]).astype('float32'), 1 * np.ones(hiddendim - 2).astype('float32'))))
            self.weight_hist = []
            # train
            if not self.ae:
                encode, encode_std = self.encoder_symm(inputs, training = True)
                encode_gauss = encode + encode_std*embedding_inputs
                decode = self.decoder_symm(encode_gauss, training = True)
            else:
                encode, _ = self.encoder_symm(inputs, training = True)
                decode = self.decoder_symm(encode, training = True)

            # self.test5 = decode

            if self.change_net:
                decode, generation_loss = self.turnresult(decode, self.symmetry_feature, inputs)
            # else:
            #     decode = tf.reshape(decode, [tf.shape(decode)[0], self.part_num, -1])
            #     a = decode[:,:,:self.hiddendim[0]]
            #     b = tf.round(decode[:,:,self.hiddendim[0]:(self.hiddendim[0]+2*self.part_num+1)])
            #     c = decode[:,:,-8:-5]
            #     d = tf.round(decode[:,:,-5:])
            #     decode = tf.reshape(tf.concat([a, b, c, d], axis=2), [tf.shape(decode)[0], -1])

            if not self.ae:
                kl_diver = 0.5 * tf.reduce_sum(tf.square(encode) + tf.square(encode_std / object_stddev) - tf.log(1e-8 + tf.square(encode_std / object_stddev)) - 1, 1)
                kl_diver = self.lambda3*tf.reduce_mean(kl_diver)
            else:
                kl_diver = 0.0
            # test
            test_encode, test_encode_std = self.encoder_symm(inputs, training = False)
            test_decode = self.decoder_symm(test_encode, training = False)
            if self.change_net:
                test_decode, test_generation_loss = self.turnresult(test_decode, self.symmetry_feature, inputs)
            # else:
                # test_decode = tf.reshape(test_decode, [tf.shape(test_decode)[0], self.part_num, -1])
                # a = test_decode[:,:,:self.hiddendim[0]]
                # b = tf.round(test_decode[:,:,self.hiddendim[0]:(self.hiddendim[0]+2*self.part_num+1)])
                # c = test_decode[:,:,-8:-5]
                # d = tf.round(test_decode[:,:,-5:])
                # test_decode = tf.reshape(tf.concat([a, b, c, d], axis=2), [tf.shape(test_decode)[0], -1])


            if not self.ae:
                testkl_diver = 0.5 * tf.reduce_sum(tf.square(test_encode) + tf.square(test_encode_std / object_stddev) - tf.log(1e-8 + tf.square(test_encode_std / object_stddev)) - 1, 1)
                testkl_diver = self.lambda3*tf.reduce_mean(testkl_diver)
            else:
                testkl_diver = 0.0

            embedding_decode = self.decoder_symm(embedding_inputs, training = False)
            if self.change_net:
                embedding_decode, _ = self.turnresult(embedding_decode, self.symmetry_feature, inputs)

            if self.change_net:
                generation_loss = self.lambda0 * generation_loss * 100
                test_generation_loss = self.lambda0 * test_generation_loss * 100
            else:
                generation_loss = self.lambda0 * tf.reduce_mean(tf.reduce_sum(tf.pow(inputs-decode, 2.0), [1])) * 100

                test_generation_loss = self.lambda0 * tf.reduce_mean(tf.reduce_sum(tf.pow(inputs-test_decode, 2.0), [1])) * 100

            reg_losses = set(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope = name))
            reg_losses = [var for var in reg_losses if 'batch_norm' not in var.name]
            # [print(x) for x in reg_losses]
            weight_norm = tf.constant(0.0, dtype=tf.float32)

            l2_loss = self.lambda4 * sum(reg_losses)

            total_loss = generation_loss + kl_diver + l2_loss
            if tensorboard:
                s1 = tf.summary.scalar('generation_loss', generation_loss)
                s3 = tf.summary.scalar('kl_diver', kl_diver)
                s5 = tf.summary.scalar('l2_loss', l2_loss)

                if self.ae:
                    train_summary_var = [s1, s3, s5, self.weight_hist]
                else:
                    s4 = tf.summary.scalar('kl_diver', kl_diver)
                    train_summary_var = [s1, s3, s4, s5, self.weight_hist]

                s1 = tf.summary.scalar('valid_generation_loss', test_generation_loss)

                if self.ae:
                    valid_summary_var = [s1]
                else:
                    s2 = tf.summary.scalar('valid_KL_loss', testkl_diver)
                    valid_summary_var = [s1, s2]
            else:
                train_summary_var = []
                valid_summary_var = []

        if self.ae:
            return encode, decode, test_encode, test_decode, embedding_inputs, embedding_decode, generation_loss, weight_norm, l2_loss, test_generation_loss, train_summary_var, valid_summary_var, total_loss
        else:
            return encode, decode, encode_std, test_encode, test_decode, test_encode_std, embedding_inputs, embedding_decode, generation_loss, weight_norm, kl_diver, l2_loss, test_generation_loss, testkl_diver, train_summary_var, valid_summary_var, total_loss

    def build_vae_block(self, inputs, hiddendim, name = 'vae_block'):
        with tf.variable_scope(name) as scope:
            vae_block_para_en = []
            vae_block_para_de = []
            n_weight_en = []
            e_weight_en = []
            n_weight_de = []
            e_weight_de = []

            embedding_inputs = tf.placeholder(tf.float32, [None, hiddendim], name = 'embedding_inputs')

            object_stddev = tf.constant(np.concatenate((np.array([1, 1]).astype('float32'), 1 * np.ones(hiddendim - 2).astype('float32'))))

            if self.graph_conv:
                for i in range(0, self.layers):
                    if i == self.layers - 1:
                        # n, e = self.get_conv_weights(9, self.finaldim, name = 'convw'+str(i+1))
                        n_encoder = tf.get_variable("convw_encoder"+str(i+1), [self.vertex_dim * self.K, self.finaldim], tf.float32, tf.random_normal_initializer(stddev=0.02))
                        n_decoder = tf.get_variable("convw_decoder"+str(i+1), [self.vertex_dim * self.K, self.finaldim], tf.float32, tf.random_normal_initializer(stddev=0.02))
                    else:
                        # n, e = self.get_conv_weights(9, 9, name = 'convw'+str(i))
                        n_encoder = tf.get_variable("convw_encoder"+str(i), [self.vertex_dim * self.K, self.vertex_dim], tf.float32, tf.random_normal_initializer(stddev=0.02))
                        n_decoder = tf.get_variable("convw_decoder"+str(i), [self.vertex_dim * self.K, self.vertex_dim], tf.float32, tf.random_normal_initializer(stddev=0.02))

                    n_weight_en.append(n_encoder)
                    e_weight_en.append(n_decoder)
                    n_weight_de.append(n_encoder)
                    e_weight_de.append(n_decoder)
                    if tensorboard:
                        tf.summary.histogram('convw_n_decoder'+str(i+1), n_decoder)
                        tf.summary.histogram('convw_n_encoder'+str(i+1), n_encoder)

                    if i == 0:
                        l2_loss = tf.nn.l2_loss(n_encoder) + tf.nn.l2_loss(n_decoder)
                    else:
                        l2_loss += tf.nn.l2_loss(n_encoder) + tf.nn.l2_loss(n_decoder)
            else:

                for i in range(0, self.layers):
                    if i == self.layers - 1:
                        n_en, e_en = self.get_conv_weights(self.vertex_dim, self.finaldim, name = 'en_convw'+str(i+1))
                        n_de, e_de = self.get_conv_weights(self.vertex_dim, self.finaldim, name = 'de_convw'+str(i+1))
                    else:
                        n_en, e_en = self.get_conv_weights(self.vertex_dim, self.vertex_dim, name = 'en_convw'+str(i))
                        n_de, e_de = self.get_conv_weights(self.vertex_dim, self.vertex_dim, name = 'de_convw'+str(i))

                    n_weight_en.append(n_en)
                    e_weight_en.append(e_en)
                    n_weight_de.append(n_de)
                    e_weight_de.append(e_de)
                    if tensorboard:
                        tf.summary.histogram('convw_n_en'+str(i+1), n_en)
                        tf.summary.histogram('convw_e_en'+str(i+1), e_en)
                        tf.summary.histogram('convw_n_de'+str(i+1), n_de)
                        tf.summary.histogram('convw_e_de'+str(i+1), e_de)

                    if i == 0:
                        l2_loss = tf.nn.l2_loss(n_en) + tf.nn.l2_loss(e_en) + tf.nn.l2_loss(n_de) + tf.nn.l2_loss(e_de)
                    else:
                        l2_loss += tf.nn.l2_loss(n_en) + tf.nn.l2_loss(e_en) + tf.nn.l2_loss(n_de) + tf.nn.l2_loss(e_de)


            # fcparams_group = tf.transpose(tf.reshape(fcparams, [self.pointnum, self.finaldim, hiddendim]), perm = [2, 0, 1])
            # selfdot = tf.reduce_sum(tf.pow(fcparams_group, 2.0), axis = 2)

            # a = tf.reshape(fcparams, [hiddendim, self.pointnum, self.finaldim])
            # a = tf.reduce_sum(tf.pow(a, 2.0), axis = 2)
            # Eps = tf.constant(1e-6, tf.float32)
            # Eps = 0.005*tf.ones_like(selfdot)
            # self_region = tf.cast(tf.greater(selfdot, Eps), tf.float32)
            # self_region = selfdot

            # region_sumone = tf.cast(tf.count_nonzero(self_region, 0), dtype = 'float32') - tf.constant(1.0, dtype = 'float32')
            # zeros = tf.zeros_like(region_sumone)
            # region_norm = self.lambda5 * tf.abs(tf.reduce_sum(tf.maximum(region_sumone, zeros)-tf.minimum(region_sumone, zeros)) - self.pointnum)

            vae_block_para_en.append(n_weight_en)
            vae_block_para_en.append(e_weight_en)
            vae_block_para_de.append(n_weight_de)
            vae_block_para_de.append(e_weight_de)

            fcparams_en = tf.get_variable("weights_en", [self.pointnum*self.finaldim, hiddendim], tf.float32, tf.random_normal_initializer(stddev=0.02))
            vae_block_para_en.append(fcparams_en)
            fcparams_de = tf.get_variable("weights_de", [self.pointnum*self.finaldim, hiddendim], tf.float32, tf.random_normal_initializer(stddev=0.02))
            vae_block_para_de.append(fcparams_de)

            if not self.ae:
                fcparams_std_en = tf.get_variable("std_weights_en", [self.pointnum*self.finaldim, hiddendim], tf.float32, tf.random_normal_initializer(stddev=0.02))
                # fcparams_group_std = tf.transpose(tf.reshape(fcparams_std, [self.pointnum, self.finaldim, hiddendim]), perm = [2, 0, 1])
                # selfdot_std = tf.reduce_sum(tf.pow(fcparams_group_std, 2.0), axis = 2)
                vae_block_para_en.append(fcparams_std_en)

            # maxdimension = tf.argmax(selfdot, axis = 1)

            # maxlaplacian = tf.gather(laplacian, maxdimension)

            if self.ae:
                # distance_norm = self.lambda1*tf.reduce_mean(tf.reduce_sum(tf.sqrt(selfdot) * maxlaplacian, 1))
                l2_loss += tf.nn.l2_loss(fcparams_de) + tf.nn.l2_loss(fcparams_en)
            else:
                # distance_norm = self.lambda1 * tf.reduce_mean(tf.reduce_sum((tf.sqrt(selfdot)+tf.sqrt(selfdot_std)) * maxlaplacian, 1))
                l2_loss += tf.nn.l2_loss(fcparams_de) + tf.nn.l2_loss(fcparams_en) + tf.nn.l2_loss(fcparams_std_en)

            l2_loss = self.lambda4 * l2_loss

            if not self.ae:
                # encode, weights_norm, l0 = self.encoder(inputs, vae_block_para, train = True, name = name)
                encode, l0, weights_norm = self.encoder(inputs, vae_block_para_en, train = True, name = name)
                encode_std = self.encoder_std(l0, fcparams_std_en)
                encode_gauss = encode + encode_std*embedding_inputs
                decode = self.decoder(encode_gauss, vae_block_para_de, train = True, name = name)
            else:
                # encode, weights_norm, _ = self.encoder(inputs, vae_block_para, train = True, name = name)
                encode, _ , weights_norm= self.encoder(inputs, vae_block_para_en, train = True, name = name)
                decode = self.decoder(encode, vae_block_para_de, train = True, name = name)

            weights_norm = self.lambda2*weights_norm

            if not self.ae:
                kl_diver = 0.5 * tf.reduce_sum(tf.square(encode) + tf.square(encode_std / object_stddev) - tf.log(1e-8 + tf.square(encode_std / object_stddev)) - 1, 1)
                kl_diver = self.lambda3*tf.reduce_mean(kl_diver)
            else:
                kl_diver = 0.0
            # test
            test_encode, test_l0 = self.encoder(inputs, vae_block_para_en, train = False, name = name)
            test_decode = self.decoder(test_encode, vae_block_para_de, train = False, name = name)

            if not self.ae:
                test_encode_std = self.encoder_std(test_l0, fcparams_std_en)
                testkl_diver = 0.5 * tf.reduce_sum(tf.square(test_encode) + tf.square(test_encode_std / object_stddev) - tf.log(1e-8 + tf.square(test_encode_std / object_stddev)) - 1, 1)
                testkl_diver = self.lambda3*tf.reduce_mean(testkl_diver)
            else:
                testkl_diver = 0.0

            embedding_decode = self.decoder(embedding_inputs, vae_block_para_de, train = False, name = name)
            # total loss
            generation_loss = self.lambda0 * tf.reduce_mean(tf.reduce_sum(tf.pow(inputs-decode, 2.0), [1,2]))

            test_generation_loss = self.lambda0 * tf.reduce_mean(tf.reduce_sum(tf.pow(inputs-test_decode, 2.0), [1,2]))

            # total_loss = generation_loss + weights_norm + distance_norm + kl_diver + l2_loss# + region_norm
            total_loss = generation_loss + kl_diver + l2_loss + weights_norm
            # optimizer_vae = []
            # slr = []

            # if train_inorder:
            #     global_step_vae = tf.Variable(0, trainable = False, name = 'vae_step')

            #     learning_rate = tf.train.exponential_decay(self.lr, global_step_vae, self.decay_step, self.decay_rate, staircase = True)
            #     slr = tf.summary.scalar('learning_rate', learning_rate)

            #     optimizer_vae = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step = global_step_vae)
            if tensorboard:
                s1 = tf.summary.scalar('generation_loss', generation_loss)
                s2 = tf.summary.scalar('weights_norm', weights_norm)
                # s3 = tf.summary.scalar('distance_norm', distance_norm)
                s2 = tf.summary.scalar('l2_loss', l2_loss)
                # s6 = tf.summary.scalar('region_norm', region_norm)

                s1hist = tf.summary.histogram('fcparams', fcparams_en)

                if self.ae:
                    # train_summary_var = [s1, s2, s3, s5, s6, s1hist, slr]
                    train_summary_var = [s1, s2, s1hist]
                else:
                    s4 = tf.summary.scalar('kl_diver', kl_diver)
                    s2hist = tf.summary.histogram('fcparams_std', fcparams_std_en)
                    # train_summary_var = [s1, s2, s3, s4, s5, s6, s1hist, s2hist, slr]
                    train_summary_var = [s1, s2, s4, s1hist, s2hist]

                s1 = tf.summary.scalar('valid_generation_loss', test_generation_loss)

                if self.ae:
                    valid_summary_var = [s1]
                else:
                    s2 = tf.summary.scalar('valid_KL_loss', testkl_diver)
                    valid_summary_var = [s1, s2]

            else:
                train_summary_var = []
                valid_summary_var = []

        # if self.ae:
        #     return encode, decode, test_encode, test_decode, embedding_inputs, embedding_decode, generation_loss, distance_norm, weights_norm, l2_loss, test_generation_loss, train_summary_var, valid_summary_var, total_loss, optimizer_vae, self_region
        # else:
        #     return encode, decode, encode_std, test_encode, test_decode, test_encode_std, embedding_inputs, embedding_decode, generation_loss, distance_norm, weights_norm, kl_diver, l2_loss, test_generation_loss, testkl_diver, train_summary_var, valid_summary_var, total_loss, optimizer_vae, self_region
        if self.ae:
            return encode, decode, test_encode, test_decode, embedding_inputs, embedding_decode, generation_loss, weights_norm, l2_loss, test_generation_loss, train_summary_var, valid_summary_var, total_loss
        else:
            return encode, decode, encode_std, test_encode, test_decode, test_encode_std, embedding_inputs, embedding_decode, generation_loss, weights_norm, kl_diver, l2_loss, test_generation_loss, testkl_diver, train_summary_var, valid_summary_var, total_loss

    def build_fcvae_block(self, inputs, hiddendim, name = 'fcvae_block'):
        with tf.variable_scope(name) as scope:
            self.weight_hist = []

            embedding_inputs = tf.placeholder(tf.float32, [None, hiddendim], name = 'embedding_inputs')

            object_stddev = tf.constant(np.concatenate((np.array([1, 1]).astype('float32'), 1 * np.ones(hiddendim - 2).astype('float32'))))

            if not self.ae:
                # encode, weights_norm, l0 = self.encoder(inputs, vae_block_para, train = True, name = name)
                encode, encode_std = self.encoder_fc(inputs, training = True)
                encode_gauss = encode + encode_std*embedding_inputs
                decode = self.decoder_fc(encode_gauss, training = True)
            else:
                # encode, weights_norm, _ = self.encoder(inputs, vae_block_para, train = True, name = name)
                encode, _ = self.encoder_fc(inputs, training = True)
                decode = self.decoder_fc(encode, training = True)

            if not self.ae:
                kl_diver = 0.5 * tf.reduce_sum(tf.square(encode) + tf.square(encode_std / object_stddev) - tf.log(1e-8 + tf.square(encode_std / object_stddev)) - 1, 1)
                kl_diver = self.lambda3*tf.reduce_mean(kl_diver)
            else:
                kl_diver = 0.0
            # test
            test_encode, test_encode_std = self.encoder_fc(inputs, training = False)
            test_decode = self.decoder_fc(test_encode, training = False)

            if not self.ae:
                testkl_diver = 0.5 * tf.reduce_sum(tf.square(test_encode) + tf.square(test_encode_std / object_stddev) - tf.log(1e-8 + tf.square(test_encode_std / object_stddev)) - 1, 1)
                testkl_diver = self.lambda3*tf.reduce_mean(testkl_diver)
            else:
                testkl_diver = 0.0

            embedding_decode = self.decoder_fc(embedding_inputs, training = False)
            # total loss
            generation_loss = self.lambda0 * tf.reduce_mean(tf.reduce_sum(tf.pow(inputs-decode, 2.0), [1,2]))

            test_generation_loss = self.lambda0 * tf.reduce_mean(tf.reduce_sum(tf.pow(inputs-test_decode, 2.0), [1,2]))

            reg_losses = set(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope = name))
            reg_losses = [var for var in reg_losses if 'batch_norm' not in var.name]
            l2_loss = self.lambda4 * sum(reg_losses)

            total_loss = generation_loss + kl_diver + l2_loss# + region_norm
            if tensorboard:
                s1 = tf.summary.scalar('generation_loss', generation_loss)
                s3 = tf.summary.scalar('kl_diver', kl_diver)
                s5 = tf.summary.scalar('l2_loss', l2_loss)

                if self.ae:
                    train_summary_var = [s1, s3, s5, self.weight_hist]
                else:
                    s4 = tf.summary.scalar('kl_diver', kl_diver)
                    train_summary_var = [s1, s3, s4, s5, self.weight_hist]

                s1 = tf.summary.scalar('valid_generation_loss', test_generation_loss)

                if self.ae:
                    valid_summary_var = [s1]
                else:
                    s2 = tf.summary.scalar('valid_KL_loss', testkl_diver)
                    valid_summary_var = [s1, s2]
            else:
                train_summary_var = []
                valid_summary_var = []

        if self.ae:
            return encode, decode, test_encode, test_decode, embedding_inputs, embedding_decode, generation_loss, l2_loss, test_generation_loss, train_summary_var, valid_summary_var, total_loss
        else:
            return encode, decode, encode_std, test_encode, test_decode, test_encode_std, embedding_inputs, embedding_decode, generation_loss, kl_diver, l2_loss, test_generation_loss, testkl_diver, train_summary_var, valid_summary_var, total_loss

    def change_laplacian(self, distance, dmin, dmax = 1e6):
        zero = tf.zeros([tf.shape(distance)[0], tf.shape(distance)[1]], tf.float32)
        one = tf.ones([tf.shape(distance)[0], tf.shape(distance)[1]], tf.float32)

        if dmax == 1e6:
            print(dmax)
            distance = tf.where(distance > dmin, one, zero)
        else:
            print(dmax)
            distance = (distance - dmin) / (dmax - dmin)
            distance = tf.where(distance > one, one, distance)
            distance = tf.where(distance < zero, zero, distance)

        return distance

    def load(self, sess, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
        saver = self.saver

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        # import the inspect_checkpoint library
        from tensorflow.python.tools import inspect_checkpoint as chkp

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            # print all tensors in checkpoint file
            chkp.print_tensors_in_checkpoint_file(os.path.join(checkpoint_dir, ckpt_name), tensor_name='', all_tensors=False, all_tensor_names=True)
            # chkp._count_total_params

            if not ckpt_name.find('best') == -1:
                counter = 0
            else:
                counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))

            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0  # model = convMESH()

    def leaky_relu(self, input_, alpha = 0.1):
        return tf.nn.leaky_relu(input_)
        # return tf.maximum(input_, alpha*input_)

    def softplusplus(self,input_, alpha=0.2):
        return tf.log(1.0+tf.exp(input_*(1.0-alpha)))+alpha*input_-tf.log(2.0)

    def batch_norm_wrapper(self, inputs, name = 'batch_norm',is_training = False, decay = 0.9, epsilon = 1e-5):
        with tf.variable_scope(name) as scope:
            if is_training == True:
                scale = tf.get_variable('scale', dtype=tf.float32, trainable=True, initializer=tf.ones([inputs.get_shape()[-1]],dtype=tf.float32))
                beta = tf.get_variable('beta', dtype=tf.float32, trainable=True, initializer=tf.zeros([inputs.get_shape()[-1]],dtype=tf.float32))
                pop_mean = tf.get_variable('overallmean',  dtype=tf.float32,trainable=False, initializer=tf.zeros([inputs.get_shape()[-1]],dtype=tf.float32))
                pop_var = tf.get_variable('overallvar',  dtype=tf.float32, trainable=False, initializer=tf.ones([inputs.get_shape()[-1]],dtype=tf.float32))
            else:
                scope.reuse_variables()
                scale = tf.get_variable('scale', dtype=tf.float32, trainable=True)
                beta = tf.get_variable('beta', dtype=tf.float32, trainable=True)
                pop_mean = tf.get_variable('overallmean', dtype=tf.float32, trainable=False)
                pop_var = tf.get_variable('overallvar', dtype=tf.float32, trainable=False)

            if is_training:
                axis = list(range(len(inputs.get_shape()) - 1))
                batch_mean, batch_var = tf.nn.moments(inputs,axis)
                train_mean = tf.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(pop_var,pop_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, scale, epsilon)
            else:
                return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

    def convlayer(self, input_feature, input_dim, output_dim, nb_weights, edge_weights, name = 'meshconv', training = True, special_activation = False, no_activation = False, bn = True):
        with tf.variable_scope(name) as scope:

            padding_feature = tf.zeros([tf.shape(input_feature)[0], 1, input_dim], tf.float32)

            padded_input = tf.concat([padding_feature, input_feature], 1)

            #def compute_nb_feature(input_f):
                #return tf.gather(input_f, self.nb)

            #total_nb_feature = tf.map_fn(compute_nb_feature, padded_input)
            total_nb_feature = tf.gather(padded_input, self.nb, axis = 1)
            mean_nb_feature = tf.reduce_sum(total_nb_feature, axis = 2)/self.degrees
            # total_nb_feature = tf.gather(padded_input, self.nb, axis = 1) * self.cot_w
            # mean_nb_feature = tf.reduce_sum(total_nb_feature, axis = 2)

            nb_feature = tf.tensordot(mean_nb_feature, nb_weights, [[2],[0]])

            edge_bias = tf.get_variable("edge_bias", [output_dim], tf.float32, initializer=tf.constant_initializer(0.0))
            edge_feature = tf.tensordot(input_feature, edge_weights, [[2],[0]]) + edge_bias

            total_feature = edge_feature + nb_feature

            if bn == False:
                fb = total_feature
            else:
                fb = self.batch_norm_wrapper(total_feature, is_training = training)

            if no_activation == True:
                fa = fb
            elif special_activation == False:
                fa = self.leaky_relu(fb)
            else:
                fa = tf.nn.tanh(fb)
                print('tanh')

            return fa

    def graph_conv2(self, x, L, Fout, W, K, name='graph_conv', training=True, special_activation=False, no_activation=False, bn=True):
        with tf.variable_scope(name) as scope:

            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
            N, M, Fin = x.get_shape()
            L = L.tocoo()
            indices = np.column_stack((L.row, L.col))
            L = tf.SparseTensor(indices, L.data, L.shape)
            L = tf.sparse_reorder(L)
            # Transform to Chebyshev basis
            x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
            x0 = tf.reshape(x0, [M, -1])  # M x Fin*N
            x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N

            def concat(x, x_):
                x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
                return tf.concat([x, x_], axis=0)  # K x M x Fin*N
            if K > 1:
                x1 = tf.sparse_tensor_dense_matmul(L, x0)
                x = concat(x, x1)
            for k in range(2, K):
                x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
                x = concat(x, x2)
                x0, x1 = x1, x2
            x = tf.reshape(x, [K, M, Fin, -1])  # K x M x Fin x N
            x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
            x = tf.reshape(x, [-1, Fin * K])  # N*M x Fin*K
            # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
            x = tf.matmul(x, W)  # N*M x Fout

            x = tf.reshape(x, [-1, M, Fout])  # N x M x Fout

            if not bn:
                fb = x
            else:
                fb = self.batch_norm_wrapper(x, is_training=training)

            if no_activation:
                fa = fb
            elif not special_activation:
                fa = self.leaky_relu(fb)
            else:
                fa = tf.nn.tanh(fb)

            return fa

    def linear(self, input_, input_size, output_size, name='Linear', training = True, special_activation = False, no_activation = False, bn = True, stddev=0.02, bias_start=0.0):
        with tf.variable_scope(name) as scope:

            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
            matrix = tf.get_variable("weights", [input_size, output_size], tf.float32, tf.random_normal_initializer(stddev=stddev)) #tf.contrib.layers.variance_scaling_initializer() tf.contrib.layers.xavier_initializer()
            bias = tf.get_variable("bias", [output_size], tf.float32, initializer=tf.constant_initializer(bias_start))

            if training:
                if tensorboard:
                    matrixhist = tf.summary.histogram('fc_weights', matrix)
                    biashist = tf.summary.histogram('fc_bias', bias)
                    self.weight_hist.append(matrixhist)
                    self.weight_hist.append(biashist)

            output = tf.matmul(input_, matrix) + bias

            if bn == False:
                fb = output
            else:
                fb = self.batch_norm_wrapper(output, is_training = training)

            if no_activation == True:
                fa = fb
                print('dont use activate function')
            elif special_activation == False:
                if self.activate == 'elu':
                    fa = tf.nn.elu(fb)
                elif self.activate == 'spp':
                    fa = self.softplusplus(fb)
                else:
                    fa = self.leaky_relu(fb)
            else:
                fa = tf.nn.tanh(fb)
                print('tanh')

        return fa

    def linear_nobias(self, input_, input_size, output_size, name='Linear', training = True, special_activation = False, no_activation = False, bn = True, stddev=0.02, bias_start=0.0):
        with tf.variable_scope(name) as scope:

            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
            matrix = tf.get_variable("weights", [input_size, output_size], tf.float32, tf.random_normal_initializer(stddev=stddev)) #tf.contrib.layers.variance_scaling_initializer() tf.contrib.layers.xavier_initializer()
            # bias = tf.get_variable("bias", [output_size], tf.float32, initializer=tf.constant_initializer(bias_start))

            if training:
                if tensorboard:
                    matrixhist = tf.summary.histogram('fc_weights', matrix)
                # biashist = tf.summary.histogram('fc_bias', bias)
                    self.weight_hist.append(matrixhist)
                # self.weight_hist.append(biashist)

            output = tf.matmul(input_, matrix)# + bias

            if bn == False:
                fb = output
            else:
                fb = self.batch_norm_wrapper(output, is_training = training)

            if no_activation == True:
                fa = fb
            elif special_activation == False:
                if self.activate == 'elu':
                    fa = tf.nn.elu(fb)
                else:
                    fa = self.leaky_relu(fb)
            else:
                fa = tf.nn.tanh(fb)
                print('tanh')

        return fa

    def vae_encode(self, input_, input_size, output_size, train = True):
        with tf.variable_scope("encoder") as scope:
            if train == False:
                scope.reuse_variables()
            # input_ = self.leaky_relu(input_)

            mean = self.linear(input_, input_size, output_size, name='fclayers_mean', training = train, no_activation = True, bn = False)
            std = tf.nn.softplus(self.linear(input_, input_size, output_size, name='fclayers_std', training = train, no_activation = True, bn = False))

        return mean, std

    def vae_decode(self, input_, input_size, output_size, train = True):
        with tf.variable_scope("decoder") as scope:
            if train == False:
                scope.reuse_variables()

            output = self.linear(input_, input_size, output_size, name='fclayers', training = train, no_activation = True, bn = False)

        return output

    def get_conv_weights(self, input_dim, output_dim, name = 'convweight'):
        with tf.variable_scope(name) as scope:
            n = tf.get_variable("nb_weights", [input_dim, output_dim], tf.float32, tf.random_normal_initializer(stddev=0.02))
            e = tf.get_variable("edge_weights", [input_dim, output_dim], tf.float32, tf.random_normal_initializer(stddev=0.02))

            return n, e

    def test_model(self, geodesic_weight):

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            start = time.time()

            test = self.conv1.eval({self.inputs: feature})
            end = time.time()

            print('%fs'%(end-start))
            return test

    def encoder(self, input_feature, para, train = True, name = "stack"):
        with tf.variable_scope("encoder") as scope:
            if train == False:
                scope.reuse_variables()

            prev = input_feature

            for i in range(0, self.layers):
                if i == self.layers - 1:
                    if self.layers == 1:
                        if self.graph_conv:
                            conv = self.graph_conv2(prev, self.L1_, self.finaldim, para[0][i], self.K, name='graph_conv'+str(i+1), special_activation = True, training = train, bn = False)
                        else:
                            conv = self.convlayer(prev, self.vertex_dim, self.finaldim, para[0][i], para[1][i], name = 'conv'+str(i+1), special_activation = True, training = train, bn = False)
                    else:
                        if self.graph_conv:
                            conv = self.graph_conv2(prev, self.L1_, self.finaldim, para[0][i], self.K, name='graph_conv'+str(i+1), no_activation = True, training = train, bn = False)
                        else:
                            conv = self.convlayer(prev, self.vertex_dim, self.finaldim, para[0][i], para[1][i], name = 'conv'+str(i+1), no_activation = True, training = train, bn = False)
                else:
                    if self.graph_conv:
                        prev = self.graph_conv2(prev, self.L1_, self.finaldim, para[0][i], self.K, name='graph_conv'+str(i+1), special_activation = True, training = train, bn = True)
                    else:
                        prev = self.convlayer(prev, self.vertex_dim, self.vertex_dim, para[0][i], para[1][i], name = 'conv'+str(i+1), special_activation = True, training = train, bn = False)

            l0 = tf.reshape(conv, [tf.shape(conv)[0], self.pointnum * self.finaldim])

            l1 = tf.matmul(l0, para[2])

            if train == True:
                weights_maximum = tf.reduce_max(tf.abs(l1), 0) - 0.95
                zeros = tf.zeros_like(weights_maximum)
                weights_norm = tf.reduce_mean(tf.maximum(weights_maximum, zeros))
                return l1, l0, weights_norm
                # return l1, l0
            else:
                return l1, l0

    def encoder_std(self, l0, para_std):
        # return tf.nn.softplus(tf.matmul(l0, para_std))
        return 2 * tf.nn.sigmoid(tf.matmul(l0, para_std))
        # return tf.sqrt(tf.nn.softsign(tf.matmul(l0, para_std))+1)

    def decoder(self, latent_tensor, para, train = True, name = "stack"):
        with tf.variable_scope("decoder") as scope:
            if train == False:
                scope.reuse_variables()

            l1 = tf.matmul(latent_tensor, tf.transpose(para[2]))

            l2 = tf.reshape(l1, [tf.shape(l1)[0], self.pointnum, self.finaldim])

            prev = l2

            for i in range(0, self.layers):
                if i == 0:
                    if self.graph_conv:
                        if i == self.layers-1:
                            bn=False
                        else:
                            bn=True
                        conv = self.graph_conv2(prev, self.L1_, self.finaldim, para[1][self.layers-1], self.K, name='graph_conv'+str(i+1), special_activation = True, training = train, bn = bn)
                    else:
                        conv = self.convlayer(prev, self.finaldim, self.vertex_dim, tf.transpose(para[0][self.layers-1]), tf.transpose(para[1][self.layers-1]), name = 'conv'+str(i+1), special_activation = True, training = train, bn = False)
                else:
                    if self.graph_conv:
                        if i == self.layers-1:
                            bn=False
                        else:
                            bn =True
                        conv = self.graph_conv2(prev, self.L1_, self.finaldim, para[1][self.layers-1-i], self.K, name='graph_conv'+str(i+1), special_activation = True, training = train, bn = bn)
                    else:
                        conv = self.convlayer(prev, self.vertex_dim, self.vertex_dim, tf.transpose(para[0][self.layers-1-i]), tf.transpose(para[1][self.layers-1-i]), name = 'conv'+str(i+1), special_activation = True, training = train, bn = False)

                prev = conv

        return conv

    def encoder_fc(self, input_mesh, training = True, keep_prob = 0.95):
        with tf.variable_scope("encoder_fc") as scope:
            if(training == False):
                scope.reuse_variables()

            input_mesh = tf.reshape(input_mesh, [tf.shape(input_mesh)[0], self.pointnum*self.vertex_dim])

            h1 = self.linear(input_mesh, self.pointnum*self.vertex_dim, 2048, name = 'fc_1', training = training, special_activation = False, bn = True)
            h1 = tf.nn.dropout(h1, keep_prob = keep_prob)
            # h1bn = batch_norm_wrapper(h1, name = 'fc_1bn',is_training = training, decay = 0.9)
            # h1a = leaky_relu(h1bn)

            #self.r2 = tf.nn.l2_loss(weights)

            h2 = self.linear(h1, 2048, 1024, name = 'fc_2', training = training, special_activation = False, bn = True)
            h2 = tf.nn.dropout(h2, keep_prob = keep_prob)
            # h2bn = batch_norm_wrapper(h2, name = 'fc_2bn',is_training = training, decay = 0.9)
            # h2a = leaky_relu(h2bn)

            #self.r2 += tf.nn.l2_loss(weights)
            # h3=h2
            h3 = self.linear(h2, 1024, 512, name = 'fc_3', training = training, special_activation = False, bn = True)
            h3 = tf.nn.dropout(h3, keep_prob = keep_prob)
            # h3bn = batch_norm_wrapper(h3, name = 'fc_33bn',is_training = training, decay = 0.9)
            # h3a = leaky_relu(h3bn)

            #self.r2 += tf.nn.l2_loss(weights)

            '''
            h4, weights = linear(h3a, 1024, 512, 'h4')
            h4bn = batch_norm_wrapper(h4, name = 'h4bn',is_training = training, decay = 0.9)
            h4a = leaky_relu(h4bn)
            '''

            #self.r2 += tf.nn.l2_loss(weights)

            mean = self.linear(h3, 512, self.hiddendim[0], name = 'mean', training = training, no_activation = True, bn = False)
            #self.r2 += tf.nn.l2_loss(weights)
            stddev = self.linear(h3, 512, self.hiddendim[0], name = 'stddev', training = training, no_activation = True, bn = False)
            #self.r2 += tf.nn.l2_loss(weights)
            stddev = 2 *tf.sigmoid(stddev)

        return mean, stddev

    def decoder_fc(self, z, training = True, keep_prob = 0.95):
        with tf.variable_scope("decoder_fc") as scope:
            if(training == False):
                scope.reuse_variables()

            h1 = self.linear(z, self.hiddendim[0], 512, name = 'fc_1', training = training, special_activation = False, bn = True)
            h1 = tf.nn.dropout(h1, keep_prob = keep_prob)
            # h1a = leaky_relu(batch_norm_wrapper(h1, name = 'fc_1bn', is_training = training))
            #self.r2 += tf.nn.l2_loss(weights)

            h2 = self.linear(h1, 512, 1024, name = 'fc_2', training = training, special_activation = False, bn = True)
            h2 = tf.nn.dropout(h2, keep_prob = keep_prob)
            # h2a = leaky_relu(batch_norm_wrapper(h2, name = 'fc_1bn', is_training = training))
            #self.r2 += tf.nn.l2_loss(weights)
            # h3=h2
            h3 = self.linear(h2, 1024, 2048, name = 'fc_3', training = training, special_activation = False, bn = True)
            h3 = tf.nn.dropout(h3, keep_prob = keep_prob)
            # h3a = leaky_relu(batch_norm_wrapper(h3, name = 'fc_1bn', is_training = training))
            #self.r2 += tf.nn.l2_loss(weights)

            #h4 = linear(h3a, 2048, 4096, 'h4')
            #h4a = leaky_relu(batch_norm_wrapper(h4, name = 'h4bn', is_training = training))
            #self.r2 += tf.nn.l2_loss(weights)

            output = self.linear(h3,2048, self.pointnum*self.vertex_dim, name = 'fc_4', training = training, no_activation = True, bn = False)
            #self.r2 += tf.nn.l2_loss(weights)

            output = tf.nn.tanh(output)

            output = tf.reshape(output, [tf.shape(output)[0], self.pointnum, self.vertex_dim])

        return output

    def encoder_symm(self, input_mesh, training = True, keep_prob = 0.95):
        with tf.variable_scope("encoder_symm") as scope:
            if(training == False):
                keep_prob = 1.0
                scope.reuse_variables()

            if self.union:
                bn = True
            else:
                bn = False

            # input_mesh = tf.reshape(input_mesh, [tf.shape(input_mesh)[0], self.pointnum*self.vertex_dim])
            # if training:
                # self.test1 = input_mesh
            h1 = self.linear(input_mesh, self.part_num*(self.hiddendim[0]+self.part_dim), 2048, name = 'fc_1', training = training, special_activation = False, bn = bn)
            # h1 = tf.nn.dropout(h1, rate = 1 - keep_prob)
            h1 = tf.nn.dropout(h1, keep_prob = keep_prob)

            # if training:
                # self.test2 = h1


            # h1bn = batch_norm_wrapper(h1, name = 'fc_1bn',is_training = training, decay = 0.9)
            # h1a = leaky_relu(h1bn)

            #self.r2 = tf.nn.l2_loss(weights)

            h2 = self.linear(h1, 2048, 512, name = 'fc_2', training = training, special_activation = False, bn = bn)
            # h2 = tf.nn.dropout(h2, rate = 1 - keep_prob)
            h2 = tf.nn.dropout(h2, keep_prob = keep_prob)
            # if training:
                # self.test3 = h2
            # if training:
                # self.test2 = h2
            # self.test2 = h2
            # h2bn = batch_norm_wrapper(h2, name = 'fc_2bn',is_training = training, decay = 0.9)
            # h2a = leaky_relu(h2bn)

            #self.r2 += tf.nn.l2_loss(weights)

            h3 = self.linear(h2, 512, 128, name = 'fc_3', training = training, special_activation = False, bn = bn)
            # h3 = tf.nn.dropout(h3, rate = 1 - keep_prob)
            h3 = tf.nn.dropout(h3, keep_prob = keep_prob)
            # if training:
                # self.test4 = h3
            # if training:
                # self.test3 = h3
            # self.test3 = h3

            # h3bn = batch_norm_wrapper(h3, name = 'fc_33bn',is_training = training, decay = 0.9)
            # h3a = leaky_relu(h3bn)

            #self.r2 += tf.nn.l2_loss(weights)

            '''
            h4, weights = linear(h3a, 1024, 512, 'h4')
            h4bn = batch_norm_wrapper(h4, name = 'h4bn',is_training = training, decay = 0.9)
            h4a = leaky_relu(h4bn)
            '''

            #self.r2 += tf.nn.l2_loss(weights)

            mean = self.linear(h3, 128, self.hiddendim[1], name = 'mean', training = training, no_activation = True, bn = False)
            #self.h1=mean
            #self.r2 += tf.nn.l2_loss(weights)
            stddev = self.linear(h3, 128, self.hiddendim[1], name = 'stddev', training = training, no_activation = True, bn = False)
            #self.r2 += tf.nn.l2_loss(weights)
            # stddev = 2 * tf.nn.sigmoid(stddev)
            # if training:
                # self.test5 = mean
                # self.test6 = stddev
            stddev = tf.sqrt(tf.nn.softsign(stddev)+1.0)

        return mean, stddev

    def decoder_symm(self, z, training = True, keep_prob = 0.95):
        with tf.variable_scope("decoder_symm") as scope:
            if(training == False):
                keep_prob = 1.0
                scope.reuse_variables()

            if self.union:
                bn = True
            else:
                bn = False

            h1 = self.linear(z, self.hiddendim[1], 128, name = 'fc_1', training = training, special_activation = False, bn = bn)
            # h1 = tf.nn.dropout(h1, rate = 1 - keep_prob)
            h1 = tf.nn.dropout(h1, keep_prob = keep_prob)
            # if training:
                # self.test2 = h1
            # h1a = leaky_relu(batch_norm_wrapper(h1, name = 'fc_1bn', is_training = training))
            #self.r2 += tf.nn.l2_loss(weights)

            h2 = self.linear(h1, 128, 512, name = 'fc_2', training = training, special_activation = False, bn = bn)
            # h2 = tf.nn.dropout(h2, rate = 1 - keep_prob)
            h2 = tf.nn.dropout(h2, keep_prob = keep_prob)
            # if training:
                # self.test2 = h2
            # h2a = leaky_relu(batch_norm_wrapper(h2, name = 'fc_1bn', is_training = training))
            #self.r2 += tf.nn.l2_loss(weights)

            h3 = self.linear(h2, 512, 2048, name = 'fc_3', training = training, special_activation = False, bn = bn)
            # h3 = tf.nn.dropout(h3, rate = 1 - keep_prob)
            h3 = tf.nn.dropout(h3, keep_prob = keep_prob)
            # if training:
                # self.test3 = h3
            # h3a = leaky_relu(batch_norm_wrapper(h3, name = 'fc_1bn', is_training = training))
            #self.r2 += tf.nn.l2_loss(weights)

            #h4 = linear(h3a, 2048, 4096, 'h4')
            #h4a = leaky_relu(batch_norm_wrapper(h4, name = 'h4bn', is_training = training))
            #self.r2 += tf.nn.l2_loss(weights)

            if self.change_net:
                output = self.linear(h3, 2048, self.part_num*(self.hiddendim[0]+self.part_dim+2*self.part_num +2+4), name = 'fc_4', training = training, no_activation = True, bn = False)
            else:
                output = self.linear(h3, 2048, self.part_num*(self.hiddendim[0]+self.part_dim), name = 'fc_4', training = training, no_activation = True, bn = False)
            #self.r2 += tf.nn.l2_loss(weights)
            # if training:
                # self.test4 = output
            # output = tf.nn.tanh(output)
            # output = tf.reshape(output, [tf.shape(output)[0], self.pointnum, self.vertex_dim])

        return output

    def train(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()

            for epoch in range(0, self.maxepoch+1):

                start = time.time()
                random = gaussian(len(self.feature), self.hiddendim)
                sess.run([self.optimizer], feed_dict = {self.inputs: self.feature, self.embedding_inputs: random})

                cost_generation, cost_norm, cost_weights, cost_kl = sess.run([self.test_loss, self.distance_norm, self.weights_norm, self.kl_diver], {self.inputs:self.feature, self.embedding_inputs: random})
                print("Epoch: [%5d|total] generation_loss: %.8f  norm_loss: %.8f weight_loss: %.8f kl_loss: %.8f" % (epoch, cost_generation, cost_norm, cost_weights, cost_kl))

                if (epoch+1) % 5000 == 0 or epoch == self.maxepoch:
                    self.saver.save(sess, './convmesh-model', global_step = epoch+1)

                end = time.time()

                print('time: %fs'%(end-start))

    def train_total_vae(self):# trian with split the dataset to test the generalization error
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        rng = np.random.RandomState(23456)
        batch_size = self.batch_size
        if timeline_use: # use the timeline to analyze the efficency of the program
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            many_runs_timeline = TimeLiner()
        else:
            options = None
            run_metadata = None

        with tf.Session(config = self.config) as sess:

            could_load, checkpoint_counter = self.load(sess, self.outputdir)
            if tensorboard:
                summary_writer = tf.summary.FileWriter(self.outputdir+'/logs', sess.graph)

            tf.global_variables_initializer().run()
            train_id, valid_id = spilt_dataset(len(self.feature), self.train_percent, self.iddat_name)

            inf = float('inf')

            for epoch in range(checkpoint_counter, self.maxepoch+1):

                rng.shuffle(train_id)
                rng.shuffle(valid_id)
                # train_feature = self.feature[train_id]
                valid_feature = self.feature[valid_id]

                # start = time.time()
                # train
                for bidx in xrange(0, len(train_id)//batch_size + 1):

                    train_feature = [self.feature[i] for i in train_id[bidx*batch_size:min(len(train_id), bidx*batch_size+batch_size)]]
                    if len(train_feature) == 0:
                        continue

                    if self.ae:
                        random = np.zeros((len(train_feature),200)).astype('float32')
                        dictbase = {self.inputs: train_feature}
                        # dictrand = {x: random[:, 0: np.shape(x)[1]] for x in self.embedding_inputs}
                        dictrand = {x: np.tile(random[:, 0: np.shape(x)[1]], (i, 1)) for x, i in zip(self.embedding_inputs, [1]+list(self.hiddendim))}
                        feed_dict = merge_two_dicts(dictbase, dictrand)
                        # _, cost_allvae, cost_generation, cost_distance, cost_weights, cost_l2, train_summary = sess.run([self.optimizer, self.traintotalvae_loss, self.total_generation_loss, self.total_distance_norm, self.total_weights_norm, self.total_l2_loss, self.train_summary], feed_dict = feed_dict, options=options, run_metadata=run_metadata)
                        _, cost_generation, cost_l2, train_summary = sess.run([self.optimizer, self.total_generation_loss, self.total_l2_loss, self.train_summary], feed_dict = feed_dict, options=options, run_metadata=run_metadata)
                        # a = sess.run([self.l0], feed_dict = feed_dict)[0]
                        # print(a)
                        #sio.savemat('laplacian_check.mat', {'region':f})

                        printout(self.flog,"Epoch: [%5d|total] generation_loss: %.8f l2_loss: %.8f" % (epoch, cost_generation, cost_l2), epoch)

                        if not len(valid_feature) == 0 and bidx == len(train_id)//batch_size:
                            cost_generation_valid, valid_summary = sess.run([self.total_test_loss, self.valid_summary], feed_dict = {self.inputs:valid_feature})

                            printout(self.flog,"Epoch: [%5d|total] valid_loss: %.8f" % (epoch, cost_generation_valid),epoch)

                    else:

                        # random = gaussian(len(train_feature), self.hiddendim[-1])
                        dictbase = {self.inputs: train_feature}
                        # dictrand = {x: gaussian(len(train_feature), np.shape(x)[1]) for x in self.embedding_inputs}
                        # dictrand = {x: np.tile(gaussian(len(train_feature), np.shape(x)[1]), (i, 1)) for x, i in zip(self.embedding_inputs, [1]+list(self.hiddendim))}
                        dictrand = {self.embedding_inputs[0]: gaussian(len(train_feature), self.hiddendim[0])}
                        feed_dict = merge_two_dicts(dictbase, dictrand)
                        # _, cost_allvae, cost_generation, cost_kl, cost_l2, train_summary = sess.run([self.optimizer, self.traintotalvae_loss, self.total_generation_loss, self.total_kl_loss, self.total_l2_loss, self.train_summary], feed_dict = feed_dict)
                        _, cost_generation, cost_kl, cost_l2, train_summary = sess.run([self.optimizer, self.total_generation_loss, self.total_kl_loss, self.total_l2_loss, self.train_summary], feed_dict = feed_dict)
                        # a = sess.run([self.decode[0]], feed_dict = feed_dict)[0]
                        # print(cost_generation)

                        printout(self.flog,"Epoch: [%5d|total] generation_loss: %.8f kl_loss: %.8f l2_loss: %.8f" % (epoch, cost_generation, cost_kl, cost_l2), epoch)

                        #valid
                        if not len(valid_feature) == 0 and bidx == len(train_id)//batch_size:
                            random = np.zeros((len(valid_feature), self.hiddendim[-1])).astype('float32')
                            dictbase = {self.inputs: valid_feature}
                            dictrand = {x: np.tile(random[:, 0: np.shape(x)[1]], (i, 1)) for x, i in zip(self.embedding_inputs, [1]+list(self.hiddendim))}
                            feed_dict = merge_two_dicts(dictbase, dictrand)
                            cost_generation_valid, valid_summary = sess.run([self.total_test_loss, self.valid_summary], feed_dict = feed_dict)

                            printout(self.flog,"Epoch: [%5d|total] valid_allvae: %.8f, valid_loss: %.8f" % (epoch, cost_generation_valid),epoch)

                    if tensorboard:
                        summary_writer.add_summary(train_summary, epoch)
                        if not len(valid_feature) == 0:
                            summary_writer.add_summary(valid_summary, epoch)

                    if timeline_use:
                        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        many_runs_timeline.update_timeline(chrome_trace)

                    if (epoch+1) % 5000 == 0:
                        self.saver.save(sess, self.outputdir + '/convmesh-model', global_step = epoch+1)

                    if not len(valid_feature) == 0 and bidx == len(train_id)//batch_size:
                        if cost_generation_valid < 50 and cost_generation_valid < inf:
                            inf = cost_generation_valid
                            printout(self.flog,"Save Best(allvae_valid): %.8f\n"%(cost_generation_valid))
                            self.saver.save(sess, self.outputdir + '/convmesh-modelbest')
                    else:
                        if cost_generation < 60 and cost_generation < inf:
                            inf = cost_generation
                            printout(self.flog,"Save Best(cost_generation): %.8f\n"%(cost_generation))
                            # self.saver.save(sess, self.outputdir + '/convmesh-modelbest')

                    # end = time.time()

                    # print('time: %fs'%(end-start))

            if timeline_use:
                many_runs_timeline.save('timeline_03_merged_%d_runs.json' % (self.maxepoch-checkpoint_counter))

    def train_gen_loss(self):# trian with split the dataset to test the generalization error
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        rng = np.random.RandomState(23456)

        with tf.Session(config=config) as sess:
            could_load, checkpoint_counter = self.load(sess, self.outputdir)
            summary_writer = tf.summary.FileWriter(self.outputdir+'/logs')

            tf.global_variables_initializer().run()

            train_id, valid_id = spilt_dataset(len(self.feature),self.train_percent)
            #train_feature = feature[train_id]
            #valid_feature = feature[valid_id]

            inf = float('inf')

            for epoch in range(checkpoint_counter, self.maxepoch+1):

                rng.shuffle(train_id)
                rng.shuffle(valid_id)
                train_feature = self.feature[train_id]
                valid_feature = self.feature[valid_id]

                start = time.time()
                # train
                if self.ae:
                    random = np.zeros((len(train_feature), self.hiddendim)).astype('float32')
                    sess.run([self.optimizer], feed_dict = {self.inputs: train_feature})

                    cost_generation, cost_distance, cost_weights, cost_l2, train_summary = sess.run([self.generation_loss, self.distance_norm, self.weights_norm, self.l2_loss, self.train_summary], {self.inputs:train_feature})
                    printout(self.flog,"Epoch: [%5d|total] generation_loss: %.8f  dist_loss: %.8f weight_loss: %.8f l2_loss: %.8f" % (epoch, cost_generation, cost_distance, cost_weights, cost_l2))

                    cost_generation_valid, valid_summary = sess.run([self.test_loss, self.valid_summary], {self.inputs:valid_feature})
                    printout(self.flog,"Epoch: [%5d|total] valid_loss: %.8f" % (epoch, cost_generation_valid))

                else:
                    random = gaussian(len(train_feature), self.hiddendim)

                    sess.run([self.optimizer], feed_dict = {self.inputs: train_feature, self.embedding_inputs: random})

                    cost_generation, cost_distance, cost_weights, cost_kl, cost_l2, train_summary = sess.run([self.generation_loss, self.distance_norm, self.weights_norm, self.kl_diver, self.l2_loss, self.train_summary], {self.inputs:train_feature, self.embedding_inputs: random})
                    printout(self.flog,"Epoch: [%5d|total] generation_loss: %.8f  dist_loss: %.8f weight_loss: %.8f kl_loss: %.8f l2_loss: %.8f" % (epoch, cost_generation, cost_distance, cost_weights, cost_kl, cost_l2))

                    #valid
                    random = np.zeros((len(valid_feature), self.hiddendim))
                    cost_generation_valid, valid_summary = sess.run([self.test_loss, self.valid_summary], {self.inputs:valid_feature, self.embedding_inputs: random})
                    printout(self.flog,"Epoch: [%5d|total] valid_loss: %.8f" % (epoch, cost_generation_valid))

                summary_writer.add_summary(train_summary, epoch)
                summary_writer.add_summary(valid_summary, epoch)

                if (epoch+1) % 5000 == 0 or epoch == self.maxepoch:
                    self.saver.save(sess, self.outputdir + '/convmesh-model', global_step = epoch+1)

                if cost_generation_valid < 55 and cost_generation_valid < inf:
                    inf = cost_generation_valid
                    printout(self.flog,"Save Best: %.8f\n"%(cost_generation_valid))
                    self.saver.save(sess, self.outputdir + '/convmesh-modelbest')

                end = time.time()

                print('time: %fs'%(end-start))

#------------------------------------------------------------training function------------------------------------------------------------------------------------

    def train_total_deform_structure(self):# trian with split the dataset to test the generalization error
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        rng = np.random.RandomState(23456)
        batch_size = self.batch_size
        inf = float('inf')
        if timeline_use: # use the timeline to analyze the efficency of the program
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            many_runs_timeline = TimeLiner()
        else:
            options = None
            run_metadata = None

        # with tf.Session(config = self.config) as sess:
        tf.global_variables_initializer().run()

        could_load, checkpoint_counter = self.load(self.sess, self.checkpoint_dir_structure)
        if tensorboard:
            summary_writer = tf.summary.FileWriter(self.outputdir+'/logs', self.sess.graph)

        # train_id, valid_id = spilt_dataset(len(self.feature), self.train_percent, self.iddat_name)

        for epoch in range(checkpoint_counter, self.maxepoch_deform+self.maxepoch_structure+1):
            # train
            rng.shuffle(self.train_id)
            # printout(self.flog,"Train Epoch: %5d" % epoch)
            for bidx in xrange(0, len(self.train_id)//batch_size + 1):

                train_feature = [self.feature[i] for i in self.train_id[bidx*batch_size:min(len(self.train_id), bidx*batch_size+batch_size)]]
                train_symmetry_feature = [self.symmetry_feature[i] for i in self.train_id[bidx*batch_size:min(len(self.train_id), bidx*batch_size+batch_size)]]
                if len(train_feature) == 0:
                    continue
                # train_feature = np.unique(train_feature, axis=0)
                dictbase = {self.inputs_feature: train_feature, self.inputs_symmetry: train_symmetry_feature}
                if self.ae:
                    random = np.zeros((len(train_feature),200)).astype('float32')
                    dictrand = {x: random[:, 0: np.shape(x)[1]] for x in self.embedding_inputs}
                    feed_dict = merge_two_dicts(dictbase, dictrand)
                    _, cost_generation, weight_norm, cost_l2, train_summary = self.sess.run([self.total_trainop, self.total_generation_loss, self.total_weights_norm, self.total_l2_loss, self.train_summary], feed_dict = feed_dict, options=options, run_metadata=run_metadata)
                    # a = sess.run([self.l0], feed_dict = feed_dict)[0]
                    # print(a)
                    #sio.savemat('laplacian_check.mat', {'region':f})
                    printout(self.flog,"Epoch: {:6d} generation_loss: {:08.4f} weight_loss: {:08.4f} l2_loss: {:08.4f}".format(epoch, cost_generation, weight_norm, cost_l2),epoch)

                else:
                    dictrand = {x: gaussian(len(train_feature), np.shape(x)[1]) for x in self.embedding_inputs}
                    feed_dict = merge_two_dicts(dictbase, dictrand)
                    _, cost_generation, cost_kl, weight_norm, cost_l2, train_summary = self.sess.run([self.total_trainop, self.total_generation_loss, self.total_kl_loss, self.total_weights_norm, self.total_l2_loss, self.train_summary], feed_dict = feed_dict, options=options, run_metadata=run_metadata)
                    # a,b,c= self.sess.run([self.loss_vae[0],self.loss_vae[1],self.loss_vae[2]], feed_dict = feed_dict)
                    # print(a)
                    # sio.savemat('loss_test.mat', {'a':a,'b':b,'c':c,'d':d,'f':f,'g':g,'h':h})
                    # printout(self.flog,"a %.8f b: %.8f c: %.8f" % (a, b, c), epoch)

                    printout(self.flog,"Epoch: {:6d} generation_loss: {:08.4f} weight_loss: {:08.4f} kl_loss: {:08.4f} l2_loss: {:08.4f}".format(epoch, cost_generation, weight_norm, cost_kl, cost_l2), epoch)

                if (epoch+1) % 5000 == 0:
                    self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-model', global_step = epoch+1)

                if cost_generation < 60 and len(self.valid_id)==0 and cost_generation < inf:
                    inf = cost_generation
                    printout(self.flog,"Save Best(cost_generation): {:08.4f}\n".format(cost_generation))

            # valid
            # printout(self.flog,"Valid Epoch: %5d" % epoch)
            rng.shuffle(self.valid_id)
            valid_loss = 0
            for bidx in xrange(0, len(self.valid_id)//batch_size + 1):

                valid_feature = [self.feature[i] for i in self.valid_id[bidx*batch_size:min(len(self.valid_id), bidx*batch_size+batch_size)]]
                valid_symmetry_feature = [self.symmetry_feature[i] for i in self.valid_id[bidx*batch_size:min(len(self.valid_id), bidx*batch_size+batch_size)]]
                if len(valid_feature) == 0:
                    continue
                # valid_feature = np.unique(valid_feature, axis=0)
                dictbase = {self.inputs_feature: valid_feature, self.inputs_symmetry: valid_symmetry_feature}
                random = np.zeros((len(valid_feature),200)).astype('float32')
                if self.ae:
                    dictrand = {x: random[:, 0: np.shape(x)[1]] for x in self.embedding_inputs}
                    feed_dict = merge_two_dicts(dictbase, dictrand)

                    cost_generation_valid, valid_summary = self.sess.run([self.total_test_loss, self.valid_summary], feed_dict = feed_dict)

                    printout(self.flog,"Epoch: {:6d} valid_gen_loss: {:08.4f}".format(epoch, cost_generation_valid), epoch)
                else:

                    dictrand = {x: random[:, 0: np.shape(x)[1]] for x in self.embedding_inputs}
                    feed_dict = merge_two_dicts(dictbase, dictrand)
                    cost_generation_valid, cost_kl_valid, valid_summary = self.sess.run([self.total_test_loss, self.total_testkl_loss, self.valid_summary], feed_dict = feed_dict)

                    printout(self.flog,"Epoch: {:6d} valid_gen_loss:{:08.4f} valid_kl_loss: {:08.4f}".format(epoch, cost_generation_valid, cost_kl_valid),epoch)
                valid_loss+=cost_generation_valid*len(valid_feature)
            if len(self.valid_id)>0:
                valid_loss/=len(self.valid_id)
                if valid_loss < 50 and valid_loss < inf:
                    inf = valid_loss
                    printout(self.flog,"Save Best(cost_generation_valid): {:08.4f}\n".format(valid_loss))
                    self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-modelbest')


            if tensorboard:
                summary_writer.add_summary(train_summary, epoch)
                if not len(valid_feature) == 0:
                    summary_writer.add_summary(valid_summary, epoch)

            if timeline_use:
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                many_runs_timeline.update_timeline(chrome_trace)

        if timeline_use:
            many_runs_timeline.save('timeline_03_merged_{}_runs.json'.format(self.maxepoch_deform-checkpoint_counter))

    def train_pre(self):
        tf.global_variables_initializer().run()
        # tf.local_variables_initializer().run()
        # self.write = tf.summary.FileWriter(logfolder + '/logs/', self.sess.graph)
        if tensorboard:
            self.summary_writer = tf.summary.FileWriter(self.outputdir+'/logs', self.sess.graph)

        # if not os.path.exists(self.checkpoint_dir_defrom):
            # os.makedirs(self.checkpoint_dir_defrom)

        if not os.path.exists(self.checkpoint_dir_structure):
            os.makedirs(self.checkpoint_dir_structure)

        # could_load_deform, checkpoint_counter_deform = self.load(self.sess, self.checkpoint_dir_defrom)
        could_load_struture, checkpoint_counter_struture = self.load(self.sess, self.checkpoint_dir_structure)

        if (could_load_struture and checkpoint_counter_struture <= self.maxepoch_deform):
            self.start = 'DEFORM'
            self.start_step_deform = checkpoint_counter_struture
            self.start_step_structure = 0
        elif (could_load_struture and checkpoint_counter_struture <= self.maxepoch_structure):
            self.start = 'STRUCTURE'
            self.start_step_structure = checkpoint_counter_struture
        else:
            self.start_step_deform = 0
            self.start_step_structure = 0
            self.start = 'DEFORM'
            print('we start from VAE...')

    def train_deform(self):
        printout(self.flog,"Train DEFORM Net...")
        rng = np.random.RandomState(23456)
        batch_size = self.batch_size
        inf = float('inf')
        if timeline_use: # use the timeline to analyze the efficency of the program
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            many_runs_timeline = TimeLiner()
        else:
            options = None
            run_metadata = None

        # with tf.Session(config = self.config) as sess:
        # train_id, valid_id = spilt_dataset(len(self.feature), self.train_percent, self.iddat_name)

        for epoch in range(self.start_step_deform, self.maxepoch_deform):
            # train
            rng.shuffle(self.train_id)

            # for part_id in range(self.part_num):
                # printout(self.flog,"Train Epoch: %5d Part ID: %5d Part Name: %s" % (epoch, part_id, self.part_name[part_id]))
            # time1=time.time()
            for bidx in xrange(0, len(self.train_id)//batch_size + 1):

                train_feature = [self.feature[i,:,:,:] for i in self.train_id[bidx*batch_size:min(len(self.train_id), bidx*batch_size+batch_size)]]
                # train_symmetry_feature = [self.symmetry_feature[i] for i in train_id[bidx*batch_size:min(len(train_id), bidx*batch_size+batch_size)]]
                if len(train_feature) == 0:
                    continue
                # train_feature = np.unique(train_feature, axis=0)
                dictbase = {self.inputs_feature: train_feature}
                if self.ae:
                    random = np.zeros((len(train_feature),200)).astype('float32')
                    # dictrand = {self.embedding_inputs[part_id]: random[:, 0: np.shape(self.embedding_inputs[part_id])[1]]}
                    dictrand = {x: random[:, 0: np.shape(x)[1]] for x in self.embedding_inputs[:-1]}
                    feed_dict = merge_two_dicts(dictbase, dictrand)
                    # _, cost_generation, weights_norm, cost_l2, train_summary = self.sess.run([self.train_op[part_id], self.generation_loss[part_id], self.weights_norm[part_id], self.l2_loss[part_id], self.train_summary_one[part_id]], feed_dict = feed_dict, options=options, run_metadata=run_metadata)
                    _, cost_generation, weights_norm, cost_l2, train_summary = self.sess.run([self.train_op[:-1], self.generation_loss[:-1], self.weights_norm[:-1], self.l2_loss[:-1], self.train_summary_one[:-1]], feed_dict = feed_dict, options=options, run_metadata=run_metadata)
                    # a = sess.run([self.l0], feed_dict = feed_dict)[0]
                    # print(a)
                    #sio.savemat('laplacian_check.mat', {'region':f})

                    printout(self.flog,"Train Epoch: {:6d} \nPart  Name: {} \ngener_loss: {} \nweight_loss: {} \nl2222_loss: {}".format(epoch, self.part_name, cost_generation, weights_norm, cost_l2),epoch)

                else:
                    # dictrand = {self.embedding_inputs[part_id]: gaussian(len(train_feature), np.shape(self.embedding_inputs[part_id])[1])}
                    dictrand = {x: gaussian(len(train_feature), np.shape(x)[1]) for x in self.embedding_inputs[:-1]}
                    feed_dict = merge_two_dicts(dictbase, dictrand)
                    # _, cost_generation, weights_norm, cost_kl, cost_l2, train_summary = self.sess.run([self.train_op[part_id], self.generation_loss[part_id], self.weights_norm[part_id], self.kl_diver[part_id], self.l2_loss[part_id], self.train_summary_one[part_id]], feed_dict = feed_dict, options=options, run_metadata=run_metadata)
                    _, cost_generation, weights_norm, cost_kl, cost_l2, train_summary = self.sess.run([self.train_op[:-1], self.generation_loss[:-1], self.weights_norm[:-1], self.kl_diver[:-1], self.l2_loss[:-1], self.train_summary_one[:-1]], feed_dict = feed_dict, options=options, run_metadata=run_metadata)
                    # a = sess.run([self.decode[0]], feed_dict = feed_dict)[0]
                    # print(cost_generation)

                    printout(self.flog, "Train Epoch: {:6d} \nPart  Name: {} \ngener_loss: {} \nweigh_loss: {} \nkllll_loss: {} \nl2222_loss: {}".format(epoch, self.part_name, cost_generation, weights_norm, cost_kl, cost_l2), epoch)

                if (epoch+1) % 5000 == 0:
                    self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-model', global_step = epoch+1)

                if np.max(cost_generation) < 60 and len(self.valid_id)==0 and np.max(cost_generation) < inf:
                    inf = cost_generation
                    printout(self.flog,"Save Best(cost_generation): %.8f"%(cost_generation))
            # print(time.time()-time1)
            # valid
            rng.shuffle(self.valid_id)
            # printout(self.flog, "Valid Epoch: %5d Part ID: %5d Part Name: %s" % (epoch, part_id, self.part_name[part_id]))
            valid_loss=0
            for bidx in xrange(0, len(self.valid_id)//batch_size + 1):

                valid_feature = [self.feature[i,:,:,:] for i in self.valid_id[bidx*batch_size:min(len(self.valid_id), bidx*batch_size+batch_size)]]
                # valid_symmetry_feature = [self.symmetry_feature[i] for i in valid_id[bidx*batch_size:min(len(valid_id), bidx*batch_size+batch_size)]]
                if len(valid_feature) == 0:
                    continue
                # valid_feature = np.unique(valid_feature, axis=0)
                dictbase = {self.inputs_feature: valid_feature}
                random = np.zeros((len(valid_feature),200)).astype('float32')
                if self.ae:
                    # dictrand = {self.embedding_inputs[part_id]: random[:, 0: np.shape(self.embedding_inputs[part_id])[1]]}
                    dictrand = {x: random[:, 0: np.shape(x)[1]] for x in self.embedding_inputs[:-1]}
                    feed_dict = merge_two_dicts(dictbase, dictrand)

                    cost_generation_valid, valid_summary = self.sess.run([self.test_loss[:-1], self.valid_summary_one[:-1]], feed_dict = feed_dict)

                    printout(self.flog,"Train Epoch: {:6d} \nPart  Name: {} \nvalid_gen_loss: {}".format(epoch, self.part_name[:-1], cost_generation_valid),epoch)
                else:

                    # dictrand = {self.embedding_inputs[part_id]: random[:, 0: np.shape(self.embedding_inputs[part_id])[1]]}
                    dictrand = {x: random[:, 0: np.shape(x)[1]] for x in self.embedding_inputs[:-1]}
                    feed_dict = merge_two_dicts(dictbase, dictrand)
                    cost_generation_valid, cost_kl_valid, valid_summary = self.sess.run([self.test_loss[:-1], self.testkl_diver[:-1], self.valid_summary_one[:-1]], feed_dict = feed_dict)

                    printout(self.flog,"Train Epoch: {:6d} \nPart  Name: {} \nvalid_Gen_loss: {} \nvalid_kl_loss: {}".format(epoch, self.part_name, cost_generation_valid, cost_kl_valid),epoch)
                valid_loss+=np.max(cost_generation_valid)*len(valid_feature)

            if len(self.valid_id)>0:
                valid_loss/=len(self.valid_id)
                if valid_loss < 50 and valid_loss < inf:
                    inf = valid_loss
                    printout(self.flog,"Save Best(cost_generation_valid): {:08.4f}".format(valid_loss))
                    self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-modelbest')

            if tensorboard:
                self.summary_writer.add_summary(train_summary, epoch)
                if not len(valid_feature) == 0:
                    self.summary_writer.add_summary(valid_summary, epoch)

            if timeline_use:
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                many_runs_timeline.update_timeline(chrome_trace)

        if timeline_use:
            many_runs_timeline.save('timeline_03_merged_{}_runs.json'.format(epoch))

    def train_structure(self):
        printout(self.flog,"Train Structure Net...")
        rng = np.random.RandomState(23456)
        batch_size = self.batch_size
        inf = float('inf')
        if timeline_use: # use the timeline to analyze the efficency of the program
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            many_runs_timeline = TimeLiner()
        else:
            options = None
            run_metadata = None

        # train_id, valid_id = spilt_dataset(len(self.feature), self.train_percent, self.iddat_name)

        for epoch in range(self.start_step_structure, self.maxepoch_structure):
            # train
            rng.shuffle(self.train_id)
            # printout(self.flog,"Train Epoch: %5d" % epoch)
            # time1=time.time()
            for bidx in xrange(0, len(self.train_id)//batch_size + 1):

                train_feature = [self.feature[i,:,:,:] for i in self.train_id[bidx*batch_size:min(len(self.train_id), bidx*batch_size+batch_size)]]
                train_symmetry_feature = [self.symmetry_feature[i] for i in self.train_id[bidx*batch_size:min(len(self.train_id), bidx*batch_size+batch_size)]]
                if len(train_feature) == 0:
                    continue
                # train_feature = np.unique(train_feature, axis=0)
                dictbase = {self.inputs_feature: train_feature, self.inputs_symmetry: train_symmetry_feature}
                if self.ae:
                    random = np.zeros((len(train_feature),200)).astype('float32')
                    dictrand = {self.embedding_inputs[-1]: random[:, 0: np.shape(self.embedding_inputs[-1])[1]]}
                    feed_dict = merge_two_dicts(dictbase, dictrand)
                    _, cost_generation, cost_l2, train_summary = self.sess.run([self.train_op[-1], self.generation_loss[-1], self.l2_loss[-1], self.train_summary_one[-1]], feed_dict = feed_dict, options=options, run_metadata=run_metadata)
                    # a = sess.run([self.l0], feed_dict = feed_dict)[0]
                    # print(a)
                    #sio.savemat('laplacian_check.mat', {'region':f})

                    printout(self.flog,"Epoch: {:6d} generation_loss: {:08.4f} l2_loss: {:08.4f}".format(epoch, cost_generation, cost_l2),epoch)

                else:
                    dictrand = {self.embedding_inputs[-1]: gaussian(len(train_feature), np.shape(self.embedding_inputs[-1])[1])}
                    feed_dict = merge_two_dicts(dictbase, dictrand)
                    _, cost_generation, cost_kl, cost_l2, train_summary = self.sess.run([self.train_op[-1], self.generation_loss[-1], self.kl_diver[-1], self.l2_loss[-1], self.train_summary_one[-1]], feed_dict = feed_dict, options=options, run_metadata=run_metadata)
                    # a = sess.run([self.decode[0]], feed_dict = feed_dict)[0]
                    # print(cost_generation)

                    printout(self.flog,"Epoch: {:6d} generation_loss: {:08.4f} kl_loss: {:08.4f} l2_loss: {:08.4f}".format(epoch, cost_generation, cost_kl, cost_l2),epoch)

                if (epoch+1) % 5000 == 0:
                    self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-model', global_step = epoch+1)

                if cost_generation < 60 and len(self.valid_id)==0 and cost_generation < inf:
                    inf = cost_generation
                    printout(self.flog,"Save Best(cost_generation): {:08.4f}".format(cost_generation))
            # print(time.time()-time1)
            # valid
            rng.shuffle(self.valid_id)
            # printout(self.flog,"Valid Epoch: %5d" % epoch)
            valid_loss=0
            for bidx in xrange(0, len(self.valid_id)//batch_size + 1):

                valid_feature = [self.feature[i,:,:,:] for i in self.valid_id[bidx*batch_size:min(len(self.valid_id), bidx*batch_size+batch_size)]]
                valid_symmetry_feature = [self.symmetry_feature[i] for i in self.valid_id[bidx*batch_size:min(len(self.valid_id), bidx*batch_size+batch_size)]]
                if len(valid_feature) == 0:
                    continue
                # valid_feature = np.unique(valid_feature, axis=0)
                dictbase = {self.inputs_feature: valid_feature,self.inputs_symmetry: valid_symmetry_feature}
                random = np.zeros((len(valid_feature),200)).astype('float32')
                if self.ae:
                    dictrand = {self.embedding_inputs[-1]: random[:, 0: np.shape(self.embedding_inputs[-1])[1]]}
                    feed_dict = merge_two_dicts(dictbase, dictrand)

                    cost_generation_valid, valid_summary = self.sess.run([self.test_loss[-1], self.valid_summary_one[-1]], feed_dict = feed_dict)

                    printout(self.flog,"valid_gen_loss: {:08.4f}".format(cost_generation_valid),epoch)
                else:

                    dictrand = {self.embedding_inputs[-1]: random[:, 0: np.shape(self.embedding_inputs[-1])[1]]}
                    feed_dict = merge_two_dicts(dictbase, dictrand)
                    cost_generation_valid, cost_kl_valid, valid_summary = self.sess.run([self.test_loss[-1], self.testkl_diver[-1], self.valid_summary_one[-1]], feed_dict = feed_dict)

                    printout(self.flog,"valid_gen_loss: {:08.4f}, valid_kl_loss: {:08.4f}".format(cost_generation_valid, cost_kl_valid),epoch)
                valid_loss+=cost_generation_valid*len(valid_feature)
            if len(self.valid_id)>0:
                valid_loss/=len(self.valid_id)
                if valid_loss < 50 and valid_loss < inf:
                    inf = valid_loss
                    printout(self.flog,"Save Best(cost_generation_valid): {:08.4f}".format(valid_loss))
                    self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-modelbest')

            if tensorboard:
                self.summary_writer.add_summary(train_summary, epoch)
                if not len(valid_feature) == 0:
                    self.summary_writer.add_summary(valid_summary, epoch)

            if timeline_use:
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                many_runs_timeline.update_timeline(chrome_trace)

        if timeline_use:
            many_runs_timeline.save('timeline_03_merged_{}_runs.json'.format(epoch))

    def train_total_deform_structure1(self):# trian with split the dataset to test the generalization error

        inf = float('inf')

        tf.global_variables_initializer().run()
        # tf.local_variables_initializer().run()

        if not os.path.exists(self.checkpoint_dir_structure):
            os.makedirs(self.checkpoint_dir_structure)
        could_load, checkpoint_counter = self.load(self.sess, self.checkpoint_dir_structure)
        if tensorboard:
            summary_writer = tf.summary.FileWriter(self.outputdir+'/logs', self.sess.graph)

        train_feature = self.feature[self.train_id]
        train_symmetry_feature = self.symmetry_feature[self.train_id]
        train_feature, train_symmetry_feature = get_batch_data(train_feature, train_symmetry_feature, self.batch_size)
        dictbase_train = {self.inputs_feature: train_feature, self.inputs_symmetry: train_symmetry_feature}
        random_train = np.zeros((self.batch_size,200)).astype('float32')


        valid_feature = self.feature[self.valid_id]
        valid_symmetry_feature = self.symmetry_feature[self.valid_id]
        valid_feature, valid_symmetry_feature = get_batch_data(valid_feature, valid_symmetry_feature, self.batch_size)
        dictbase_valid = {self.inputs_feature: valid_feature, self.inputs_symmetry: valid_symmetry_feature}
        random_valid = np.zeros((self.batch_size,200)).astype('float32')


        for epoch in range(self.maxepoch_deform+self.maxepoch_structure):
            train_handle = self.sess.run(self.train_iterator.string_handle())
            valid_handle = self.sess.run(self.valid_iterator.string_handle())
            self.sess.run(self.train_iterator.initializer, feed_dict=dictbase_train)
            self.sess.run(self.valid_iterator.initializer, feed_dict=dictbase_valid)
            printout(self.flog, self.iddat_name , epoch, interval=100, write_to_file = False)

            if self.ae:
                dictrand = {x: random_train[:, 0: np.shape(x)[1]] for x in self.embedding_inputs}
                feed_dict_train = merge_two_dicts({self.handle: train_handle}, dictrand)
            else:
                dictrand = {x: gaussian(self.batch_size, np.shape(x)[1]) for x in self.embedding_inputs}
                feed_dict_train = merge_two_dicts({self.handle: train_handle}, dictrand)
            lowvalue = float('inf')
            while True:
                try:
                    _, cost_generation, cost_kl, weights_norm, cost_l2, train_summary = self.sess.run([self.total_trainop, self.total_generation_loss, self.total_kl_loss, self.total_weights_norm, self.total_l2_loss, self.train_summary], feed_dict = feed_dict_train)
                    if len(self.valid_id)==0 and np.max(cost_generation) < lowvalue:
                        lowvalue = np.max(cost_generation)
                except tf.errors.OutOfRangeError:
                    break

            # a = sess.run([self.l0], feed_dict = feed_dict)[0]
            # print(a)
            #sio.savemat('laplacian_check.mat', {'region':f})

            printout(self.flog,"Train Epoch: {:6d} gener_loss: {:08.4f} weight_loss: {:08.4f} kllll_loss: {:08.4f} l2222_loss: {:08.4f}".format(epoch, cost_generation, weights_norm, cost_kl, cost_l2), epoch)

            if (epoch+1) % 5000 == 0:
                self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-model', global_step = epoch+1)

            if lowvalue < 50 and len(self.valid_id) == 0 and lowvalue < inf:
                inf = lowvalue
                printout(self.flog,"Save Best(cost_generation): {:08.4f}\n".format(inf))

            if len(valid_feature)>0:
            # dictrand = {x: random_valid[:, 0: np.shape(x)[1]] for x in self.embedding_inputs}
            # feed_dict_valid = merge_two_dicts(dictbase_valid, dictrand)
                while True:
                    try:
                        cost_generation_valid, cost_kl_valid, valid_summary = self.sess.run([self.total_test_loss, self.total_testkl_loss, self.valid_summary], feed_dict = {self.handle: valid_handle})
                        if np.max(cost_generation_valid) < lowvalue:
                            lowvalue = np.max(cost_generation_valid)
                    except tf.errors.OutOfRangeError:
                        break

                printout(self.flog,"Train Epoch: {:6d} valid_gen_loss:{:08.4f} valid_kl_loss: {:08.4f}".format(epoch, cost_generation_valid, cost_kl_valid),epoch)

                if lowvalue < 60 and lowvalue < inf:
                    inf = lowvalue
                    printout(self.flog,"Save Best(cost_generation_valid): {:08.4f}".format(inf))
                    self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-modelbest')

            if tensorboard:
                self.summary_writer.add_summary(train_summary, epoch)
                if not len(valid_feature) == 0:
                    self.summary_writer.add_summary(valid_summary, epoch)

    def train_deform_structure_seperate1(self): # advance api
        printout(self.flog,"Train Structure Net...")
        inf = float('inf')
        tf.global_variables_initializer().run()
        # tf.local_variables_initializer().run()

        if tensorboard:
            self.summary_writer = tf.summary.FileWriter(self.outputdir+'/logs', self.sess.graph)

        if not os.path.exists(self.checkpoint_dir_structure):
            os.makedirs(self.checkpoint_dir_structure)

        could_load_struture, checkpoint_counter_struture = self.load(self.sess, self.checkpoint_dir_structure)

        train_feature = self.feature[self.train_id]
        train_symmetry_feature = self.symmetry_feature[self.train_id]
        train_feature, train_symmetry_feature = get_batch_data(train_feature, train_symmetry_feature, self.batch_size)
        dictbase_train = {self.inputs_feature: train_feature, self.inputs_symmetry: train_symmetry_feature}
        random_train = np.zeros((self.batch_size, 200)).astype('float32')
        # self.sess.run(self.iterator.initializer, feed_dict=dictbase_train)

        valid_feature = self.feature[self.valid_id]
        valid_symmetry_feature = self.symmetry_feature[self.valid_id]
        valid_feature, valid_symmetry_feature = get_batch_data(valid_feature, valid_symmetry_feature, self.batch_size)
        dictbase_valid = {self.inputs_feature: valid_feature, self.inputs_symmetry: valid_symmetry_feature}
        random_valid = np.zeros((self.batch_size, 200)).astype('float32')

        train_handle = self.sess.run(self.train_iterator.string_handle())
        valid_handle = self.sess.run(self.valid_iterator.string_handle())

        for epoch in range(self.maxepoch_deform+self.maxepoch_structure):

            self.sess.run(self.train_iterator.initializer, feed_dict=dictbase_train)
            self.sess.run(self.valid_iterator.initializer, feed_dict=dictbase_valid)
            lowvalue = float('inf')
            printout(self.flog, self.iddat_name, epoch, interval=100, write_to_file = False)
            if epoch < self.maxepoch_deform:
                if self.ae:
                    dictrand = {x: random_train[:, 0: np.shape(x)[1]] for x in self.embedding_inputs[:-1]}
                    feed_dict_train = merge_two_dicts(dictrand, {self.handle: train_handle})
                else:
                    dictrand = {x: gaussian(self.batch_size, np.shape(x)[1]) for x in self.embedding_inputs[:-1]}
                    feed_dict_train = merge_two_dicts(dictrand, {self.handle: train_handle})
                # time1=time.time()
                while True:
                    try:
                        _, cost_generation, weights_norm, cost_kl, cost_l2, train_summary = self.sess.run([self.train_op[:-1], self.generation_loss[:-1], self.weights_norm[:-1], self.kl_diver[:-1], self.l2_loss[:-1], self.train_summary_one[:-1]], feed_dict=feed_dict_train)
                        if len(self.valid_id)==0 and np.max(cost_generation) < lowvalue:
                            lowvalue = np.max(cost_generation)
                    except tf.errors.OutOfRangeError:
                        break
                # print(time.time()-time1)
                # a = sess.run([self.l0], feed_dict = feed_dict)[0]
                # print(a)
                #sio.savemat('laplacian_check.mat', {'region':f})

                printout(self.flog,"Train Epoch: {:6d} \nPart  Name: {} \ngener_loss: {} \nweigh_loss: {} \nkllll_loss: {} \nl2222_loss: {}".format(epoch, self.part_name, cost_generation, weights_norm, cost_kl, cost_l2), epoch)

                if (epoch + 1) % 5000 == 0:
                    self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-model', global_step = epoch+1)

                if lowvalue < 50 and len(self.valid_id)==0 and lowvalue < inf:
                    inf = lowvalue
                    printout(self.flog,"Save Best(cost_generation): {:08.4f}".format(inf))

                # dictrand = {x: random_valid[:, 0: np.shape(x)[1]] for x in self.embedding_inputs[:-1]}
                # feed_dict_valid = merge_two_dicts({self.caseid: 1}, dictbase_valid)
                # self.sess.run(self.iterator.initializer, feed_dict=feed_dict_valid)
                if len(valid_feature)>0:
                    while True:
                        try:
                            cost_generation_valid, cost_kl_valid, valid_summary = self.sess.run([self.test_loss[:-1], self.testkl_diver[:-1], self.valid_summary_one[:-1]],feed_dict={self.handle: valid_handle})
                            if np.max(cost_generation_valid) < lowvalue:
                                lowvalue = np.max(cost_generation_valid)
                        except tf.errors.OutOfRangeError:
                            break

                    printout(self.flog,"Train Epoch: {:6d} \nPart  Name: {} \nvalid_Gen_loss: {} \nvalid_kl_loss: {}".format(epoch, self.part_name, cost_generation_valid, cost_kl_valid),epoch)

                    if lowvalue < 60 and lowvalue < inf:
                        inf = lowvalue
                        printout(self.flog,"Save Best(cost_generation_valid): {:08.4f}".format(inf))
                        self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-modelbest')

                if tensorboard:
                    self.summary_writer.add_summary(train_summary, epoch)
                    if not len(valid_feature) == 0:
                        self.summary_writer.add_summary(valid_summary, epoch)

            else:

                if self.ae:
                    dictrand = {self.embedding_inputs[-1]: random_train[:, 0: np.shape(self.embedding_inputs[-1])[1]]}
                    feed_dict_train = merge_two_dicts({self.handle: train_handle}, dictrand)
                else:
                    dictrand = {self.embedding_inputs[-1]: gaussian(self.batch_size, np.shape(self.embedding_inputs[-1])[1])}
                    feed_dict_train = merge_two_dicts({self.handle: train_handle}, dictrand)
                # time1 = time.time()
                while True:
                    try:
                        _, cost_generation, cost_kl, cost_l2, train_summary = self.sess.run([self.train_op[-1], self.generation_loss[-1], self.kl_diver[-1], self.l2_loss[-1], self.train_summary_one[-1]], feed_dict = feed_dict_train)
                        if len(self.valid_id)==0 and np.max(cost_generation) < lowvalue:
                            lowvalue = np.max(cost_generation)
                    except tf.errors.OutOfRangeError:
                        break
                # print(time.time()-time1)
                # a = sess.run([self.decode[0]], feed_dict = feed_dict)[0]
                # print(cost_generation)

                printout(self.flog,"Epoch: {:6d} gener_loss: {:08.4f} kllll_loss: {:08.4f} l2222_loss: {:08.4f}".format(epoch, cost_generation, cost_kl, cost_l2),epoch)

                if (epoch+1) % 5000 == 0:
                    self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-model', global_step = epoch+1)

                if lowvalue < 50 and len(self.valid_id)==0 and lowvalue < inf:
                    inf = lowvalue
                    printout(self.flog,"Save Best(cost_generation): {:08.4f}".format(inf))
                # dictrand = {self.embedding_inputs[-1]: random_valid[:, 0: np.shape(self.embedding_inputs[-1])[1]]}
                # feed_dict_valid = merge_two_dicts(dictbase_valid, dictrand)
                if len(valid_feature)>0:
                    while True:
                        try:
                            cost_generation_valid, cost_kl_valid, valid_summary = self.sess.run([self.test_loss[-1], self.testkl_diver[-1], self.valid_summary_one[-1]], feed_dict = {self.handle: valid_handle})
                            if np.max(cost_generation_valid) < lowvalue:
                                lowvalue = np.max(cost_generation_valid)
                        except tf.errors.OutOfRangeError:
                            break

                    printout(self.flog,"valid_gen_loss: {:08.4f}, valid_kl_loss: {:08.4f}".format(cost_generation_valid, cost_kl_valid),epoch)

                    if lowvalue < 60 and lowvalue < inf:
                        inf = lowvalue
                        printout(self.flog,"Save Best(cost_generation_valid): {:08.4f}".format(inf))
                        self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-modelbest')

                if tensorboard:
                    self.summary_writer.add_summary(train_summary, epoch)
                    if not len(valid_feature) == 0:
                        self.summary_writer.add_summary(valid_summary, epoch)

    def train_scvae(self):

        if advance_api:
            with tf.Session(config = self.config) as self.sess:
                if self.union:
                    self.train_total_deform_structure1()
                else:
                    self.train_deform_structure_seperate1()

            print(self.outputdir)
        else:
            with tf.Session(config = self.config) as self.sess:
                if self.union:
                    self.train_total_deform_structure()
                else:
                    self.train_pre()
                    if self.start == 'DEFORM':
                        self.train_deform()
                        self.train_structure()
                    elif self.start == 'STRUCTURE':
                        self.train_structure()
                    else:
                        print('Training Ending!')

            print(self.outputdir)

#------------------------------------------------------------training function------------------------------------------------------------------------------------

#------------------------------------------------------------applications--------------------------------------------------------------------------------------

    def recover_mesh(self, datainfo, epoch = 0):
        with tf.Session(config = self.config) as sess:
            tf.global_variables_initializer().run()
            _, epoch = self.load(sess, self.checkpoint_dir_structure)
            path = self.checkpoint_dir_structure +'/../recon'+str(epoch)

            if not os.path.isdir(path):
                os.makedirs(path)
            # train_id, valid_id = spilt_dataset(len(self.feature),self.train_percent,self.iddat_name)

            random = np.zeros((len(self.feature), 200)).astype('float32')
            dictbase = {self.inputs_feature: self.feature, self.inputs_symmetry: self.symmetry_feature}
            dictrand = {x: random[:, 0: np.shape(x)[1]] for x in self.embedding_inputs}
            feed_dict = merge_two_dicts(dictbase, dictrand)

            if advance_api:
                app_handle = sess.run(self.app_iterator.string_handle())

            for i in range(self.part_num+1):
                if advance_api:
                    sess.run(self.app_iterator.initializer, feed_dict=dictbase)
                    while True:
                        try:
                            recover, emb, std = sess.run([self.test_decode[i], self.test_encode[i], self.test_encode_std[i]], feed_dict = merge_two_dicts({self.handle: app_handle}, dictrand))
                        except tf.errors.OutOfRangeError:
                            break
                else:
                    recover, emb, std = sess.run([self.test_decode[i], self.test_encode[i], self.test_encode_std[i]], feed_dict = feed_dict)

                if i < self.part_num:
                    deforma_v1, deforma_v_align = sess.run([self.deform_vertex[i], self.deform_vertex_align[i]], feed_dict={self.feature2point: recover, self.controlpoint: np.tile([0,0,0], (np.shape(recover)[0], 1, 1))})
                    # deforma_v2, deforma_v_align = sess.run([self.deform_vertex, self.deform_vertex_align], feed_dict={self.feature2point: recover2, self.controlpoint: np.tile([0,0,0], (np.shape(recover2)[0], 1, 1))})
                    deforma_v1 = self.alignallmodels(deforma_v1, id = i, one=False)
                    # deforma_v2 = self.alignallmodels(deforma_v2, one=False)
                    # deforma_v_align = alignallmodels(deforma_v_align)
                    # num_list_new = map(lambda x:str(x), np.arange(np.shape(deforma_v1)[0]))
                    # num_list_new = [str(x) for x in np.arange(np.shape(deforma_v1)[0])]
                    self.v2objfile(deforma_v1, path + '/' + self.part_name[i], self.modelname,np.arange(np.shape(deforma_v1)[0]),self.part_name[i])

                    # render_parallel(path + '/recon'+restore[-4:])
                    rs, rlogr = recover_data(recover, datainfo.logrmin[i], datainfo.logrmax[i], datainfo.smin[i], datainfo.smax[i], self.pointnum)
                    sio.savemat(path + '/' + self.part_name[i]+'/recover.mat', {'RS':rs, 'RLOGR':rlogr, 'tid':self.train_id, 'vid':self.valid_id, 'emb':emb, 'std':std})
                    printout(self.flog, "Erms: %.8f" % (self.calc_erms(deforma_v1, id = i)))
                else:
                    recoversym = np.reshape(recover, (len(recover), self.part_num, self.part_dim+self.hiddendim[0]))
                    if self.change_net:
                        # symmetryf = recover_datasym(recoversym[:,k,-self.part_dim:], datainfo.boxmin[0], datainfo.boxmax[0])
                        # for k in range(self.part_num):
                        symmetryf = np.concatenate([np.expand_dims(recover_datasymv2(recoversym[:,k,-self.part_dim:], datainfo.boxmin[k], datainfo.boxmax[k]), axis = 1) for k in range(self.part_num)], axis = 1)
                        sio.savemat(path+'/recover_sym.mat', {'symmetry_feature':symmetryf, 'tid':self.train_id, 'vid':self.valid_id, 'emb':emb, 'std':std})
                            # sio.savemat(path+'/recover_sym_',self.part_name[k],'.mat', {'symmetry_feature':symmetryf, 'tid':train_id, 'vid':valid_id, 'emb':emb, 'std':std})
                    else:
                        symmetryf = recover_datasym(recoversym[:,:,-self.part_dim:], datainfo.boxmin[0], datainfo.boxmax[0])
                        sio.savemat(path+'/recover_sym.mat', {'symmetry_feature':symmetryf, 'tid':self.train_id, 'vid':self.valid_id, 'emb':emb, 'std':std})
                        # for k in range(self.part_num):
                            # sio.savemat(path+'/recover_sym',self.part_name[k],'.mat', {'symmetry_feature':symmetryf[:,k,:], 'tid':train_id, 'vid':valid_id, 'emb':emb, 'std':std})

                    for k in range(self.part_num):
                        recover1 = sess.run([self.embedding_decode[k]], feed_dict = {self.embedding_inputs[k]: recoversym[:,k,:self.hiddendim[0]]})[0]
                        # recover1, emb, std = sess.run([self.test_decode[i], self.encode[i], self.encode_std[i]], feed_dict = feed_dict)
                        deforma_v1, deforma_v_align = sess.run([self.deform_vertex[k], self.deform_vertex_align[k]], feed_dict={self.feature2point: recover1, self.controlpoint: np.tile([0,0,0], (np.shape(recover1)[0], 1, 1))})
                        deforma_v1 = self.alignallmodels(deforma_v1, id = k, one=False)
                        self.v2objfile(deforma_v1, path + '/struc_' + self.part_name[k], self.modelname,np.arange(np.shape(deforma_v1)[0]),self.part_name[k])

                        rs, rlogr = recover_data(recover1, datainfo.logrmin[k], datainfo.logrmax[k], datainfo.smin[k], datainfo.smax[k], self.pointnum)
                        sio.savemat(path + '/struc_' + self.part_name[k]+'/recover.mat', {'RS':rs, 'RLOGR':rlogr, 'tid':self.train_id, 'vid':self.valid_id, 'emb':emb, 'std':std})
                        printout(self.flog, "Erms: %.8f" % (self.calc_erms(deforma_v1, id = k)))


            # embedding = sess.run([self.test_encode], feed_dict = {self.inputs: self.feature})[0]
            # if not self.symmetry:
            #     rs, rlogr = recover_data(recover1, datainfo.logrmin, datainfo.logrmax, datainfo.smin, datainfo.smax, self.pointnum)
            #     sio.savemat(path+'/recon'+restore[-4:]+'/recover.mat', {'RS':rs, 'RLOGR':rlogr, 'tid':train_id, 'vid':valid_id, 'emb':emb, 'std':std})
            #     printout(self.flog, "Erms: %.8f" % (self.calc_erms(deforma_v1)))
            # else:
            #     symmetryf = recover_datasym(recover1, datainfo.sym_featuremin, datainfo.sym_featuremax)
            #     sio.savemat(path+'/recon'+restore[-4:]+'/recover_sym.mat', {'symmetry_feature':symmetryf, 'tid':train_id, 'vid':valid_id, 'emb':emb, 'std':std})

            # printout(self.flog, "Erms: %.8f" % (self.calc_erms(deforma_v2)))

        print(path)

        return

    def random_gen(self, datainfo, epoch = 0):
        with tf.Session(config = self.config) as sess:
            tf.global_variables_initializer().run()
            _, epoch = self.load(sess, self.checkpoint_dir_structure)
            path = self.checkpoint_dir_structure +'/../random'+str(epoch)

            if not os.path.isdir(path):
                os.makedirs(path)
            # train_id, valid_id = spilt_dataset(len(self.feature),self.train_percent,self.iddat_name)

            for i in range(self.part_num+1):

                if i < self.part_num:
                    random = gaussian(200, self.hiddendim[0])
                    recover = sess.run([self.embedding_decode[i]], feed_dict = {self.embedding_inputs[i]: random})[0]
                    deforma_v1, deforma_v_align = sess.run([self.deform_vertex[i], self.deform_vertex_align[i]], feed_dict={self.feature2point: recover, self.controlpoint: np.tile([0,0,0], (np.shape(recover)[0], 1, 1))})
                    # deforma_v2, deforma_v_align = sess.run([self.deform_vertex, self.deform_vertex_align], feed_dict={self.feature2point: recover2, self.controlpoint: np.tile([0,0,0], (np.shape(recover2)[0], 1, 1))})
                    deforma_v1 = self.alignallmodels(deforma_v1,id = i)
                    # deforma_v2 = self.alignallmodels(deforma_v2, one=False)
                    # deforma_v_align = alignallmodels(deforma_v_align)
                    num_list_new = [str(x) for x in np.arange(np.shape(deforma_v1)[0])]
                    self.v2objfile(deforma_v1, path + '/' + self.part_name[i], num_list_new,num_list_new, self.part_name[i])

                    # render_parallel(path + '/recon'+restore[-4:])
                    rs, rlogr = recover_data(recover, datainfo.logrmin[i], datainfo.logrmax[i], datainfo.smin[i], datainfo.smax[i], self.pointnum)
                    sio.savemat(path + '/' + self.part_name[i]+'/random.mat', {'RS':rs, 'RLOGR':rlogr, 'tid':self.train_id, 'vid':self.valid_id, 'emb':random})
                    # printout(self.flog, "Erms: %.8f" % (self.calc_erms(deforma_v1)))
                else:
                    random = gaussian(200, self.hiddendim[1])
                    recover = sess.run([self.embedding_decode[i]], feed_dict = {self.embedding_inputs[i]: random})[0]
                    recoversym = np.reshape(recover, (len(recover), self.part_num, self.part_dim+self.hiddendim[0]))
                    # symmetryf = recover_datasym(recoversym[:,:,:self.part_dim], datainfo.logrmin[i], datainfo.logrmax[i])
                    # sio.savemat(path+'/random_sym.mat', {'symmetry_feature':symmetryf, 'tid':train_id, 'vid':valid_id, 'emb':random})
                    if self.change_net:
                        symmetryf = np.concatenate([np.expand_dims(recover_datasymv2(recoversym[:,k,-self.part_dim:], datainfo.boxmin[k], datainfo.boxmax[k]), axis = 1) for k in range(self.part_num)], axis = 1)
                        sio.savemat(path+'/random_sym.mat', {'symmetry_feature':symmetryf, 'tid':self.train_id, 'vid':self.valid_id, 'emb':random})
                    else:
                        symmetryf = recover_datasym(recoversym[:,:,-self.part_dim:], datainfo.boxmin[0], datainfo.boxmax[0])
                        sio.savemat(path+'/random_sym.mat', {'symmetry_feature':symmetryf, 'tid':self.train_id, 'vid':self.valid_id, 'emb':random})

                    for k in range(self.part_num):
                        recover1 = sess.run([self.embedding_decode[k]], feed_dict = {self.embedding_inputs[k]: recoversym[:,k,:self.hiddendim[0]]})[0]
                        # recover1, emb, std = sess.run([self.test_decode[i], self.encode[i], self.encode_std[i]], feed_dict = feed_dict)
                        deforma_v1, deforma_v_align = sess.run([self.deform_vertex[k], self.deform_vertex_align[k]], feed_dict={self.feature2point: recover1, self.controlpoint: np.tile([0,0,0], (np.shape(recover1)[0], 1, 1))})
                        deforma_v1 = self.alignallmodels(deforma_v1, id = k)
                        num_list_new = [str(x) for x in np.arange(np.shape(deforma_v1)[0])]
                        self.v2objfile(deforma_v1, path + '/struc_' + self.part_name[k], num_list_new, num_list_new, self.part_name[k])

                        rs, rlogr = recover_data(recover1, datainfo.logrmin[k], datainfo.logrmax[k], datainfo.smin[k], datainfo.smax[k], self.pointnum)
                        sio.savemat(path + '/struc_' + self.part_name[k]+'/random.mat', {'RS':rs, 'RLOGR':rlogr, 'tid':self.train_id, 'vid':self.valid_id, 'emb':recoversym[:,k,self.part_dim:]})
                        # printout(self.flog, "Erms: %.8f" % (self.calc_erms(deforma_v1)))

            # if not self.symmetry:
            #     start = time.time()
            #     deforma_v, deforma_v_align = sess.run([self.deform_vertex, self.deform_vertex_align], feed_dict={self.feature2point: recover, self.controlpoint: np.tile([0,0,0], (np.shape(recover)[0], 1, 1))})
            #     print('time: %fs'%(time.time()-start))
            #     print(np.shape(deforma_v))
            #     deforma_v = self.alignallmodels(deforma_v)
            #     # deforma_v_align = alignallmodels(deforma_v_align)
            #     self.v2objfile(deforma_v, path + '/random'+restore[-4:], np.arange(np.shape(deforma_v)[0]))
            #     # render_parallel(path + '/random'+restore[-4:])

            # # embedding = sess.run([self.test_encode], feed_dict = {self.inputs: self.feature})[0]
            # if not self.symmetry:
            #     rs, rlogr = recover_data(recover, datainfo.logrmin, datainfo.logrmax, datainfo.smin, datainfo.smax, self.pointnum)
            #     sio.savemat(path+'/random'+restore[-4:]+'/random.mat', {'RS':rs, 'RLOGR':rlogr, 'tid':train_id, 'vid':valid_id, 'emb':random})
            # else:
            #     symmetryf = recover_datasym(recover, datainfo.sym_featuremin, datainfo.sym_featuremax)
            #     sio.savemat(path+'/random'+restore[-4:]+'/random_sym.mat', {'symmetry_feature':symmetryf, 'tid':train_id, 'vid':valid_id, 'emb':random})

        print(path)

        return

    def interpolate1(self, datainfo, inter_id, epoch = 0): # [2, 10]
        with tf.Session(config = self.config) as sess:
            tf.global_variables_initializer().run()
            if epoch == 0:
                _success, epoch = self.load(sess, self.checkpoint_dir_structure)

            if not _success:
                raise Exception("抛出一个异常")
            path = self.checkpoint_dir_structure +'/../interpolation'+str(epoch)
            print(np.expand_dims([i for i in datainfo.modelname], axis=0))
            for i in range(len(inter_id)):
                if isinstance(inter_id[i], str) and len(inter_id[i])>len(str(datainfo.modelnum)):
                    inter_id[i] = datainfo.modelname.index(inter_id[i])
                else:
                    inter_id[i] = int(inter_id[i])
                print('ID: {:3d} Name: {}'.format(inter_id[i], datainfo.modelname[inter_id[i]]))

            if not os.path.isdir(path):
                os.makedirs(path)
            shutil.copy2(self.featurefile, path)
            
            if advance_api:
                app_handle = sess.run(self.app_iterator.string_handle())

            for i in range(self.part_num+1):

                if inter_id:
                    x = np.zeros([2, self.pointnum, self.finaldim])
                    inter_feature = self.feature[inter_id]
                    symmetry_feature = self.symmetry_feature[inter_id]
                    if advance_api:
                        sess.run(self.app_iterator.initializer, feed_dict={self.inputs_feature: self.feature, self.inputs_symmetry: self.symmetry_feature})
                        while True:
                            try:
                                embedding = sess.run([self.encode[i]], feed_dict = {self.handle: app_handle})[0]
                            except tf.errors.OutOfRangeError:
                                break
                        embedding = embedding[inter_id]
                    else:
                        embedding = sess.run([self.encode[i]], feed_dict = {self.inputs_feature: inter_feature, self.inputs_symmetry: symmetry_feature})[0]

                else:
                    embedding = gaussian(2, self.hiddendim[0])

                random2_intpl = interpolate.griddata(np.linspace(0, 1, len(embedding) * 1), embedding, np.linspace(0, 1, 200), method='linear')

                if i < self.part_num:
                    # random = gaussian(200, self.hiddendim[0])
                    recover = sess.run([self.embedding_decode[i]], feed_dict = {self.embedding_inputs[i]: random2_intpl})[0]
                    deforma_v1, deforma_v_align = sess.run([self.deform_vertex[i], self.deform_vertex_align[i]], feed_dict={self.feature2point: recover, self.controlpoint: np.tile([0,0,0], (np.shape(recover)[0], 1, 1))})
                    # deforma_v2, deforma_v_align = sess.run([self.deform_vertex, self.deform_vertex_align], feed_dict={self.feature2point: recover2, self.controlpoint: np.tile([0,0,0], (np.shape(recover2)[0], 1, 1))})
                    deforma_v1 = self.alignallmodels(deforma_v1, id = i)
                    # deforma_v2 = self.alignallmodels(deforma_v2, one=False)
                    # deforma_v_align = alignallmodels(deforma_v_align)
                    num_list_new = [str(x) for x in np.arange(np.shape(deforma_v1)[0])]
                    self.v2objfile(deforma_v1, path + '/' + self.part_name[i], num_list_new, num_list_new, self.part_name[i])

                    # render_parallel(path + '/recon'+restore[-4:])
                    rs, rlogr = recover_data(recover, datainfo.logrmin[i], datainfo.logrmax[i], datainfo.smin[i], datainfo.smax[i], self.pointnum)
                    sio.savemat(path + '/' + self.part_name[i]+'/inter.mat', {'RS':rs, 'RLOGR':rlogr, 'emb':random2_intpl})
                    # printout(self.flog, "Erms: %.8f" % (self.calc_erms(deforma_v1)))
                else:
                    # random = gaussian(200, self.hiddendim[1])
                    recover = sess.run([self.embedding_decode[i]], feed_dict = {self.embedding_inputs[i]: random2_intpl})[0]
                    recoversym = np.reshape(recover, (len(recover), self.part_num, self.part_dim+self.hiddendim[0]))
                    # symmetryf = recover_datasym(recoversym[:,:,:self.part_dim], datainfo.logrmin[i], datainfo.logrmax[i])
                    # sio.savemat(path+'/inter_sym.mat', {'symmetry_feature':symmetryf, 'emb':random2_intpl})
                    if self.change_net:
                        symmetryf = np.concatenate([np.expand_dims(recover_datasymv2(recoversym[:,k,-self.part_dim:], datainfo.boxmin[k], datainfo.boxmax[k]), axis = 1) for k in range(self.part_num)], axis = 1)
                        sio.savemat(path+'/inter_sym.mat', {'symmetry_feature':symmetryf, 'emb':random2_intpl})
                    else:
                        symmetryf = recover_datasym(recoversym[:,:,-self.part_dim:], datainfo.boxmin[0], datainfo.boxmax[0])
                        sio.savemat(path+'/inter_sym.mat', {'symmetry_feature':symmetryf, 'emb':random2_intpl})

                    for k in range(self.part_num):
                        recover1 = sess.run([self.embedding_decode[k]], feed_dict = {self.embedding_inputs[k]: recoversym[:,k,:self.hiddendim[0]]})[0]
                        # recover1, emb, std = sess.run([self.test_decode[i], self.encode[i], self.encode_std[i]], feed_dict = feed_dict)
                        deforma_v1, deforma_v_align = sess.run([self.deform_vertex[k], self.deform_vertex_align[k]], feed_dict={self.feature2point: recover1, self.controlpoint: np.tile([0,0,0], (np.shape(recover1)[0], 1, 1))})
                        deforma_v1 = self.alignallmodels(deforma_v1, id = k)
                        num_list_new = [str(x) for x in np.arange(np.shape(deforma_v1)[0])]
                        self.v2objfile(deforma_v1, path + '/struc_' + self.part_name[k], num_list_new, num_list_new, self.part_name[k])

                        rs, rlogr = recover_data(recover1, datainfo.logrmin[k], datainfo.logrmax[k], datainfo.smin[k], datainfo.smax[k], self.pointnum)
                        sio.savemat(path + '/struc_' + self.part_name[k]+'/inter.mat', {'RS':rs, 'RLOGR':rlogr, 'emb':recoversym[:,k,self.part_dim:]})
                        # printout(self.flog, "Erms: %.8f" % (self.calc_erms(deforma_v1)))

            # recover = sess.run([self.embedding_decode[0]], feed_dict = {self.embedding_inputs[0]: random2_intpl})[0]
            # if not self.symmetry:
            #     deforma_v, _ = sess.run([self.deform_vertex, self.deform_vertex_align], feed_dict={self.feature2point: recover, self.controlpoint: np.tile([0,0,0], (np.shape(recover)[0], 1, 1))})

            #     deforma_v = self.alignallmodels(deforma_v)
            #     # deforma_v_align = alignallmodels(deforma_v_align)
            #     self.v2objfile(deforma_v, path + '/interpolation'+restore[-4:], np.arange(np.shape(deforma_v)[0]))
            #     # render_parallel(path + '/interpolation'+restore[-4:])

            # # embedding = sess.run([self.test_encode], feed_dict = {self.inputs: self.feature})[0]
            # if not self.symmetry:
            #     rs, rlogr = recover_data(recover, datainfo.logrmin, datainfo.logrmax, datainfo.smin, datainfo.smax, self.pointnum)
            #     sio.savemat(path+'/interpolation'+restore[-4:]+'/inter.mat', {'RS':rs, 'RLOGR':rlogr, 'emb':random2_intpl})
            # else:
            #     symmetryf = recover_datasym(recover, datainfo.sym_featuremin, datainfo.sym_featuremax)
            #     sio.savemat(path+'/interpolation'+restore[-4:]+'/inter_sym.mat', {'symmetry_feature':symmetryf, 'emb':random2_intpl})

        print(path)

        return

#------------------------------------------------------------applications--------------------------------------------------------------------------------------

#------------------------------------------------------------feature2vertex--------------------------------------------------------------------------------------

    def runfeature2point(self, feature, idx, path): # feature must have two dimensions [m, n]
        assert(len(np.shape(feature)) == 2)
        all_feature = tf.expand_dims(feature, axis = 0)
        controlpoint = tf.gather(self.all_vertex[idx], idx, axis = 0)

        deforma_v, deforma_v_align = self.sess.run([self.deform_vertex, self.deform_vertex_align], feed_dict={self.feature2point: all_feature, self.controlpoint: controlpoint})
        self.v2objfile(deforma_v, path)
        self.v2objfile(deforma_v_align, path+'_align')

    def feature2point_pre(self, datainfo):
        with tf.device('/cpu:0'):
            self.reconmatrix = tf.constant(self.reconm, dtype = 'float32', shape = [self.pointnum, self.pointnum], name = 'reconmatrix')
            self.weight_vdiff = tf.constant(self.w_vdiff, dtype = 'float32', shape = [self.pointnum, self.maxdegree, 3], name = 'wvdiff')
            self.deform_reconmatrix = tf.constant(self.deform_reconmatrix_holder, dtype = 'float32', shape = [self.pointnum, self.pointnum + len(self.control_id)], name = 'deform_reconmatrix')

            self.feature2point  = tf.placeholder(tf.float32, [None, self.pointnum, self.vertex_dim], name = 'input_feature2point')

            self.controlpoint = tf.placeholder(tf.float32, [None, len(datainfo.control_idx), 3], name = 'controlpoint')
            # self.deform_reconmatrix_holder = tf.placeholder(tf.float32, shape = (self.pointnum, self.pointnum+len(datainfo.control_idx)), name = 'deform_reconmatrix')
            self.deform_vertex=[]
            self.deform_vertex_align=[]
            for i in range(len(datainfo.smax)):
                self.deform_vertex.append(self.Ttov(self.ftoT(self.feature2point, datainfo, i)))
                self.deform_vertex_align.append(self.deformTtov(self.ftoT(self.feature2point, datainfo, i)))
            # self.deform_vertex = self.Ttov(self.ftoT(self.feature2point, datainfo, self.id))
            # self.deform_vertex_align = self.deformTtov(self.ftoT(self.feature2point, datainfo, self.id))

    def ftoT(self, input_feature, datainfo, id = 0):
        resultmax = 0.95
        resultmin = -0.95
        batch_size = tf.shape(input_feature)[0]

        logr = input_feature[:,:,0:3]
        logr = (datainfo.logrmax[id] - datainfo.logrmin[id]) * (logr - resultmin) / (resultmax - resultmin) + datainfo.logrmin[id]

        theta = tf.sqrt(tf.reduce_sum(logr*logr, axis = 2))

        logr = tf.concat([logr,-logr, tf.zeros([batch_size, self.pointnum, 1], dtype = tf.float32)], 2)

        logr33 = tf.gather(logr, [6,0,1,3,6,2,4,5,6], axis = 2)
        logr33 = tf.reshape(logr33, (-1, self.pointnum, 3, 3))

        R = tf.eye(3, batch_shape=[batch_size, self.pointnum], dtype=tf.float32)

        condition = tf.equal(theta, 0)

        theta = tf.where(condition, tf.ones_like(theta), theta)

        x = logr33 / tf.expand_dims(tf.expand_dims(theta, 2),3)

        sintheta = tf.expand_dims(tf.expand_dims(tf.sin(theta), 2),3)
        costheta = tf.expand_dims(tf.expand_dims(tf.cos(theta), 2),3)

        R_ = R + x*sintheta + tf.matmul(x, x)*(1-costheta)

        condition = tf.reshape(tf.tile(condition, [1, 9]), (-1, self.pointnum, 3, 3))

        R = tf.where(condition, R, R_)

        S = tf.gather(input_feature, [3,4,5,4,6,7,5,7,8], axis = 2)
        S = tf.reshape(S, (-1, self.pointnum, 3, 3))# + tf.eye(3, batch_shape=[batch_size, self.pointnum], dtype=tf.float32)

        S = (datainfo.smax[id]-datainfo.smin[id])*(S-resultmin) / (resultmax - resultmin) + datainfo.smin[id]

        T = tf.matmul(R, S)

        return T

    def Ttov(self, input_feature):
        padding_feature = tf.zeros([tf.shape(input_feature)[0], 1, 3, 3], tf.float32)

        padded_input = tf.concat([padding_feature, input_feature], 1)

        nb_T = tf.gather(padded_input, self.nb, axis = 1)

        sum_T = nb_T + tf.expand_dims(input_feature, 2)

        bdiff = tf.einsum('abcde,bce->abcd',sum_T, self.weight_vdiff)

        b = tf.reduce_sum(bdiff, axis = 2)

        v = tf.einsum('ab,cbe->cae', self.reconmatrix, b)

        return v

    def deformTtov(self, input_feature):
        padding_feature = tf.zeros([tf.shape(input_feature)[0], 1, 3, 3], tf.float32)

        padded_input = tf.concat([padding_feature, input_feature], 1)

        nb_T = tf.gather(padded_input, self.nb, axis = 1)

        sum_T = nb_T + tf.expand_dims(input_feature, 2)

        bdiff = tf.einsum('abcde,bce->abcd',sum_T, self.weight_vdiff)

        # b = tf.concat([tf.reduce_sum(bdiff, axis = 2), tf.expand_dims(self.controlpoint, 0)], axis=1)
        b = tf.concat([tf.reduce_sum(bdiff, axis = 2), self.controlpoint], axis=1)

        v = tf.einsum('ab,cbe->cae', self.deform_reconmatrix, b)

        return v

    def alignallmodels(self, deforma_v, id = 0, one = True):
        model_num = np.shape(deforma_v)[0]
        align_deforma_v = np.zeros_like(deforma_v).astype('float32')
        if not one:
            for idx in range(model_num):
                source = self.all_vertex[idx, id,:,:]
                target = deforma_v[idx]
                T,R,t = icp.best_fit_transform(target, source)
                C = np.ones((np.shape(source)[0], 4))
                C[:,0:3] = target
                align_deforma_v[idx] = (np.dot(T, C.T).T)[:,0:3]
        else:
            source = self.all_vertex[1, id,:,:]
            for idx in range(model_num):
                target = deforma_v[idx]
                T,R,t = icp.best_fit_transform(target, source)
                C = np.ones((np.shape(source)[0], 4))
                C[:,0:3] = target
                align_deforma_v[idx] = (np.dot(T, C.T).T)[:,0:3]

        return align_deforma_v

    def v2objfile(self, deforma_v, path, sortid1, sortid2 = [0]*1000, post_suffix = ''):
        num = np.shape(deforma_v)[0]
        if not os.path.isdir(path):
            os.mkdir(path)
        for i in range(num):
            # namelist = [str(sortid1[i]+1), str(sortid2[i]+1), post_suffix]
            namelist = [sortid1[i], post_suffix]
            savemesh(self.mesh, path + '/'+ '_'.join(namelist)+ '.obj', deforma_v[i])
            # print(self.mesh, path + '/'+'_'.join(namelist) + '.obj')

    def calc_erms(self, ver, id = 0):
        assert(np.shape(ver)[0] == np.shape(self.all_vertex[:,id,:,:])[0])
        error = 1000*np.sqrt(np.sum(np.power(ver-self.all_vertex[:,id,:,:], 2))/(3.0 * np.shape(ver)[0] * self.pointnum))

        return error

#------------------------------------------------------------feature2vertex--------------------------------------------------------------------------------------

#------------------------------------------------------------other functions--------------------------------------------------------------------------------------

    def individual_dimension_vae(self, restore, datainfo):
        with tf.Session() as sess:
            self.saver.restore(sess, restore)
            path = os.path.split(restore)[0]
            random = np.zeros((len(self.feature), 200)).astype('float32')
            all_vae = 0
            def generate_embedding_input(_min, _max, dimension, rest):
                # x = np.zeros((25, self.hiddendim[vaeid])).astype('float32')
                x = np.zeros((25, len(rest))).astype('float32')
                # x_randn = np.zeros((25, self.hiddendim[vaeid])).astype('float32')

                for idx in xrange(0, self.hiddendim[vaeid]):
                    if idx == dimension:
                        # x[:, idx] = random_sample_range(_min[idx], _max[idx], num = 25)
                        x[:, idx] = np.linspace(_min[idx], _max[idx], num = 25)
                        print("dimension:%s, range:%s--%s" % (idx, _min[idx], _max[idx]))
                    else:
                        x[:, idx] = rest[idx]
                        # x_randn[:, idx] = rest[idx]

                return x#, x_randn

            for vaeid in range(0, self.num_vaes):
                if vaeid==0:
                    units_vae = 1
                else:
                    units_vae = self.hiddendim[vaeid-1]

                for vae_id in range(0, units_vae):
                    savepath = path + '/components_'+str(vaeid)+'_'+str(vae_id)
                    if not os.path.isdir(savepath):
                        os.mkdir(savepath)

                    dictbase = {self.inputs: self.feature}
                    dictrand = {x: random[:, 0: np.shape(x)[1]] for x in self.embedding_inputs}
                    feed_dict = merge_two_dicts(dictbase, dictrand)

                    embedding = sess.run([self.test_encode[all_vae]], feed_dict = feed_dict)[0]
                    # print(np.shape(embedding))

                    min_embedding = np.amin(embedding, axis = 0)

                    max_embedding = np.amax(embedding, axis = 0)

                    components_max = np.zeros((self.hiddendim[vaeid], self.pointnum, self.finaldim)).astype('float32')
                    components_min = np.zeros((self.hiddendim[vaeid], self.pointnum, self.finaldim)).astype('float32')

                    components_maxnorm = np.zeros(self.hiddendim[vaeid]).astype('float32')

                    components = np.zeros((2, self.pointnum, self.finaldim)).astype('float32')
                    for idx in xrange(0, self.hiddendim[vaeid]):
                        embedding_data = generate_embedding_input(min_embedding, max_embedding, idx, embedding[0, :])

                        recover = sess.run([self.embedding_decode[all_vae]], feed_dict = {self.embedding_inputs[all_vae]: embedding_data})[0]

                        deforma_v, deforma_v_align = sess.run([self.deform_vertex, self.deform_vertex_align], feed_dict={self.feature2point: recover, self.controlpoint: np.tile([0,0,0], (np.shape(recover)[0], 1, 1))})
                        deforma_v = self.alignallmodels(deforma_v, one = True)
                        self.v2objfile(deforma_v, savepath+'/25', [idx]*np.shape(deforma_v)[0], range(0, np.shape(deforma_v)[0]))
                        # self.v2objfile(deforma_v_align, path + '/components_'+str(vaeid)+'/25_align', [idx]*np.shape(deforma_v_align)[0], range(0, np.shape(deforma_v_align)[0]))

                        components_max[idx, :], components_min[idx, :], components_maxnorm[idx], recover_sub= self.select_components(recover[[0, 24]], datainfo)
                        if idx == 0:
                            components = recover_sub
                        else:
                            components = np.concatenate((components, recover_sub), axis = 0)

                        rs, rlogr = recover_data(recover, datainfo.logrmin, datainfo.logrmax, datainfo.smin, datainfo.smax, self.pointnum)
                        sio.savemat(savepath+'/25/dimension'+str(idx+1)+'.mat', {'RS':rs, 'RLOGR':rlogr})

                    sort_id = sorted(range(len(components_maxnorm)), key=lambda k: components_maxnorm[k], reverse = True)

                    components_max = components_max[sort_id]  # use this components
                    components_min = components_min[sort_id]

                    min_comp = [components[2*i+1] for i in sort_id]
                    max_comp = [components[2*i] for i in sort_id]

                    deforma_v, deforma_v_align = sess.run([self.deform_vertex, self.deform_vertex_align], feed_dict={self.feature2point: min_comp, self.controlpoint: np.tile([0,0,0], (np.shape(min_comp)[0], 1, 1))})
                    deforma_v = self.alignallmodels(deforma_v, one = True)
                    self.v2objfile(deforma_v, savepath + '/2', range(0, np.shape(deforma_v)[0]), sort_id, '2')
                    # self.v2objfile(deforma_v_align, path + '/components_'+str(vaeid)+'/2_align', range(0, np.shape(deforma_v_align)[0]), sort_id, '2')

                    deforma_v, deforma_v_align = sess.run([self.deform_vertex, self.deform_vertex_align], feed_dict={self.feature2point: max_comp, self.controlpoint: np.tile([0,0,0], (np.shape(max_comp)[0], 1, 1))})
                    deforma_v = self.alignallmodels(deforma_v, one = True)
                    self.v2objfile(deforma_v, savepath + '/2', range(0, np.shape(deforma_v)[0]), sort_id, '1')
                    # self.v2objfile(deforma_v_align, path + '/components_'+str(vaeid)+'/2_align', range(0, np.shape(deforma_v_align)[0]), sort_id, '1')

                    # components_max_rs, components_max_rlogr = recover_data(components_max, datainfo.logrmin, datainfo.logrmax, datainfo.smin, datainfo.smax, self.pointnum)
                    # components_min_rs, components_min_rlogr = recover_data(components_min, datainfo.logrmin, datainfo.logrmax, datainfo.smin, datainfo.smax, self.pointnum)

                    sio.savemat(savepath+'/2/components_max.mat', {'RS':components_max[:,:,3:self.vertex_dim], 'RLOGR':components_max[:,:,0:3]})
                    sio.savemat(savepath+'/2/components_min.mat', {'RS':components_min[:,:,3:self.vertex_dim], 'RLOGR':components_min[:,:,0:3]})

                    print(savepath)
                    all_vae+=1

    def random_individual_dimension(self, restore, datainfo):
        with tf.Session() as sess:
            self.saver.restore(sess, restore)
            path = os.path.split(restore)[0]
            if not os.path.isdir(path + '/randn_dimension'):
                os.mkdir(path + '/randn_dimension')
            if not os.path.isdir(path + '/randrange_dimension'):
                os.mkdir(path + '/randrange_dimension')

            embedding = sess.run([self.test_encode], feed_dict = {self.inputs: self.feature})[0]

            min_embedding = np.amin(embedding, axis = 0)

            max_embedding = np.amax(embedding, axis = 0)

            def generate_embedding_input(_min, _max, dimension, rest):
                x = np.zeros((25, self.hiddendim)).astype('float32')
                x_randn = np.zeros((25, self.hiddendim)).astype('float32')

                for idx in xrange(0, self.hiddendim):
                    if idx == dimension:
                        x[:, idx] = random_sample_range(_min[idx], _max[idx], num = 25)
                        x_randn[:, idx] = np.random.randn(25)
                        print("dimension:%s, range:%s--%s" % (idx, _min[idx], _max[idx]))
                    else:
                        x[:, idx] = rest[idx]
                        x_randn[:, idx] = rest[idx]

                return x, x_randn

            for idx in xrange(0, self.hiddendim):
                embedding_data_randrange, embedding_data_randn = generate_embedding_input(min_embedding, max_embedding, idx, embedding[0, :])

                recover_randrange = sess.run([self.embedding_decode], feed_dict = {self.embedding_inputs: embedding_data_randrange})[0]
                recover_randn = sess.run([self.embedding_decode], feed_dict = {self.embedding_inputs: embedding_data_randn})[0]
                rs, rlogr = recover_data(recover_randrange, datainfo.logrmin, datainfo.logrmax, datainfo.smin, datainfo.smax, self.pointnum)
                sio.savemat(path+'/randrange_dimension/randrange_dimension'+str(idx+1)+'.mat', {'RS':rs, 'RLOGR':rlogr})
                rs, rlogr = recover_data(recover_randn, datainfo.logrmin, datainfo.logrmax, datainfo.smin, datainfo.smax, self.pointnum)
                sio.savemat(path+'/randn_dimension/randn_dimension'+str(idx+1)+'.mat', {'RS':rs, 'RLOGR':rlogr})

            print(path)

    def synthesis(self, restore, datainfo, inputweight):
        with tf.Session() as sess:
            self.saver.restore(sess, restore)

            embedding = sess.run([self.test_encode], feed_dict = {self.inputs: self.feature})[0]

            rest = embedding[0,:]

            min_embedding = np.amin(embedding, axis = 0).reshape((hidden_dim, 1))

            max_embedding = np.amax(embedding, axis = 0).reshape((hidden_dim, 1))

            extreme_embedding = np.concatenate((min_embedding, max_embedding), axis = 1)

            direction = sio.loadmat('maxdirection.mat')
            direction = direction['maxdirection']

            eemb = np.zeros(self.hidden_dim)

            for i in xrange(0, self.hidden_dim):
                eemb[i] = extreme_embedding[i, direction[i]- 1]

            modelnum = len(inputweight)

            x = np.zeros((modelnum, self.hidden_dim))

            x[:,:] = rest

            for i in range(0, modelnum):
                for dim, weight in inputweight[i]:
                    x[i, dim] = rest[dim] + (eemb[dim] - rest[dim]) * weight

            recover = sess.run(self.embedding_decode, feed_dict = {self.embedding_inputs: x})[0]
            rs, rlogr = recover_data(recover, datainfo.logrmin, datainfo.logrmax, datainfo.smin, datainfo.smax, self.pointnum)
            sio.savemat('synthesis.mat', {'RS':rs, 'RLOGR':rlogr})

    def get_res_feature(self, restore, datainfo):
        with tf.Session() as sess:
            self.saver.restore(sess, restore)
            path = os.path.split(restore)[0]
            if not os.path.isdir(path + '/res_feature'):
                os.mkdir(path + '/res_feature')

            embedding = sess.run([self.test_encode], feed_dict = {self.inputs: self.feature})[0]

            min_embedding = np.amin(embedding, axis = 0)

            max_embedding = np.amax(embedding, axis = 0)

            def generate_embedding_input(_min, _max, dimension, rest):
                x = np.zeros((25, self.hiddendim)).astype('float32')

                for idx in xrange(0, self.hiddendim):
                    if idx == dimension:
                        x[:, idx] = np.linspace(_min[idx], _max[idx], num = 25)
                    else:
                        x[:, idx] = rest[idx]

                return x

            components_max = np.zeros((self.hiddendim, self.pointnum, self.finaldim)).astype('float32')
            components_min = np.zeros((self.hiddendim, self.pointnum, self.finaldim)).astype('float32')

            # components_max_rs = np.zeros((self.hiddendim, self.pointnum, 6)).astype('float32')
            # components_max_rlogr = np.zeros((self.hiddendim, self.pointnum, 3)).astype('float32')
            # components_min_rs = np.zeros((self.hiddendim, self.pointnum, 6)).astype('float32')
            # components_min_rlogr = np.zeros((self.hiddendim, self.pointnum, 3)).astype('float32')
            components_maxnorm = np.zeros(self.hiddendim).astype('float32')


            for idx in xrange(0, self.hiddendim):
                embedding_data = generate_embedding_input(min_embedding, max_embedding, idx, embedding[0, :])

                recover = sess.run([self.embedding_decode], feed_dict = {self.embedding_inputs: embedding_data})[0]
                # recover_s, recover_logr = recover_data(recover, datainfo.logrmin, datainfo.logrmax, datainfo.smin, datainfo.smax, self.pointnum)
                # # logrs = np.concatenate((rlogr, rs), axis = 2)
                # first_s, first_logr = recover_data(np.expand_dims(self.feature[0], axis=0), datainfo.logrmin, datainfo.logrmax, datainfo.smin, datainfo.smax, self.pointnum)

                # diff_s = recover_s - first_s
                # diff_logr = recover_logr - first_logr

                # diffs = np.sum(np.sum(np.power(diff_s, 2.0), axis=2), axis=1)
                # diffr = np.sum(np.sum(np.power(diff_logr, 2.0), axis=2), axis=1)

                # diff = diffs + diffr

                logrs = recover - self.feature[0]

                diff_rs, diff_rlogr = recover_data(logrs, datainfo.logrmin, datainfo.logrmax, datainfo.smin, datainfo.smax, self.pointnum)

                diff_logrs = np.concatenate((diff_rlogr, diff_rs), axis = 2)

                diff = np.sum(np.sum(np.power(diff_logrs, 2.0), axis=2), axis=1)

                min_deform = np.amin(diff, axis = 0)
                max_deform = np.amax(diff, axis = 0)
                components_maxnorm[idx] = max_deform
                # print(max_deform,"---",min_deform)

                min_deform_id = np.where(diff == min_deform)[0]
                max_deform_id = np.where(diff == max_deform)[0]
                # print(max_deform_id,"---",min_deform_id)

                # print(np.shape(diff_s),np.shape(first_s),np.shape(first_logr),np.shape(diff_logr))

                # components_max_rs[idx, :, :] = diff_s[max_deform_id] + first_s[0]
                # components_max_rlogr[idx, :, :] = diff_logr[max_deform_id] + first_logr[0]
                # print(components_max_rlogr[idx, 0,:])

                # components_min_rs[idx, :, :] = diff_s[min_deform_id] + first_s[0]
                # components_min_rlogr[idx, :, :] = diff_logr[min_deform_id] + first_logr[0]

                components_max[idx, :] = logrs[max_deform_id] + self.feature[0]
                components_min[idx, :] = logrs[min_deform_id] + self.feature[0]

            sort_id = sorted(range(len(components_maxnorm)), key=lambda k: components_maxnorm[k], reverse = True)

            print(np.shape(components_max))
            components_max = components_max[sort_id]  # use this components
            components_min = components_min[sort_id]
            print(sort_id)
            # components_max_rs = components_max_rs[sort_id]
            # components_max_rlogr = components_max_rlogr[sort_id]
            # print(components_max_rlogr[49, 0,:])

            # components_min_rs = components_min_rs[sort_id]
            # components_min_rlogr = components_min_rlogr[sort_id]

            components_max_rs, components_max_rlogr = recover_data(components_max, datainfo.logrmin, datainfo.logrmax, datainfo.smin, datainfo.smax, self.pointnum)
            components_min_rs, components_min_rlogr = recover_data(components_min, datainfo.logrmin, datainfo.logrmax, datainfo.smin, datainfo.smax, self.pointnum)
            print(np.shape(components_max_rs))

            if not os.path.isdir(path + '/python_component'):
                os.mkdir(path + '/python_component')

            sio.savemat(path+'/python_component/components_max.mat', {'RS':components_max_rs, 'RLOGR':components_max_rlogr})
            sio.savemat(path+'/python_component/components_min.mat', {'RS':components_min_rs, 'RLOGR':components_min_rlogr})

            # get the res deformation based the compoents
            def get_norm(fea):
                return np.sqrt(np.sum(np.power(fea, 2)))

            def get_res(fea, components):
                # print(type(components))
                base = np.array([0,0,0,1,0,0,1,0,1]).astype('float32')
                resultmin = -0.95
                resultmax = 0.95
                base[0:3] = (resultmax-resultmin)*(base[0:3] - datainfo.logrmin)/(datainfo.logrmax - datainfo.logrmin) + resultmin
                base[3:self.vertex_dim] = (resultmax-resultmin)*(base[3:self.vertex_dim] - datainfo.smin)/(datainfo.smax-datainfo.smin) + resultmin
                print(base)
                components = components - base
                fea = fea - base
                if len(np.shape(components)) == 3:
                    sum_project = np.zeros_like(fea).astype('float32')
                    for idx in xrange(0, len(components)):
                        sum_project += np.sum(fea * components[idx]) / (get_norm(components[idx])**2) * components[idx]

                    print(np.shape(sum_project))
                    print(np.shape(fea))
                    print(np.shape(base))

                    return fea - sum_project + base
                else:
                    assert(len(np.shape(components)) == 2)
                    project_on1comp = np.zeros_like(fea).astype('float32')
                    norm_comp = 1.0 / get_norm(components)**2 * components
                    for idx in xrange(0, len(fea)):
                        project_on1comp[idx] = np.sum(fea[idx] * components) * norm_comp

                    return fea - project_on1comp + base

            # get all the res feature based the components

            self.feature_res = np.zeros_like(self.feature).astype('float32')
            for idx in xrange(0, len(self.feature)):
                self.feature_res[idx, :] = get_res(self.feature[idx], components_max)

            res_rs, res_rlogr = recover_data(self.feature_res, datainfo.logrmin, datainfo.logrmax, datainfo.smin, datainfo.smax, self.pointnum)

            sio.savemat(path + '/res_feature/res_feature.mat', {'RS':res_rs, 'RLOGR':res_rlogr})

            # get the res feature base the every components

            for idx in xrange(0, len(components_max)):
                print(idx)
                feature_reson1comp = get_res(self.feature, components_max[idx])
                res_rson1comp, res_rlogron1comp = recover_data(feature_reson1comp, datainfo.logrmin, datainfo.logrmax, datainfo.smin, datainfo.smax, self.pointnum)
                sio.savemat(path+'/res_feature/res_feature'+str(idx+1)+'.mat', {'RS':res_rson1comp, 'RLOGR':res_rlogron1comp})
            print(path)

            return

#------------------------------------------------------------other functions--------------------------------------------------------------------------------------

