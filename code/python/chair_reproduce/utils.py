import scipy.io as sio
import numpy as np
import random
import pickle
import configparser, argparse, os
import base64
# from zca import ZCA
import openmesh as om
import json
import h5py
import scipy
import glob

class Id:
    def __init__(self, Ia):
        self.Ia=Ia
    def show(self):
        print('A: %s\n'%(self.Ia))

class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        print ("values: {}".format(values))
        for kv in values:
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)

def load_datanew(path, binary = False, graphconv=False):
    resultmin = -0.95
    resultmax = 0.95

    data = h5py.File(path,mode = 'r')
    datalist = data.keys()

    logr = np.transpose(data['FLOGRNEW'], (3, 2, 1, 0)).astype('float32')
    s = np.transpose(data['FS'], (3, 2, 1, 0)).astype('float32')
    neighbour1 = np.transpose(data['neighbour']).astype('float32')
    cotweight1 = np.transpose(data['cotweight']).astype('float32')

    if graphconv and 'W1' in datalist:
        L = np.transpose(data['W1']).astype('float32')
        L1 = rescale_L(Laplacian(scipy.sparse.csr_matrix(L)))
    else:
        L1 = []

    nb1 = neighbour1
    cotw1 = cotweight1
    pointnum1 = neighbour1.shape[0]
    maxdegree1 = neighbour1.shape[1]
    modelnum = len(logr)

    degree1 = np.zeros((neighbour1.shape[0], 1)).astype('float32')
    for i in range(neighbour1.shape[0]):
        degree1[i] = np.count_nonzero(nb1[i])

    laplacian_matrix = np.transpose(data['L']).astype('float32')
    reconmatrix = np.transpose(data['recon']).astype('float32')
    vdiff = np.transpose(data['vdiff'], (2, 1, 0)).astype('float32')
    all_vertex = np.transpose(data['vertex'], (3, 2, 1, 0)).astype('float32')

    # sio.savemat('1.mat',{'logr':logr.astype('float64')})
    # f = np.zeros_like(logr).astype('float64')
    # f = logr
    # sio.savemat('2.mat', {'logr2': f})
    # demapping = np.transpose(data['demapping'])

    # degree = data['degrees']
    logrmin_set = []
    logrmax_set = []
    smin_set = []
    smax_set = []
    f = np.zeros((modelnum, logr.shape[1], pointnum1, 9)).astype('float32')
    for i in range(logr.shape[1]):
        logr_part = logr[:,i,:,:]
        s_part = s[:,i,:,:]

        logrmin = logr_part.min()
        logrmin = logrmin - 1e-6
        logrmax = logr_part.max()
        logrmax = logrmax + 1e-6
        smin = s_part.min()
        smin = smin - 1e-6
        smax = s_part.max()
        smax = smax + 1e-6

        logrmin_set.append(logrmin)
        logrmax_set.append(logrmax)
        smin_set.append(smin)
        smax_set.append(smax)

        rnew = (resultmax - resultmin) * (logr_part - logrmin) / (logrmax - logrmin) + resultmin
        snew = (resultmax - resultmin) * (s_part - smin) / (smax - smin) + resultmin

        print(np.shape(rnew))
        print(np.shape(snew))
        f[:,i,:,:] = np.concatenate((rnew, snew), axis=2)


    sym_feature = np.transpose(data['symmetryf'], (2, 1, 0)).astype('float32')

    partnum = sym_feature.shape[1]
    modelnum = sym_feature.shape[0]
    vertex_dim = sym_feature.shape[2]
    bbxmin_set = []
    bbxmax_set = []

    if binary:

        symmetry_feature = np.zeros_like(sym_feature).astype('float32')
        for i in range(partnum):
            # sym_part_f = sym_feature[:,i,:]
            binary_part_f = sym_feature[:,i,:(1+2*partnum)]
            bbx_center = sym_feature[:,i,(1+2*partnum):(1+2*partnum+3)]
            symmetry_exist = np.expand_dims(sym_feature[:,i,(1+2*partnum+3)], axis=1)
            symmetry_para = sym_feature[:,i,(1+2*partnum+3+1):]
            bbx_centermin = bbx_center.min() - 1e-6
            bbx_centermax = bbx_center.max() + 1e-6
            bbxmin_set.append(bbx_centermin)
            bbxmax_set.append(bbx_centermax)

            print(bbx_centermin)
            print(bbx_centermax)

            bbx_center = (resultmax-resultmin)*(bbx_center-bbx_centermin)/(bbx_centermax - bbx_centermin) + resultmin

            symmetry_feature[:,i,:]=np.concatenate([bbx_center, symmetry_para, binary_part_f, symmetry_exist], axis=1)
        sym_feature = symmetry_feature
    else:
        # symf = np.expand_dims(symf, axis=1)
        sym_feature_tmp = sym_feature
        sym_feature_tmp[np.where(sym_feature_tmp == 0)] = -1
        sym_feature_tmp[:,:,-4:]=sym_feature[:,:,-4:]
        sym_feature_tmp[:,:,-8:-5]=sym_feature[:,:,-8:-5]
        sym_feature = sym_feature_tmp
        # print(data['symmetryf'][1])
        # print(sym_feature[:,:,1])

        sym_featuremin = sym_feature.min() - 1e-6
        sym_featuremax = sym_feature.max() + 1e-6

        bbxmin_set.append(sym_featuremin)
        bbxmax_set.append(sym_featuremax)
        # print(data['symmetryf'][1])
        # print(sym_feature[:,:,1])
        # print(sym_featuremax)
        sym_feature = (resultmax-resultmin)*(sym_feature-sym_featuremin)/(sym_featuremax - sym_featuremin) + resultmin
        print(sym_featuremin)
        print(sym_featuremax)

    modelname = []
    for column in data['modelname']:
        row_data = []
        for row_number in range(len(column)):
            row_data.append(u''.join(map(chr, data[column[row_number]][:])))
        modelname.append(row_data[0])
    # modelname = np.squeeze(modelname)
    partname = data['partlist']['name'][0]
    partlist = []
    for i in range(len(partname)):
        partlist.append(''.join(chr(v) for v in data[partname[i]]))

    print(all_vertex.shape, vdiff.shape)
    print(logrmin)
    print(logrmax)
    print(smin)
    print(smax)
    # if graphconv and 'W2' in datalist:
    #     L = np.transpose(data['W2'])
    #     L2 = L
    #     L2 = rescale_L(Laplacian(scipy.sparse.csr_matrix(L2)))


    # cotw2 = np.zeros((cotweight2.shape[0], cotweight2.shape[1], 1)).astype('float64')
    # for i in range(1):
    #     cotw1[:, :, i] = cotweight1
    #     cotw2[:, :, i] = cotweight2

    # degree2 = np.zeros((neighbour2.shape[0], 1)).astype('float64')
    # for i in range(neighbour2.shape[0]):
    #     degree2[i] = np.count_nonzero(nb2[i])

    # mapping1 = np.zeros((pointnum2, mapping.shape[1])).astype('float64')
    # maxdemapping = np.zeros((pointnum1, 1)).astype('float64')

    # mapping1_col = mapping.shape[1]

    # mapping1 = mapping
    # # mapping2 = demapping
    # for i in range(pointnum1):
    #     # print i
    #     idx = np.where(mapping1 == i + 1)
    #     if idx[1][0] > 0:
    #         maxdemapping[i] = 1
    #     else:
    #         maxdemapping[i] = idx[0][0] + 1

    # meanpooling_degree = np.zeros((mapping.shape[0], 1)).astype('float64')
    # for i in range(mapping.shape[0]):
    #     meanpooling_degree[i] = np.count_nonzero(mapping1[i])

    # meandepooling_mapping = np.zeros((pointnum1, 1)).astype('float64')
    # meandepooling_degree = np.zeros((pointnum1, 1)).astype('float64')

    # for i in range(pointnum1):
    #     idx = np.where(mapping1 == i + 1)[0]
    #     meandepooling_mapping[i] = idx[0]
    #     meandepooling_degree[i] = meanpooling_degree[idx[0]]

    # degree = np.zeros((neighbour.shape[0], 1)).astype('float64')
    # for i in range(neighbour.shape[0]):
    #     degree[i] = np.count_nonzero(nb[i])

    return f, nb1, degree1, logrmin_set, logrmax_set, smin_set, smax_set, modelnum, pointnum1, maxdegree1, L1, cotw1, laplacian_matrix, reconmatrix, vdiff, all_vertex, sym_feature, bbxmin_set, bbxmax_set, modelname, partlist


def load_data(path, whitening = "False"):

    # data = sio.loadmat(path)
    # logr = data['FLOGRNEW']
    # s = data['FS']

    data = h5py.File(path,mode = 'r')
    logr = np.transpose(data['FLOGRNEW'], (2, 1, 0))
    s = np.transpose(data['FS'], (2, 1, 0))

    pointnum=logr.shape[1]
    print(whitening)
    base_s = np.array([1,0,0,1,0,1]).astype('float32')

    if whitening == "False":
        # s = s - base_s

        resultmax = 0.95
        resultmin = -0.95

        logrmin = logr.min()
        logrmin = logrmin - 1e-6
        logrmax = logr.max()
        logrmax = logrmax + 1e-6
        smin = s.min()
        smin = smin- 1e-6
        smax = s.max()
        smax = smax + 1e-6

        rnew = (resultmax-resultmin)*(logr-logrmin)/(logrmax - logrmin) + resultmin
        snew = (resultmax-resultmin)*(s - smin)/(smax-smin) + resultmin

        feature = np.concatenate((rnew,snew),axis = 2)

        f = np.zeros_like(feature).astype('float32')
        f = feature

        # base = f[0]
        # f = f - f[0]

        # print(base)

        return f, logrmin, logrmax, smin, smax, pointnum

    elif whitening == "zca":
        std_s = {}
        feature = np.concatenate((logr, s), axis = 2)
        feature = np.reshape(feature, [logr.shape[0], -1])
        trf = ZCA().fit(feature)
        feature_whitened = trf.transform(feature)
        f = np.reshape(feature_whitened, [logr.shape[0], pointnum, logr.shape[2] + s.shape[2]])

        logr = f[:,:,0:3]
        s = f[:,:,3:9]

        resultmax = 0.95
        resultmin = -0.95

        rmin = logr.min()
        rmin = rmin - 1e-6
        rmax = logr.max()
        rmax = rmax + 1e-6
        smin = s.min()
        smin = smin - 1e-6
        smax = s.max()
        smax = smax + 1e-6

        rnew = (resultmax-resultmin)*(logr-rmin)/(rmax - rmin) + resultmin
        snew = (resultmax-resultmin)*(s - smin)/(smax-smin) + resultmin

        feature = np.concatenate((rnew,snew),axis = 2)

        f = np.zeros_like(feature).astype('float32')
        f = feature

        std_s['rmin'] = rmin
        std_s['rmax'] = rmax
        std_s['smin'] = smin
        std_s['smax'] = smax

        a = 0

        return f, trf, std_s, a, a, pointnum

    else:
        # standardize
        info = {}
        mean_logr = np.mean(logr, axis=0)
        mean_s = np.mean(s, axis=0)
        std_logr = np.std(logr, axis=0)
        std_s = np.std(s, axis=0)

        rnew = (logr - mean_logr) / std_logr
        snew = (s - mean_s) / std_s

        resultmax = 0.95
        resultmin = -0.95

        rmin = rnew.min()
        rmin = rmin - 1e-6
        rmax = rnew.max()
        rmax = rmax + 1e-6
        smin = snew.min()
        smin = smin - 1e-6
        smax = snew.max()
        smax = smax + 1e-6

        rnew = (resultmax-resultmin)*(rnew - rmin)/(rmax - rmin) + resultmin
        snew = (resultmax-resultmin)*(snew - smin)/(smax - smin) + resultmin

        feature = np.concatenate((rnew,snew),axis = 2)

        f = np.zeros_like(feature).astype('float32')
        f = feature

        info['std_s'] = std_s
        info['rmin'] = rmin
        info['rmax'] = rmax
        info['smin'] = smin
        info['smax'] = smax
        rmin = rnew.min()
        rmin = rmin - 1e-6
        rmax = rnew.max()
        rmax = rmax + 1e-6
        smin = snew.min()
        smin = smin - 1e-6
        smax = snew.max()
        smax = smax + 1e-6
        print(rmin)
        print(rmax)
        print(smin)
        print(smax)

        return f, mean_logr, mean_s, std_logr, info, pointnum

def load_data_sym(path):
    resultmax = 0.95
    resultmin = -0.95

    data = h5py.File(path,mode = 'r')
    datalist = data.keys()

    sym_feature = np.transpose(data['symmetryf']).astype('float32')
    # print(data['symmetryf'][1])
    # print(sym_feature[:,:,1])

    partnum = sym_feature.shape[1]//5
    modelnum = sym_feature.shape[0]
    vertex_dim = 5
    # print(data['symmetryf'][1])
    # print(sym_feature[:,:,1])

    sym_featuremin = sym_feature.min() - 1e-6
    sym_featuremax = sym_feature.max() + 1e-6
    # print(data['symmetryf'][1])
    # print(sym_feature[:,:,1])
    # print(sym_featuremax)
    sym_feature = (resultmax-resultmin)*(sym_feature-sym_featuremin)/(sym_featuremax - sym_featuremin) + resultmin

    print(sym_featuremin)
    print(sym_featuremax)

    return sym_feature, partnum, vertex_dim, modelnum, sym_featuremin, sym_featuremax

def load_neighbour(path, pointnum):
    # data = sio.loadmat(path)
    # neighbour_ = data['neighbour']

    data = h5py.File(path,mode = 'r')

    neighbour_ = np.transpose(data['neighbour']).astype('float32')
    laplacian_matrix = np.transpose(data['L']).astype('float32')
    reconmatrix = np.transpose(data['recon']).astype('float32')
    vdiff = np.transpose(data['vdiff'], (2, 1, 0)).astype('float32')
    all_vertex = np.transpose(data['vertex'], (2, 1, 0)).astype('float32')
    print(all_vertex.shape, vdiff.shape)

    maxdegree = neighbour_.shape[1]

    # neighbour = np.zeros((pointnum, maxdegree)).astype('float32')
    neighbour = neighbour_

    degree = np.zeros((neighbour.shape[0], 1)).astype('float32')

    for i in range(neighbour.shape[0]):
        degree[i] = np.count_nonzero(neighbour[i])

    # laplacian_matrix = data['L'].astype('float32')

    # reconmatrix = data['recon'].astype('float32')

    # vdiff = data['vdiff'].astype('float32')

    # control_id = data['controlid'].astype('int32')

    # all_vertex = data['vertex'].astype('float32')

    return neighbour, degree, maxdegree, laplacian_matrix, reconmatrix, vdiff, all_vertex

def load_geodesic_weight(path, pointnum):

    # data = sio.loadmat(path)
    # data = data['distance']

    data = h5py.File(path,mode = 'r')
    data = np.transpose(data['distance'])

    distance = np.zeros((pointnum, pointnum)).astype('float32')
    distance = data/data.max()

    return distance

def recover_datasym(recover_feature, sym_featuremin, sym_featuremax):
    resultmax = 0.95
    resultmin = -0.95

    f = (sym_featuremax - sym_featuremin) * (recover_feature - resultmin) / (resultmax - resultmin) + sym_featuremin

    ftmp = f
    ftmp=np.round(ftmp)
    ftmp[np.where(ftmp==-1)]=0
    ftmp[:,:,-8:-5] = f[:,:,-8:-5]
    f = ftmp

    return f

def recover_datasymv2(recover_feature, bbx_centermin, bbx_centermax): # recover for every part
    resultmax = 0.95
    resultmin = -0.95
    modelnum = recover_feature.shape[0]
    partnum = (recover_feature.shape[1]-9)//2


    bbx_center = recover_feature[:,:3]
    symmetry_para = recover_feature[:,3:7]
    binary_part_f = recover_feature[:,7:(8+2*partnum)]
    symmetry_exist = np.expand_dims(recover_feature[:,-1], axis=1)
    bbx_center = (bbx_centermax - bbx_centermin) * (bbx_center - resultmin) / (resultmax - resultmin) + bbx_centermin

    f = np.concatenate([binary_part_f, bbx_center, symmetry_exist, symmetry_para], axis=1)

    return f

def recover_data(recover_feature, logrmin, logrmax, smin, smax, pointnum):
    # print(base)
    # recover_feature = recover_feature + base
    logr = recover_feature[:,:,0:3]
    s = recover_feature[:,:,3:9]
    base_s = np.array([1,0,0,1,0,1]).astype('float32')

    if isinstance(logrmin, np.float32) or isinstance(logrmin, np.float64):
        # print('yangji')

        resultmax = 0.95
        resultmin = -0.95

        s = (smax - smin) * (s - resultmin) / (resultmax - resultmin) + smin
        logr = (logrmax - logrmin) * (logr - resultmin) / (resultmax - resultmin) + logrmin

        # s = s + base_s

    elif isinstance(logrmin, np.ndarray):
        # mean_logr, mean_s, std_logr, std_s
        resultmax = 0.95
        resultmin = -0.95

        r_min = smax['rmin']
        r_max = smax['rmax']
        s_min = smax['smin']
        s_max = smax['smax']

        s = (s_max - s_min) * (s - resultmin) / (resultmax - resultmin) + s_min
        logr = (r_max - r_min) * (logr - resultmin) / (resultmax - resultmin) + r_min

        logr = logr * smin + logrmin
        s = s * smax['std_s'] + logrmax

    else:
        assert(isinstance(logrmin, ZCA))

        resultmax = 0.95
        resultmin = -0.95

        r_min = logrmax['rmin']
        r_max = logrmax['rmax']
        s_min = logrmax['smin']
        s_max = logrmax['smax']

        s = (s_max - s_min) * (s - resultmin) / (resultmax - resultmin) + s_min
        logr = (r_max - r_min) * (logr - resultmin) / (resultmax - resultmin) + r_min

        feature = np.concatenate((logr, s), axis = 2)

        f = np.zeros_like(feature).astype('float32')
        f = feature

        X_reconstructed = logmin.inverse_transform(np.reshape(f, [f.shape[0], -1]))
        X_reconstructed = np.reshape(X_reconstructed, [f.shape[0], pointnum, 9])

        logr = X_reconstructed[:,:,0:3]
        s = X_reconstructed[:,:,3:9]


    return s, logr

def gaussian(batch_size, n_dim, mean=0.0, var=1.0, n_labels=10, use_label_info=False):
    if use_label_info:
        if n_dim != 2:
            raise Exception("n_dim must be 2.")

        def sample(n_labels):
            x, y = np.random.normal(mean, var, (2,))
            angle = np.angle((x - mean) + 1j * (y - mean), deg=True)

            label = (int(n_labels * angle)) // 360

            if label < 0:
                label += n_labels

            return np.array([x, y]).reshape((2,)), label

        z = np.empty((batch_size, n_dim), dtype=np.float32)
        z_id = np.empty((batch_size, 1), dtype=np.int32)
        for batch in range(batch_size):
            for zi in range(int(n_dim / 2)):
                a_sample, a_label = sample(n_labels)
                z[batch, zi * 2:zi * 2 + 2] = a_sample
                z_id[batch] = a_label
        return z, z_id
    else:
        z = np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)
        return z

def spilt_dataset(num, percent_to_train, name="id.dat"):

    if os.path.isfile(name):
        id = pickle.load(open(name, 'rb'))
        id.show()
        Ia = id.Ia
    else:
        Ia = np.arange(num)
        Ia = random.sample(list(Ia), int(num * percent_to_train))

        id = Id(Ia)
        f = open(name, 'wb')
        pickle.dump(id, f, 0)
        f.close()
        id.show()

    Ia_C=list(set(np.arange(num)).difference(set(Ia)))

    return Ia, Ia_C

def printout(flog, data, epoch=0, interval = 50, write_to_file = True):
    # interval = 50
    if epoch % interval==0:
        print(data)
        flog.write(str((data + '\n')*write_to_file))

def argpaser2file(args, name='example.ini'):
    d = args.__dict__
    cfpar = configparser.ConfigParser()
    cfpar['default'] = {}
    for key in sorted(d.keys()):
        cfpar['default'][str(key)]=str(d[key])
        print('%s = %s'%(key,d[key]))

    with open(name, 'w') as configfile:
        cfpar.write(configfile)

def inifile2args(args, ininame='example.ini'):

    config = configparser.ConfigParser()
    config.read(ininame)
    defaults = config['default']
    result = dict(defaults)
    # print(result)
    # print('\n')
    # print(args)
    args1 = vars(args)
    # print(args1)

    args1.update({k: v for k, v in result.items() if v is not None})  # Update if v is not None

    # print(args1)
    args.__dict__.update(args1)

    # print(args)

    return args

def getFileName(path, postfix = '.ini'):
    ''' 获取指定目录下的所有指定后缀的文件名 '''
    filelist =[]
    f_list = os.listdir(path)
    # print f_list
    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] == postfix:
            print("[{}] {}".format(f_list.index(i),i))
            filelist.append(i)

    return filelist


def random_sample_range(_min, _max, num=25):
    rng = np.random.RandomState(12345)
    x = np.random.rand(num)
    x = x * (_max - _min) + _min
    rng.shuffle(x)

    return x

def safe_b64encode(s):
    s_bytes = s.encode("utf-8")
    return (base64.b64encode(s_bytes).decode("utf-8"))

def safe_b64decode(s):
    #length = len(s) % 4
    #for i in range(length):
        #s = s + '='
    return base64.b64decode(s).decode("utf-8")

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def readmesh(objpath):
    # mesh = om.TriMesh()
    # mesh.request_halfedge_normals()
    # mesh.request_vertex_normals()
    # result =
    return om.read_trimesh(objpath)

def savemesh(mesh, objpath, newv):
    # get all points of the mesh
    point_array = mesh.points()
    # print(np.shape(point_array))

    # translate the mesh along the x-axis
    for vh in mesh.vertices():
        # print(vh.idx())
        point_array[vh.idx()] = newv[vh.idx()]
    # point_array = newv
    # point_array += np.array([1, 0, 0])

    # write and read meshes
    om.write_mesh(objpath, mesh)

def get_batch_data(data1, data2, batch_size):
    data_num = len(data1)
    if data_num == 0:
        return data1, data2

    import math
    if data_num < batch_size:
        if batch_size//data_num>1:
        # remainder = batchsize - data_num
            reminder = math.pow(2, math.ceil(math.log(data_num, 2))) - data_num
        else:
            reminder = batch_size - data_num
    else:
        reminder = batch_size-(data_num%batch_size)

    Ia = np.arange(data_num)
    Ia = random.sample(list(Ia), int(reminder))
    data1 = np.concatenate((data1, data1[Ia]), axis = 0)
    data2 = np.concatenate((data2, data2[Ia]), axis = 0)
    return data1, data2

class TimeLiner:
    _timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)



def traversalDir_FirstDir(path, perfix = ''):
    # os.path.getmtime() 函数是获取文件最后修改时间
    # os.path.getctime() 函数是获取文件最后创建时间
    dir_list = []
    if (os.path.exists(path)):
        files = glob.glob(path + '/' + perfix + '*' )
        for file in files:
            #判断该路径下是否是文件夹
            if (os.path.isdir(file)):
                #分成路径和文件的二元元组
                h = os.path.split(file)
                dir_list.append(h[1])
        dir_list = sorted(dir_list,  key=lambda x: os.path.getmtime(os.path.join(path, x)), reverse = True)
        return dir_list

# ---------------------------------------------------graph conv------------------------------------------------

def Laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)
    # d=d.astype(W.dtype)
    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        # d += np.spacing(np.array(0.0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr_matrix
    return L


def rescale_L(L, lmax=2):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L /= lmax / 2
    L -= I
    return L


# def chebyshev(L, X, K):
#     """Return T_k X where T_k are the Chebyshev polynomials of order up to K.
#     Complexity is O(KMN)."""
#     M, N = X.shape
#     # assert L.dtype == X.dtype

#     # L = rescale_L(L, lmax)
#     # Xt = T @ X: MxM @ MxN.
#     Xt = np.empty((K, M, N), L.dtype)
#     # Xt_0 = T_0 X = I X = X.
#     Xt[0, ...] = X
#     # Xt_1 = T_1 X = L X.
#     if K > 1:
#         Xt[1, ...] = L.dot(X)
#     # Xt_k = 2 L Xt_k-1 - Xt_k-2.
#     for k in range(2, K):
#         Xt[k, ...] = 2 * L.dot(Xt[k - 1, ...]) - Xt[k - 2, ...]
#     return Xt

