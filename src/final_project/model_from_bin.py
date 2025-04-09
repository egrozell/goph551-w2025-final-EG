import numpy as np
from matplotlib import pyplot as plt
from examples.seismic.model import SeismicModel
import os


def model(directory, **kwargs):
    space_order = kwargs.pop('space_order', 2)
    h, w = get_dims(directory)
    shape = kwargs.pop('shape', (w, h))
    spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
    origin = kwargs.pop('origin', tuple([0. for _ in shape]))
    nbl = kwargs.pop('nbl', 10)
    dtype = kwargs.pop('dtype', np.float32)

    file = find_files(directory)
    properties, _, _ = read_bin(directory)

    vp = properties[file.index('vp')] * 1e-3
    vs = properties[file.index('vs')] * 1e-3
    rho = properties[file.index('rho')] / 5
    print(rho)
    return SeismicModel(space_order=space_order, vp=vp, vs=vs, b=rho,
                        origin=origin, shape=shape,
                        dtype=dtype, spacing=spacing, nbl=nbl, **kwargs)


def get_max_min(directory):
    file = find_files(directory)
    _, max, min = read_bin(directory)
    vp_max = max[file.index('vp')]
    vp_min = min[file.index('vp')]
    vs_max = max[file.index('vs')]
    vs_min = min[file.index('vs')]
    rho_max = max[file.index('rho')]
    rho_min = min[file.index('rho')]

    return vp_min, vp_max, vs_min, vs_max, rho_min, rho_max


def find_files(directory):
    files = os.listdir(directory)
    file_l = []
    for file in files:
        if file.endswith(".txt"):
            file_l.append(file[0:-4])
    return file_l


def labels(directory):
    file = find_files(directory)
    label_l = []
    for i in range(len(file)):
        with open(directory + file[i] + '.txt') as f:
            lines = f.readlines()
            label = []
            for line in lines[3:-1]:
                ni = line.find("\"") + 1
                ne = line.find("\"", ni)
                ni1 = line.find("\"", ne + 1) + 1
                ne1 = line.find("\"", ni1)
                label.append(line[ni:ne] + " " + line[ni1:ne1])
            label_l.append(label)
    return label_l


def get_dims(directory):
    """
    input
    ______
    directory: the location of the directory holding the asci data

    output
    ______
    property_list:

    """
    file = find_files(directory)
    for i in range(len(file)):
        with open(directory + file[i] + '.txt') as f:
            lines = f.readlines()
            h = 0
            w = 0
            dim = []
            for line in lines[3:5]:
                ni = line.find("=") + 1
                ne = line.find(" ", ni)
                dim.append(int(line[ni:ne]))
            h = int((dim[0] - 329))
            w = int((dim[1] - 3629))

    return h, w


def read_bin(directory):
    file = find_files(directory)
    h, w = get_dims(directory)
    print(f'dims {w} w, {h} h')
    h1 = int((h) * 1 + 329)
    w1 = int((w) * 1 + 3629)
    print(f'dims {w1} w, {h1} h')
    prop = []
    max = []
    min = []
    print(file)
    for i in range(len(file)):
        with open(directory + file[i] + '.bin', 'rb') as f:
            values = np.fromfile(f, dtype=np.float32, count=h1 * w1).reshape(w1, h1)
        smaller = values[2000:-1629, 329:]
        even_smaller = smaller[::1, ::1]
        # odd_small = even_smaller[0:-1, 0:-1]
        prop.append(even_smaller)
        max.append(np.max(prop[i]))
        min.append(np.min(prop[i]))
        plt.hist(np.reshape(prop[i], w * h, order='F'), bins=np.linspace(min[i], max[i], 200))
        plt.savefig('./figures/' + file[i] + '.png')
        plt.close()
    return prop, max, min
