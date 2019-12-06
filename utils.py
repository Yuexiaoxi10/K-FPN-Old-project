import os
import numpy as np
import random
from math import isclose
from modelZoo.DyanOF import creatRealDictionary
import torch
import matplotlib.pyplot as plt
from modelZoo.DyanOF import OFModel, fista
from torch.autograd import Variable
import torch.nn


def gridRing(N):
    # epsilon_low = 0.25
    # epsilon_high = 0.15
    # rmin = (1 - epsilon_low)
    # rmax = (1 + epsilon_high)

    epsilon_low = 0.25
    epsilon_high = 0.15
    rmin = (1 - epsilon_low)
    rmax = (1 + epsilon_high)

    thetaMin = 0.001
    thetaMax = np.pi / 2 - 0.001
    delta = 0.001
    # Npole = int(N / 4)
    Npole = int(N/2)
    Pool = generateGridPoles(delta, rmin, rmax, thetaMin, thetaMax)
    M = len(Pool)

    idx = random.sample(range(0, M), Npole)
    P = Pool[idx]
    Pall = np.concatenate((P, -P, np.conjugate(P), np.conjugate(-P)), axis=0)

    return P, Pall


## Generate the grid on poles
def generateGridPoles(delta, rmin, rmax, thetaMin, thetaMax):
    rmin2 = pow(rmin, 2)
    rmax2 = pow(rmax, 2)
    xv = np.arange(-rmax, rmax, delta)
    x, y = np.meshgrid(xv, xv, sparse=False)
    mask = np.logical_and(np.logical_and(x ** 2 + y ** 2 >= rmin2, x ** 2 + y ** 2 <= rmax2),
                          np.logical_and(np.angle(x + 1j * y) >= thetaMin, np.angle(x + 1j * y) <= thetaMax))
    px = x[mask]
    py = y[mask]
    P = px + 1j * py

    return P

def plotting(input, reconstruction,key_frame, imageName, saveDir, seqNum):
    if seqNum > 1:

        seq1 = input[:, seqNum]
        seq2 = reconstruction[:, seqNum]
        y_key = -5 * np.ones(seq1.shape)
    else:
        seq1 = input
        seq2 = reconstruction
        y_key = -5 * np.ones(seq1.shape)

    y_key[key_frame,:] = seq1[key_frame,:]
    T = np.arange(0, input.shape[0],1)
    # plt.plot(T, seq1, 'b', T, seq2, 'r', T, y_key,'g*')
    plt.plot(T, seq1, 'b', label='gt')
    plt.plot(T, seq2, 'r', label='recover')
    plt.plot(T, y_key, 'g*', label='key frames')

    plt.legend()
    plt.title(imageName)
    plt.savefig(os.path.join(saveDir, imageName + '.png'))

def loadModel(ckpt_file, T, gpu_id):
    loadedcheckpoint = torch.load(ckpt_file, map_location=lambda storage, location: storage)
    #loadedcheckpoint = torch.load(ckpt_file)
    stateDict = loadedcheckpoint['state_dict']

    # load parameters
    Dtheta = stateDict['l1.theta']
    Drr    = stateDict['l1.rr']
    model = OFModel(Drr, Dtheta, T, gpu_id)
    model.cuda(gpu_id)

    return model

def get_Dictionary(T, numPole, gpu_id, addOne):

    P, Pall = gridRing(numPole)
    # Drr = np.zeros(1)
    # Dtheta = np.zeros(1)
    # P = 0.625 + 1j * 0.773
    # print(P)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float().cuda(gpu_id)
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float().cuda(gpu_id)

    WVar = []
    Wones = torch.ones(1).cuda(gpu_id)
    Wones = Variable(Wones, requires_grad=False)
    for i in range(0, T):
        W1 = torch.mul(torch.pow(Drr, i), torch.cos(i * Dtheta))

        W3 = torch.mul(torch.pow(Drr, i), torch.sin(i * Dtheta))
        if addOne:
            W = torch.cat((Wones, W1, W3), 0)
        else:
            W = torch.cat((W1, W3), 0)
        # W = torch.cat((W1, W3), 0)
        WVar.append(W.view(1, -1))
    dic = torch.cat((WVar), 0)
    G = torch.norm(dic, p=2, dim=0)
    idx = (G == 0).nonzero()
    nG = G.clone()
    nG[idx] = np.sqrt(T)
    G = nG

    dic = dic / G

    return dic

def get_recover(D, y, key_set):
    D_r = D[key_set, :]
    y_r = y[key_set, :]

    dtd_r = np.matmul(D_r, D_r.T)

    a = np.matmul(D_r.T, np.linalg.inv(dtd_r))
    coef_r = np.matmul(a, y_r)

    y_hat = np.matmul(D, coef_r)

    return y_hat

def get_recover_fista(D, y, key_set, gpu_id):
    if type(D) is np.ndarray:
        D = torch.Tensor(D)

    D_r = D[key_set]
    if len(y.shape)==3:
        y_r = y[:,key_set]
    else:
        y_r = y[key_set]

     # lam = 0.03

    if D.is_cuda:
        c_r = fista(D_r, y_r, 0.01, 100, gpu_id)
        y_hat = torch.matmul(D, c_r)
    else:
        c_r = fista(D_r.cuda(gpu_id), y_r, 0.01, 100, gpu_id)
        y_hat = torch.matmul(D.cuda(gpu_id), c_r)

    return y_hat

def getDictionary(ckpt_file, T, gpu_id):

    loadedcheckpoint = torch.load(ckpt_file, map_location=lambda storage, location: storage)
    stateDict = loadedcheckpoint['state_dict']

    Dtheta = stateDict['l1.theta'].cuda(gpu_id)
    Drr = stateDict['l1.rr'].cuda(gpu_id)
    Dictionary = creatRealDictionary(T, Drr, Dtheta, gpu_id)

    return Dictionary