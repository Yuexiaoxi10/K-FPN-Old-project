#from https://github.com/bearpaw/pytorch-pose/blob/master/evaluation/eval_PCKh.py
import sys
from scipy.io import loadmat
from numpy import transpose
# import skimage.io as sio
import numpy as np
import os
from h5py import File
import torch


def get_PCKh_penn(Test_gt, Test_out, Visbility, Bbox, nFrames, normTorso):
    # adopted code from : https://github.com/lawy623/LSTM_Pose_Machines/blob/master/testing/src/run_benchmark_GPU_PENN.m
    # Penn Action Official Joints Info, Menglong
    # 0.  head
    # 1.  left_shoulder  2.  right_shoulder
    # 3.  left_elbow     4.  right_elbow
    # 5.  left_wrist     6.  right_wrist
    # 7.  left_hip       8.  right_hip
    # 9.  left_knee      10. right_knee
    # 11. left_ankle     12. right_ankle

    # orderToPENN = [0 2 5 4 7 5 8 9 12 10 13 11 14];
    gtJointOrder = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    thresh = 0.2


    # torso_norm = 1 # 1: Torso / 0:bbx; default as 0 -> 0.2*max(h,w)
    sample_num = Test_gt.shape[0]

    HitPoint = np.zeros((sample_num, len(gtJointOrder)))
    visible_joint = np.zeros((sample_num, len(gtJointOrder)))

    for sample in range(0, sample_num):
        # print('test sample:', sample)
        test_gt = Test_gt[sample]
        test_out = Test_out[sample]
        visibility = Visbility[sample]
        bbox = Bbox[sample]
        nframes = nFrames[sample].int()

        if nframes >= test_gt.shape[0]:
            nfr = test_gt.shape[0]
        else:
            nfr = nframes

        # num_frame = test_gt.shape[0]
        seqError = torch.zeros(nfr, len(gtJointOrder))

        seqThresh = torch.zeros(nfr, len(gtJointOrder))
        for frame in range(0, nfr):
            gt = test_gt[frame] # 13x2
            pred = test_out[frame] # 13x2
            # vis = visibility[frame] # 1x13

            if normTorso:
                bodysize = torch.norm(gt[2] - gt[7])
                if bodysize < 1:
                    bodysize = torch.norm(pred[2] - pred[7])
            else:
                bodysize = torch.max(bbox[frame,2]-bbox[frame, 0], bbox[frame, 3] - bbox[frame, 1])


            error_dis = torch.norm(gt-pred, p=2, dim=1, keepdim=False)

            seqError[frame] = error_dis
            seqThresh[frame] = (bodysize*thresh) * torch.ones(len(gtJointOrder))

        vis = visibility[0:nfr]
        visible_joint[sample] = np.sum(vis.numpy(), axis=0)
        less_than_thresh = np.multiply(seqError.numpy()<=seqThresh.numpy(), vis.numpy())
        # visibleJoint = np.sum(visibility.numpy(), axis=0)
        HitPoint[sample] = np.sum(less_than_thresh, axis=0)

    finalPCK = np.divide(np.sum(HitPoint, axis=0), np.sum(visible_joint, axis=0))
    finalMean = np.mean(finalPCK)
    print('normTorso,    Head,      Shoulder,   Elbow,    Wrist,     Hip,     Knee,    Ankle,  Mean')
    print('{:5s}        {:.4f}      {:.4f}     {:.4f}     {:.4f}      {:.4f}    {:.4f}    {:.4f}   {:.4f}'.format(str(normTorso),
          finalPCK[0], 0.5*(finalPCK[1]+finalPCK[2]), 0.5*(finalPCK[3]+finalPCK[4]), 0.5*(finalPCK[5]+finalPCK[6]),
          0.5*(finalPCK[7]+finalPCK[8]), 0.5*(finalPCK[9]+finalPCK[10]),  0.5*(finalPCK[11]+finalPCK[12]), finalMean))

    return finalMean, finalPCK


def get_PCKh_jhmdb(Test_gt, Test_out, Bbox, nFrames,imgPath ,normTorso):

    # 0: neck    1:belly   2: face
    # 3: right shoulder  4: left shoulder
    # 5: right hip       6: left hip
    # 7: right elbow     8: left elbow
    # 9: right knee      10: left knee
    # 11: right wrist    12: left wrist
    # 13: right ankle    14: left ankle

    orderJHMDB = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    # partJHMDB = 7
    thresh = 0.2
    N = Test_out.shape[1]
    if normTorso:
        torso_norm = 1 # 1: Torso / 0:bbx; default as 0 -> 0.2*max(h,w)
    else:
        torso_norm = 0
    sample_num = Test_gt.shape[0]

    HitPoint = np.zeros((sample_num, len(orderJHMDB)))
    allPoint = np.ones((sample_num,  N, len(orderJHMDB)))
    Point_to_use = np.ones((sample_num, len(orderJHMDB)))

    for sample in range(0, sample_num):
        # print('test sample:', sample)
        test_gt = Test_gt[sample]
        test_out = Test_out[sample]
        nframes = nFrames[sample]
        img_path = imgPath[sample]
        bbox = Bbox[sample]

        # num_frame = test_gt.shape[0]
        if nframes >= test_gt.shape[0]:
            nfr = test_gt.shape[0]
        else:
            nfr = nframes.int()

        seqError = torch.zeros(nfr,  len(orderJHMDB))

        seqThresh = torch.zeros(nfr,  len(orderJHMDB))
        for frame in range(0, nfr):
            gt = test_gt[frame] # 13x2
            pred = test_out[frame] # 13x2
            # vis = visibility[frame] # 1x13

            if torso_norm == 1:
                bodysize = torch.norm(gt[4] - gt[5])
                if bodysize < 1:
                    bodysize = torch.norm(pred[4] - pred[5])
            else:
                bodysize = torch.max(bbox[frame, 2]-bbox[frame, 0], bbox[frame, 3] - bbox[frame, 1])

            error_dis = torch.norm(gt-pred, p=2, dim=1, keepdim=False)

            seqError[frame] = torch.FloatTensor(error_dis)
            seqThresh[frame] = (bodysize*thresh) * torch.ones(len(orderJHMDB))

        pts = allPoint[sample, 0:nfr]
        Point_to_use[sample] = np.sum(pts, axis=0)

        less_than_thresh = seqError.numpy()<=seqThresh.numpy()
        HitPoint[sample] = np.sum(less_than_thresh, axis=0)

    finalPCK = np.divide(np.sum(HitPoint, axis=0), np.sum(Point_to_use, axis=0))
    finalMean = np.mean(finalPCK)

    print('{:5s}          {:.4f}      {:.4f}      {:.4f}     {:.4f}     {:.4f}    {:.4f}    {:.4f}   {:.4f}'.format(str(normTorso), finalPCK[2],
          0.5*(finalPCK[3]+finalPCK[4]), 0.5*(finalPCK[7]+finalPCK[8]), 0.5*(finalPCK[11]+finalPCK[12]),
          0.5*(finalPCK[5]+finalPCK[6]), 0.5*(finalPCK[9]+finalPCK[10]),  0.5*(finalPCK[13]+finalPCK[14]), finalMean))

    # return finalMean, finalPCK

if __name__ == '__main__':
    'toy example'
    test_gt = torch.randn(10, 50, 13, 2)
    test_out = torch.randn(10, 50, 13, 2)
    visibility = torch.randint(0,2,(10, 50, 13))
    bbox = torch.randn(10, 50, 4)
    nFrames = torch.randint(10, 70, (10,))
    get_PCKh_penn(test_gt, test_out, visibility, bbox, nFrames, normTorso=False)

    print('ok')