import time
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
from modelZoo.K_FPN import load_preTrained_model
import torch.nn as nn
from modelZoo.DyanOF import creatRealDictionary
from modelZoo.K_FPN import keyframeProposalNet
from utils import *
from JHMDB_dloader import *
import scipy.io
from eval_PCKh import *
from lossFunction import *
from ptflops import get_model_complexity_info
torch.manual_seed(0)
np.random.seed(0)

def test_val(net, testloader, epoch, alpha, Dictionary_pose, dataset_test, gpu_id):
    with torch.no_grad():
        T = Dictionary_pose.shape[0]
        keyFrames = []
        numKey = 0
        numJoint = 15
        sample_num = testloader.__len__()
        gtData = torch.zeros(sample_num, T, numJoint, 2)
        testData = torch.zeros(sample_num, T, numJoint, 2)

        imPath = []
        Time =[]
        BBOX = torch.zeros(sample_num, T, 4)
        nFrames = torch.zeros(sample_num)

        for i, sample in enumerate(testloader):
            print('testing sample:', i)

            sequence_to_use = sample['sequence_to_use']  # already normalized
            img_data = sample['imgSequence_to_use']
            bbox = sample['Bbox_to_use']

            nframes = sample['nframes']
            inputData = img_data[0].cuda(gpu_id)
            imagePath = sample['imgPath']
            imPath.append(imagePath)
            # baseline = sample['baseline']
            if len(inputData.shape) == 5:
                inputData = inputData.squeeze(0)
            else:
                inputData = inputData

            t0 = time.time()

            if nframes <= T:
                nframes = nframes


            feature , Dictionary,_ = net.forward(inputData)
            out = net.forward2(feature, alpha)

            # endtime = time.time() - t0
            # print('time:', endtime)

            s = out[0, :]
            key_ind = (s > 0.99).nonzero().squeeze(1)
            key_list = list(key_ind.cpu().numpy())
            # print('sample:', i, 'keyframes:', key_list)
            keyFrames.append(key_list)
            numKey = numKey + len(key_list)
            skeletonData = sequence_to_use[0].type(torch.FloatTensor).cuda(gpu_id)
            dim = 15*2
            GT = skeletonData.reshape(1, T, dim)  # Tx30
            if key_list == []:
                y_hat_gt = torch.zeros(GT.shape)

            else:
                y_hat_gt = get_recover_fista(Dictionary_pose.cuda(gpu_id), GT, key_list, gpu_id)

            endtime = time.time() - t0
            Time.append(endtime)
            print('time:', endtime)
            # get mpjpe
            test_gt = GT.squeeze(0).reshape(T, -1, 2).cpu()  # T x 13 x 2
            test_yhat_gt = y_hat_gt.squeeze(0).reshape(T, -1, 2).cpu()  # T x 13 x 2

            test_gt_unnorm = dataset_test.get_unNormalized_data(test_gt)
            test_out_unnorm = dataset_test.get_unNormalized_data(test_yhat_gt)

            gtData[i] = test_gt_unnorm
            testData[i] = test_out_unnorm

            BBOX[i] = bbox
            nFrames[i] = nframes
            # gtData.append(test_gt_unnorm)
            # testData.append(test_out_unnorm)
            # BBOX.append(bbox)
            # nFrames.append = nframes

        'baseline running time per frame: 0.0149/40 , based on Geforce 1080 ti'
        # totalTime = numKey * (0.0149/40) + statistics.mean(Time)*sample_num
        # print('time/fr ms:',  1000*(totalTime/T*sample_num))

        meanNumKey = numKey / sample_num  # 261 test data

        get_PCKh_jhmdb(gtData, testData, BBOX, nFrames, imPath, normTorso=False)
        get_PCKh_jhmdb(gtData, testData, BBOX, nFrames, imPath, normTorso=True)

        print('epoch:', epoch, 'mean_keyframe:', meanNumKey)
        with torch.cuda.device(gpu_id):
            flops, params = get_model_complexity_info(net, (3, 244, 244), as_strings=True, print_per_layer_stat=True)
            print('Flops:' + flops)
            # print('Params:' + params)


def random_select(keyframes, testSkeleton, Dictionary_pose, gpu_id):
    'testSkeleton is baseline skeleton'
    maxtIter = 100
    Y_hat = torch.zeros(maxtIter, T, 13, 2)
    L = len(testSkeleton)
    k = len(keyframes)

    for iter in range(0, maxtIter):
        keys = np.random.choice(L, k)

        y_hat = get_recover_fista(Dictionary_pose.cuda(gpu_id), testSkeleton, keys, gpu_id)
        Y_hat[iter] = y_hat

    return Y_hat

if __name__ == '__main__':
    data_root = '/data/Yuexi/JHMDB'
    # preTrainModel = '/home/yuexi/Documents/keyFrameModel/RealData/PENN/gcn_dyan/featureMaps_v2/9.pth'
    Data_to_use = scipy.io.loadmat('./testData/JHMDB_2DGauNorm_train_T40_DYAN.mat')
    T = 40
    numJoint = 15
    gpu_id = 1
    dyanModelPath = './model/dyan_jhmdb.pth'
    Dict_use_pose = getDictionary(dyanModelPath, T, gpu_id)

    modelFolder = '/home/yuexi/Documents/keyFrameModel/RealData/JHMDB/resnet18'
    # modelFolder = '/path/to/your/model/folder'
    trainAnnot, testAnnot = get_train_test_annotation(data_root)
    dataset_val = jhmdbDataset(trainAnnot, testAnnot, T, split='val')
    valloader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=8)

    modelFile = os.path.join(modelFolder, 'lam41_83.pth')
    state_dict = torch.load(modelFile)['state_dict']

    Drr = state_dict['Drr']
    Dtheta = state_dict['Dtheta']

    net = keyframeProposalNet(numFrame=T, Drr=Drr, Dtheta=Dtheta, gpu_id=gpu_id)
    newDict = net.state_dict()

    pre_dict = {k: v for k, v in state_dict.items() if k in newDict}

    newDict.update(pre_dict)

    net.load_state_dict(newDict)
    # net.load_state_dict(state_dict)
    net.cuda(gpu_id)
    epoch = 3
    alpha = 3

    test_val(net, valloader, epoch, alpha, Dict_use_pose.cuda(gpu_id), dataset_val, gpu_id)


    print('done')