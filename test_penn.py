import torch
from PENN_dloader import *
import scipy.io
from eval_PCKh import *
from lossFunction import *
from modelZoo.K_FPN import *
from utils import get_recover_fista
from ptflops import get_model_complexity_info
import time
from utils import *
def test_val(net, testloader, alpha, epoch, Dict_pose_use, dataset_test, gpu_id):
    with torch.no_grad():
        T = Dict_pose_use.shape[0]
        keyFrames = []
        imgPath =[]
        numKey = 0
        Time = []
        # sample_num = len(testAnnot)
        sample_num = testloader.__len__()
        gtData = torch.zeros(sample_num, T, 13, 2)
        testData = torch.zeros(sample_num, T, 13, 2)
        VIS = torch.zeros(sample_num, T, 13)
        BBOX = torch.zeros(sample_num, T, 4)
        nFrames = torch.zeros(sample_num)
        randIdx = []
        for i, sample in enumerate(testloader):
            skeleton_to_use = sample['skeleton_to_use']
            vis_to_use = sample['vis_to_use']
            bbox_to_use = sample['bbox_to_use']
            imgSequence_to_use = sample['imgSequence_to_use']
            nframes = sample['nframes']
            idx = sample['randIdx']
            imgFolderPath = sample['imgPath']
            imgPath.append(imgFolderPath[0])
            randIdx.append(idx)


            img_data = imgSequence_to_use.squeeze(0)
            inputData = img_data.cuda(gpu_id)

            t0 = time.time()
            Dictionary, feature = net.forward(inputData)
            out = net.forward2(feature, alpha)

            s = out[0, :]
            key_ind = (s > 0.99).nonzero().squeeze(1)
            key_list = list(key_ind.cpu().numpy())

            keyFrames.append(key_list)
            numKey = numKey + len(key_list)
            skeletonData = skeleton_to_use[0].type(torch.FloatTensor).cuda(gpu_id)
            input = skeletonData.reshape(1, T, 13 * 2)  # Tx26

            if key_list == []:
                y_hat = torch.zeros(input.shape)
            else:
                y_hat = get_recover_fista(Dict_pose_use.cuda(gpu_id), input, key_list, gpu_id)
            endtime = time.time() - t0
            # print('time:', endtime)
            Time.append(endtime)

            test_gt = input.reshape(T, -1, 2).cpu()  # T x 13 x 2
            test_yhat = y_hat.reshape(T, -1, 2).cpu()  # T x 13 x 2
            test_gt_unnorm = dataset_test.unnormData(test_gt.unsqueeze(0))
            test_out_unnorm = dataset_test.unnormData(test_yhat.unsqueeze(0))

            gtData[i] = test_gt_unnorm
            testData[i] = test_out_unnorm
            VIS[i] = vis_to_use
            BBOX[i] = bbox_to_use
            nFrames[i] = nframes
        #
        with torch.cuda.device(gpu_id):
            flops, params = get_model_complexity_info(net, (3, 244, 244), as_strings=True, print_per_layer_stat=True)
            print('Flops:' + flops)
        #     print('Params:' + params)
        # print('time:', endtime)

        meanNumKey = numKey / sample_num  # 1068 test data

        get_PCKh_penn(gtData, testData, VIS, BBOX, nFrames, normTorso=True)
        get_PCKh_penn(gtData, testData, VIS, BBOX, nFrames, normTorso=False)
        print('epoch:', epoch, 'mean_keyframe:', meanNumKey, 'meantime:', statistics.mean(Time))
    # savefile:

    # data = {'keyframes': keyFrames, 'start':randIdx, 'imgPath': imgPath}
    # scipy.io.savemat('./PENN_keyframes_part3', mdict=data)


def random_select(keyframes, testSkeleton, Dictionary_pose, gpu_id):
    'testSkeleton is baseline skeleton'
    maxtIter = 100
    Y_hat = torch.zeros(maxtIter, T, 13, 2)
    L = len(testSkeleton)
    k = len(keyframes)
    Error = []
    for iter in range(0, maxtIter):
        keys = np.random.choice(L, k)

        y_hat = get_recover_fista(Dictionary_pose.cuda(gpu_id), testSkeleton, keys, gpu_id)
        Y_hat[iter] = y_hat.squeeze(0)
        error = torch.norm(testSkeleton-y_hat)
        Error.append(error)

    return Y_hat, Error


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    data_root = '/data/Yuexi/Penn_Action'
    T = 40
    gpu_id = 1
    trainAnnot, testAnnot = get_train_test_annot(data_root)
    dataset_test = pennDataset(trainAnnot, testAnnot, T, split='val')
    testloader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)

    modelPath = '/home/yuexi/Documents/keyFrameModel/RealData/PENN/resnet50/lam1_2_5_40/'
    # modelPath = '/path/to/your/model/folder/'
    modelFile = os.path.join(modelPath, '80.pth')
    state_dict = torch.load(modelFile)['state_dict']
    Drr = state_dict['Drr']
    Dtheta = state_dict['Dtheta']

    # net = keyframeProposalNet(numFrame=T, gpu_id=gpu_id, if_bn=True, if_init=True)
    net = keyframeProposalNet(numFrame=T, Drr=Drr, Dtheta=Dtheta, gpu_id=gpu_id)
    net.load_state_dict(state_dict)
    net.eval()

    net.cuda(gpu_id)

    dyanModelPath = './model/dyan_penn.pth'
    Dict_use_pose = getDictionary(dyanModelPath, T, gpu_id)

    alpha = 4
    epoch = 80
    test_val(net, testloader, alpha, epoch, Dict_pose, dataset_test, gpu_id)
    print('done')
    # random select a video

    # for iter in range(0, maxtIter):
        # y_hat = random_select(keyframes)