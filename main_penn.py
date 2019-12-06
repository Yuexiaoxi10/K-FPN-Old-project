import torch
import numpy as np
import os
import torchvision.models as models
from modelZoo.K_FPN import *
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
from utils import *
from PENN_dloader import *
import scipy.io
from eval_PCKh import *
from lossFunction import *
from test_penn import test_val

torch.manual_seed(0)
np.random.seed(0)
N = 4 * 40
P, Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()

modelFolder = '/home/yuexi/Documents/keyFrameModel/RealData/PENN/resnet50/lam1_2_5_40/'
# modelFolder = '/path/to/your/model'
if not os.path.exists(modelFolder):
	os.makedirs(modelFolder)

data_root = '/data/Yuexi/Penn_Action'
T = 40
gpu_id = 1
numJoint = 13
trainAnnot, testAnnot = get_train_test_annot(data_root)

dataset_train = pennDataset(trainAnnot, testAnnot, T, split='train')
# dataset_test = pennDataset(trainAnnot, testAnnot, T, split='test')
dataset_val = pennDataset(trainAnnot, testAnnot, T, split='val')

trainloader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=8)
# testloader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)
valloader = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=8)

'get pose dictionary'
dyanModelPath = './model/dyan_jhmdb.pth'
Dict_use_pose = getDictionary(dyanModelPath, T, gpu_id)

net = keyframeProposalNet(numFrame=T, Drr=Drr, Dtheta=Dtheta, gpu_id=gpu_id)

for m in net.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)

        nn.init.constant_(m.bias, 0)

resnet = models.resnet34(pretrained=True, progress=False)
net.modifiedResnet = load_preTrained_model(resnet, net.modifiedResnet)
net.cuda(gpu_id)

Epoch = 80

optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-8, momentum=0.9, weight_decay=0.001)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.1)
batchSize = 1
if_plot = False
if_val = True
Loss = []
Loss1 = []
Loss2 = []
Loss3 = []

alpha = 4
for epoch in range(1, Epoch+1):
    print('start training epoch:', epoch)
    # scheduler.step()
    lossVal = []
    loss1 = []
    loss2 = []
    loss3 = []

    for i, sample in enumerate(trainloader):
        optimizer.zero_grad()
        skeleton_to_use = sample['skeleton_to_use']
        vis_to_use = sample['vis_to_use']
        bbox_to_use = sample['bbox_to_use']
        imgSequence_to_use = sample['imgSequence_to_use']
        nframes = sample['nframes']
        mask_idx = sample['mask_idx']
        numSeq = imgSequence_to_use.shape[0]

        for num in range(0, numSeq):
            img_data = imgSequence_to_use[num]
            inputData = img_data.cuda(gpu_id)

            Dictionary, feature = net.forward(inputData)
            out = net.forward2(feature, alpha)
            loss, l1, l2, l3 = minInverse_loss(out, Dictionary, feature, T, [2.5, 4.5], gpu_id, 'penn')

            loss.backward()

            optimizer.step()
            lossVal.append(loss.data.item())
            loss1.append((l1.data.item()))
            loss2.append(l2.data.item())
            loss3.append(l3.data.item())

    scheduler.step()
    loss_val = np.mean(np.array(lossVal))
    loss_1 = np.mean(np.array(loss1))
    loss_2 = np.mean(np.array(loss2))
    loss_3 = np.mean(np.array(loss3))

    print('Epoch', epoch, '|loss : %.4f' % loss_val, '|sparse loss : %.4f' % loss_1,
          '|reconstruction loss : %.4f' % loss_2,
          '|0/1 loss: %.4f' % loss_3)

    Loss.append(loss_val)
    Loss1.append(loss_1)
    Loss2.append(loss_2)
    Loss3.append(loss_3)

    if if_val :
        print('doing validation:')
        test_val(net, valloader, alpha, epoch, Dict_use_pose, dataset_val, gpu_id)

    torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()}, modelFolder + str(epoch) + '.pth')

print('done')
