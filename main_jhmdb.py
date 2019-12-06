
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
from utils import *
from JHMDB_dloader import *
import scipy.io
from eval_PCKh import *
from lossFunction import *
from test_jhmdb import test_val
import torch.nn as nn
from modelZoo.K_FPN import keyframeProposalNet, load_preTrained_model

torch.manual_seed(1)
random.seed(1)

T = 40  # length of video

dataRoot = '/data/Yuexi/JHMDB'
# dataRoot = '/path/to/your/data'
modelFolder = '/home/yuexi/Documents/keyFrameModel/RealData/JHMDB/resnet34/lam41_80'
# modelFolder = '/path/to/your/model/folder
if not os.path.exists(modelFolder):
	os.makedirs(modelFolder)

trainAnnot, testAnnot = get_train_test_annotation(dataRoot)
trainSet = jhmdbDataset(trainAnnot, testAnnot, T=T, split='train')
trainloader = DataLoader(trainSet, batch_size=1, shuffle=False, num_workers=8)

# testSet = jhmdbDataset(trainAnnot, testAnnot, T=T, split='test')
# testloader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=8)


valSet = jhmdbDataset(trainAnnot, testAnnot, T=T, split='val')
valloader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=8)

alpha = 4 # step size for sigmoid
gpu_id = 1

'get pose dictionary'
dyanModelPath = './models/dyan_jhmdb.pth'
Dict_use_pose = getDictionary(dyanModelPath, T, gpu_id)

numJoint = 15
N = 4 * 40
P, Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()

net = keyframeProposalNet(numFrame=T, Drr=Drr, Dtheta=Dtheta, gpu_id=gpu_id)

for m in net.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)

        nn.init.constant_(m.bias, 0)

resnet = models.resnet18(pretrained=True, progress=False)
net.modifiedResnet = load_preTrained_model(resnet, net.modifiedResnet)
net.cuda(gpu_id)

Epoch = 80

optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-8, weight_decay=0.001)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 70], gamma=0.1)
# iteration = 1000
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

        sequence_to_use = sample['sequence_to_use']  # already normalized
        img_data = sample['imgSequence_to_use']
        bbox = sample['Bbox_to_use']
        baseline_to_use = sample['baseline_to_use']

        # img_data = imgSequence_to_use[num]
        inputData = img_data[0].cuda(gpu_id)
        if len(inputData.shape) == 5:
            inputData = inputData.squeeze(0)
        else:
            inputData = inputData
        feature, Dictionary, imgFeature = net.forward(inputData)
        out = net.forward2(feature, alpha)
        # print('dict:', Dictionary.detach().cpu().numpy(), 'out:', out)

        # skeleton_data = skeleton_to_use[num].squeeze(0).type(torch.FloatTensor).cuda(gpu_id)
        loss, l1, l2, l3 = minInverse_loss(out, Dictionary, feature, T, [4, 1.8], gpu_id, 'jhmdb')

        loss.backward()

        # allNorm = []
        # for p in net.parameters():
        #     if p.grad is not None:
        #         allNorm.append(p.grad.data.norm(2))
        #
        # torch.nn.utils.clip_grad_norm_(net.parameters(), max(allNorm), norm_type=2)

        optimizer.step()
        lossVal.append(loss.data.item())
        loss1.append((l1.data.item()))
        loss2.append(l2.data.item())
        loss3.append(l3.data.item())

    loss_val = np.mean(np.array(lossVal))
    loss_1 = np.mean(np.array(loss1))
    loss_2 = np.mean(np.array(loss2))
    loss_3 = np.mean(np.array(loss3))
    # print('sequence:', i, 'output:', out)
    print('Epoch', epoch, '|loss : %.4f' % loss_val, '|sparse loss : %.4f' % loss_1,
          '|reconstruction loss : %.4f' % loss_2,
          '|0/1 loss: %.4f' % loss_3)

    Loss.append(loss_val)
    Loss1.append(loss_1)
    Loss2.append(loss_2)
    Loss3.append(loss_3)
    if if_val:
        print('start to validate:')
        test_val(net, valloader, epoch, alpha, Dict_use_pose, valSet, gpu_id)
    scheduler.step()
    print('check')
    # torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
    #             'optimizer': optimizer.state_dict()}, modelFolder + str(epoch) + '.pth')
print('done')