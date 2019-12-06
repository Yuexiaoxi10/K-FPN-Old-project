import torch
# torch.manual_seed(0)
import torch.nn as nn
import torchvision.models as models

from modelZoo.resNet import ResNet, Bottleneck, BasicBlock
from modelZoo.DyanOF import OFModel, creatRealDictionary
from utils import generateGridPoles, gridRing
import numpy as np

def load_preTrained_model(pretrained, newModel):
    'load pretrained resnet 50 to self defined model '
    'modified resnet has no last two layers, only return feature map'
    # resnet101 = models.resnet101(pretrained=True, progress=False)
    pre_dict = pretrained.state_dict()

    # modifiedResnet = ResNet(block=Bottleneck, layers=[3, 4, 23, 3], zero_init_residual=False,
    #              groups=1, width_per_group=64, replace_stride_with_dilation=None,
    #              norm_layer=None)

    new_dict = newModel.state_dict()

    pre_dict = {k: v for k, v in pre_dict.items() if k in new_dict}

    new_dict.update(pre_dict)

    newModel.load_state_dict(new_dict)

    for param in newModel.parameters():
        param.requires_grad = False

    return newModel


class keyframeProposalNet(nn.Module):
    def __init__(self, numFrame, Drr, Dtheta, gpu_id):
        super(keyframeProposalNet, self).__init__()
        self.num_frame = numFrame
        self.gpu_id = gpu_id

        # self.modifiedResnet = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], zero_init_residual=False,
        #                          groups=1, width_per_group=64, replace_stride_with_dilation=None,
        #                          norm_layer=None)

        self.modifiedResnet = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], zero_init_residual=False,
                               groups=1, width_per_group=64, replace_stride_with_dilation=None,
                               norm_layer=None)
        """""
        self.Conv2d = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        """""
        self.relu = nn.LeakyReLU(inplace=True)

        # self.convLSTM = ConvLSTM(input_size=(7, 7), input_dim=512, hidden_dim=[256, 128, 64],
        #                          kernel_size=(3, 3), num_layers=3, gpu_id=self.gpu_id, batch_first=True, bias=True,
        #                          return_all_layers=False)
        'reduce feature map'
        self.layer2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1)
        # self.layer2 = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1)
        self.bn_l2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1)
        self.bn_l3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4= nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.bn_l4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # self.convBILSTM = ConvBILSTM(input_size=(7, 7), input_dim=512, hidden_dim=[256, 128, 64],
        #                          kernel_size=(3, 3), num_layers=3, gpu_id=self.gpu_id, batch_first=True, bias=True,
        #                          return_all_layers=False)
        self.Drr = nn.Parameter(Drr, requires_grad=True)
        self.Dtheta = nn.Parameter(Dtheta, requires_grad=True)

        # self.DYAN = OFModel(Drr, Dtheta, self.num_frame, self.gpu_id)
        'embeded info along time'
        # self.fc1 = nn.Linear(64 * self.num_frame, self.num_frame)
        # self.fc2 = nn.Linear(128*self.num_frame, self.num_frame)
        self.fcn1 = nn.Conv2d(self.num_frame, 25, kernel_size=1, stride=1, padding=0, groups=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fcn2 = nn.Conv2d(25, 10, kernel_size=1, stride=1, padding=0, groups=1, bias=False, dilation=1)
        self.bn3 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  # 64 x 10 x 3 x 3

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64*10*3*3, self.num_frame)
        # self.fc = nn.Linear(128 * 10 * 3 * 3, self.num_frame)
        # self.drop = nn.Dropout(0.2)
        self.sig = nn.Sigmoid()

    # def __init_weight(self):
    #     for m in self.modules():
    #         # if isinstance(m, nn.Conv1d):
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             # m.weight.data.normal_(0, math.sqrt(2. / n))
    #             # torch.nn.init.xavier_normal_(m.weight, gain=1)
    #         elif isinstance(m, nn.Linear):
    #             torch.nn.init.xavier_uniform_(m.weight, gain=1)
    #             # m.weight.data.fill_(0.01)
    #             # m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)


    def forward(self, x):
        imageFeature = self.modifiedResnet(x)
        Dictionary = creatRealDictionary(self.num_frame, self.Drr, self.Dtheta, self.gpu_id)
        # convx = self.Conv2d(imageFeature)
        # convx = self.bn1(convx)
        # convx = self.relu(convx)

        convx = imageFeature
        x2 = self.layer2(convx)
        x2 = self.bn_l2(x2)
        x2 = self.relu(x2)

        x3 = self.layer3(x2)
        x3 = self.bn_l3(x3)
        x3 = self.relu(x3)
        # feature = x3

        x4 = self.layer4(x3)
        x4 = self.bn_l4(x4)
        feature = self.relu(x4)

        return feature, Dictionary,imageFeature
        # imageFeature = self.modifiedResnet(x)  # T X 512 X 7 X 7
        # # print('freeze layers:', imageFeature[0,0])
        #
        # convx = self.Conv2d(imageFeature)  # reduce feature size
        # convx = self.bn1(convx)
        # convx = self.relu(convx)
        #
        # # print('conv2d weigth', self.Conv2d.weight[0,0])
        # # x1 = convx.unsqueeze(0)   # 1 x T x 512 x 7 x 7
        # x1 = convx.unsqueeze(0)
        # x1_re = torch.flip(x1, [0, 1])
        #
        # temporalFeature, _ = self.convLSTM(x1)  # convLSTM: 1 x T x 64 x 7 x 7
        # x = temporalFeature[0].squeeze(0)
        # reverseTemp, _ = self.convLSTM(x1_re)
        #
        # x_re = torch.flip(reverseTemp[0], [0, 1]).squeeze(0)
        # Bifeature = torch.cat((x, x_re), 1)
        #
        # return temporalFeature, Bifeature

    # def forward(self, x):
    #     PRINT_GRADS = True
    #     # feature, _ = self.featureExtractor(x)
    #     #
    #     # x = feature[0]
    #
    #     feature = self.featureExtractor(x)
    #     x = feature.view(feature.shape[0], -1).unsqueeze(0)  # N x T x (wxh)
    #
    #     # reconstFeature = self.DYAN(x)
    #     # _, Dictionary = self.DYAN.l1(x)
    #     Dictionary = creatRealDictionary(self.num_frame, self.Drr,  self.Dtheta, self.gpu_id)
    #     # Dictionary.register_hook(
    #     #     lambda grad: print("Dictionary.grad = {}".format(grad)) if PRINT_GRADS else False
    #     # )
    #
    #     # return Dictionary, feature[0]
    #     return Dictionary, feature

    def forward2(self, feature, alpha):

        # feature, _ = self.featureExtractor(x)
        # x = feature[0].squeeze(0).permute(1, 0, 2, 3)
        x = feature.permute(1, 0, 2, 3)
        x = self.fcn1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fcn2(x)
        x = self.bn3(x)
        x = self.relu(x)
        # x = self.avg_pool(x)
        x = x.view(1, -1)
        x = self.fc(x)
        out = self.sig(alpha*x)
        return out


if __name__ == "__main__":

    gpu_id = 2
    alpha = 4 # step size for sigmoid
    N = 4 * 40
    P, Pall = gridRing(N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()

    net = keyframeProposalNet(numFrame=40, Drr=Drr, Dtheta=Dtheta, gpu_id=gpu_id)
    net.cuda(gpu_id)

    X = torch.randn(1, 40, 3, 224, 224).cuda(gpu_id)
    imF = []
    fcF = []
    temF =[]
    convF = []
    Y =[]
    for i in range(0, X.shape[0]):
        x = X[i]
        Dictionary, feature,_ = net.forward(x)
        out = net.forward2(feature, alpha)
        print('check')

        # y, imf, temf, fcf, convf = net(x, alpha)
        # imF.append(imf.mean().detach().cpu().data.item())
        # temF.append(temf.mean().detach().cpu().data.item())
        # fcF.append(fcf.mean().detach().cpu().data.item())
        # convF.append(convf.mean().detach().cpu().data.item())
        # Y.append(y.squeeze(0).detach().cpu().numpy())
        #
    print('done')

    # total_params = sum(p.numel() for p in net.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(
    #     p.numel() for p in net.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
