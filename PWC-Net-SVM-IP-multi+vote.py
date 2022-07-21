#!/usr/bin/env python
"""
@author: sssssyf
@blog: https://github.com/sssssyf
@email: sincere_sunyf@163.com
"""
import getopt
import math
import numpy
import sys
import torch
import scipy.io as sio
from tqdm import tqdm
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
import time
import datetime
from generate_pic import generate_png



try:
    from .correlation import correlation # the custom cost volume layer
except:
    sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python
# end

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'default' # 'default', or 'chairs-things'
arguments_strOne = './images/one.png'
arguments_strTwo = './images/two.png'
arguments_strOut = './out.flo'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
    if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use
    if strOption == '--one' and strArgument != '': arguments_strOne = strArgument # path to the first frame
    if strOption == '--two' and strArgument != '': arguments_strTwo = strArgument # path to the second frame
    if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

backwarp_tenGrid = {}
backwarp_tenPartial = {}

def get_accuracy(y_true, y_pred):
    num_perclass = np.zeros((y_true.max() + 1))
    num = np.zeros((y_true.max() + 1))
    for i in range(len(y_true)):
        num_perclass[y_true[i]] += 1
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            num[y_pred[i]] += 1
    for i in range(len(num)):
        num[i] = num[i] / num_perclass[i]
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    ac = np.zeros((y_true.max() + 1 + 2))
    ac[:y_true.max() + 1] = num
    ac[-1] = acc
    ac[-2] = kappa
    return ac  # acc,num.mean(),kappa

def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
    # end

    if str(tenFlow.shape) not in backwarp_tenPartial:
        backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones([ tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3] ])
    # end

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
    tenInput = torch.cat([ tenInput, backwarp_tenPartial[str(tenFlow.shape)] ], 1)

    tenOutput = torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)

    tenMask = tenOutput[:, -1:, :, :]; tenMask[tenMask > 0.999] = 1.0; tenMask[tenMask < 1.0] = 0.0

    return tenOutput[:, :-1, :, :] * tenMask
# end

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Extractor(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
            # end

            def forward(self, tenInput):
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)

                return [ tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix ]
            # end
        # end

        class Decoder(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
                intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

                if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6: self.fltBackwarp = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
                )
            # end

            def forward(self, tenOne, tenTwo, objPrevious):
                tenFlow = None
                tenFeat = None

                if objPrevious is None:
                    tenFlow = None
                    tenFeat = None

                    tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=tenTwo), negative_slope=0.1, inplace=False)

                    tenFeat = torch.cat([ tenVolume ], 1)

                elif objPrevious is not None:
                    tenFlow = self.netUpflow(objPrevious['tenFlow'])
                    tenFeat = self.netUpfeat(objPrevious['tenFeat'])

                    tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=backwarp(tenInput=tenTwo, tenFlow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)

                    tenFeat = torch.cat([ tenVolume, tenOne, tenFlow, tenFeat ], 1)

                # end

                tenFeat = torch.cat([ self.netOne(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netTwo(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netThr(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netFou(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netFiv(tenFeat), tenFeat ], 1)

                tenFlow = self.netSix(tenFeat)

                return {
                    'tenFlow': tenFlow,
                    'tenFeat': tenFeat
                }
            # end
        # end

        class Refiner(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
                )
            # end

            def forward(self, tenInput):
                return self.netMain(tenInput)
            # end
        # end

        self.netExtractor = Extractor()

        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)

        self.netRefiner = Refiner()

        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-pwc/network-' + arguments_strModel + '.pytorch', file_name='pwc-' + arguments_strModel).items() })


    # end

    def forward(self, tenOne, tenTwo):
        tenOne = self.netExtractor(tenOne)
        tenTwo = self.netExtractor(tenTwo)

        objEstimate = self.netSix(tenOne[-1], tenTwo[-1], None)
        objEstimate = self.netFiv(tenOne[-2], tenTwo[-2], objEstimate)
        objEstimate = self.netFou(tenOne[-3], tenTwo[-3], objEstimate)
        objEstimate = self.netThr(tenOne[-4], tenTwo[-4], objEstimate)
        objEstimate = self.netTwo(tenOne[-5], tenTwo[-5], objEstimate)

        return (objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])) * 20.0
    # end
# end

netNetwork = None

##########################################################

def estimate(tenOne, tenTwo):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().cuda().eval()
    # end

    assert(tenOne.shape[1] == tenTwo.shape[1])
    assert(tenOne.shape[2] == tenTwo.shape[2])

    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    #assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    #assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

    tenPreprocessedOne = torch.nn.functional.interpolate(input=tenPreprocessedOne, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenPreprocessedTwo, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    output=netNetwork(tenPreprocessedOne, tenPreprocessedTwo)
    #output=output[:,:,:int(output.shape[2]/2),:int(output.shape[3]/2)]

    tenFlow = torch.nn.functional.interpolate(input=output, size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    #tenFlow=tenFlow[:,:,:int(intHeight/2),:int(intWidth/2)]

    #tenFlow = torch.nn.functional.interpolate(input=tenFlow,size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    return tenFlow[0, :, :, :].cpu()
# end

##########################################################

if __name__ == '__main__':

    experiment_num=10
    dataname = 'IP'
    model_name = 'PWCnet_svm-multi+vote'
    #num_range=[10,20,50,80,100,150,200]
    num_range = [10]


    path='./HSI_data/'
    img=sio.loadmat(path+'Indian_pines_corrected.mat')
    img=img['indian_pines_corrected']

    gt=sio.loadmat(path+'Indian_pines_gt.mat')
    gt=gt['indian_pines_gt']


    spec=img.copy()
    spec=spec/spec.max()
    m,n,b=img.shape

    distance_list=[0,1,2,3,4,5,6,7,8,9,10]


    for j in range(len(distance_list)):
        distance=distance_list[j]
        for i in tqdm(range(distance+3, b)):
            x1 = img[:, :, i - distance-3:i-distance]
            x2 = img[:, :, i-2:i+1]

            tenOne = torch.FloatTensor(numpy.ascontiguousarray(x1.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
            tenTwo = torch.FloatTensor(numpy.ascontiguousarray(x2.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

            f = estimate(tenOne, tenTwo)
            if i == distance+3:
                feature = f
            else:
                feature = np.concatenate((feature, f),0)

        if distance ==0:
            feature_multi = feature
        else:
            feature_multi = np.concatenate((feature_multi, feature), 0)

    feature_multi=feature_multi.transpose(1,2,0)

    v_min = feature_multi.min()
    v_max = feature_multi.max()
    feature_multi = (feature_multi - v_min) / (v_max - v_min)



    feature_multi = np.concatenate((feature_multi, spec), 2)

    import h5py
    f = h5py.File('OpticalFlow_Global_'+str(dataname)+'_multi.h5', 'w')
    f['data'] = feature_multi
    f.close()


    [m, n, b] = img.shape
    label_num = gt.max()
    data = []
    label = []
    data_global = []

    gt_index = []
    for i in tqdm(range(m)):
        for j in range(n):
            if gt[i, j] == 0:
                continue
            else:
                temp_data = feature_multi[i, j, :]
                temp_label = np.zeros((1, label_num), dtype=np.int8)
                temp_label[0, gt[i, j] - 1] = 1
                data.append(temp_data)
                label.append(temp_label)
                gt_index.append((i) * n + j)
    #            print (i,j)

    for i in tqdm(range(m)):
        for j in range(n):
            temp_data = feature_multi[i, j, :]
            data_global.append(temp_data)

    print('end')
    data = np.array(data)
    data = np.squeeze(data)

    data_global = np.array(data_global)
    data_global = np.squeeze(data_global)

    label = np.array(label)
    label = np.squeeze(label)
    label = label.argmax(1)
    data = np.float32(data)
    data_global = np.float32(data_global)

    bands_list = []
    for dis_i in range(len(distance_list)):
        bands_list.append((b - distance_list[dis_i] - 3)*2)

    Experiment_result = np.zeros([label_num + 5, experiment_num + 2])
    for i_num in range(num_range.__len__()):
        num = num_range[i_num]

        for iter_num in range(experiment_num):

            np.random.seed(iter_num + 123456)
            indices = np.arange(data.shape[0])
            shuffled_indices = np.random.permutation(indices)

            preds = []
            preds_global=[]
            sum=0
            for vote_i in range(len(bands_list)):
                # data = feas[:,100*i:100*(i+1)]



                data_current = data[:, sum:sum + bands_list[vote_i]]
                data_global_current=data_global[:, sum:sum + bands_list[vote_i]]

                sum = sum + bands_list[vote_i]

                data_current = np.concatenate((data_current, data[:,-b:]), 1)
                data_global_current= np.concatenate((data_global_current, data_global[:,-b:]), 1)

                images = data_current[shuffled_indices]
                labels = label[shuffled_indices]
                y = labels  #
                n_classes = int(y.max() + 1)
                i_labeled = []
                for c in range(n_classes):
                    if dataname=='IP':
                        if num == 10:
                            i = indices[y == c][:num]
                        if num == 20:

                            if c + 1 == 7:
                                i = indices[y == c][:10]  # 50
                            elif c + 1 == 9:
                                i = indices[y == c][:10]  # 50
                            else:
                                i = indices[y == c][:num]  # 50

                        if num == 50:
                            if c + 1 == 1:
                                i = indices[y == c][:26]  # 50
                            elif c + 1 == 7:
                                i = indices[y == c][:16]  # 50
                            elif c + 1 == 9:
                                i = indices[y == c][:11]  # 50
                            else:
                                i = indices[y == c][:num]  # 50

                        if num == 80:
                            if c + 1 == 1:
                                i = indices[y == c][:26]  # 50
                            elif c + 1 == 7:
                                i = indices[y == c][:16]  # 50
                            elif c + 1 == 9:
                                i = indices[y == c][:11]  # 50
                            elif c + 1 == 16:
                                i = indices[y == c][:60]  # 50
                            else:
                                i = indices[y == c][:num]  # 50
                        if num == 100:
                            if c + 1 == 1:
                                i = indices[y == c][:33]  # 50
                            elif c + 1 == 7:
                                i = indices[y == c][:20]  # 50
                            elif c + 1 == 9:
                                i = indices[y == c][:14]  # 50
                            elif c + 1 == 16:
                                i = indices[y == c][:75]  # 50
                            else:
                                i = indices[y == c][:num]  # 50
                        if num == 150:
                            if c + 1 == 1:
                                i = indices[y == c][:36]  # 50
                            elif c + 1 == 7:
                                i = indices[y == c][:22]  # 50
                            elif c + 1 == 9:
                                i = indices[y == c][:16]  # 50
                            elif c + 1 == 16:
                                i = indices[y == c][:80]  # 50
                            else:
                                i = indices[y == c][:num]  # 50
                        if num == 200:
                            if c + 1 == 1:
                                i = indices[y == c][:39]  # 50
                            elif c + 1 == 7:
                                i = indices[y == c][:24]  # 50
                            elif c + 1 == 9:
                                i = indices[y == c][:18]  # 50
                            elif c + 1 == 16:
                                i = indices[y == c][:85]  # 50
                            else:
                                i = indices[y == c][:num]  # 50
                    else :
                         i = indices[y == c][:num]
                    i_labeled += list(i)

                l_images = images[i_labeled]
                l_labels = y[i_labeled]


                svc = SVC(kernel='rbf', class_weight='balanced', )
                c_range = np.logspace(-2, 7, 10, base=2)
                gamma_range = np.logspace(-2, 7, 10, base=2)
                # 网格搜索交叉验证的参数范围，cv=3,3折交叉，n_jobs=-1，多核计算
                param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
                grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)

                # 训练模型
                train_time1 = time.time()
                clf = grid.fit(l_images, l_labels)
                train_time2 = time.time()


                # 计算测试集精度
                score = grid.score(data_current, label)
                print('精度为%s' % score)

                tes_time1 = time.time()
                pred = clf.predict(data_current)
                tes_time2 = time.time()

                pred = pred.reshape(1, -1)
                preds.append(pred)

                pred_global = clf.predict(data_global_current)


                pred_global = pred_global.reshape(1, -1)
                preds_global.append(pred_global)




            preds = np.concatenate(preds, 0)
            preds_global = np.concatenate(preds_global, 0)

            import scipy

            output = scipy.stats.mode(preds, axis=0).mode[0]
            output_global = scipy.stats.mode(preds_global, axis=0).mode[0]

            predictions = (output == label)
            score = np.sum(predictions == True) / len(label)

            print("final vote results:"+str(score))

            generate_png(gt, output_global, dataname, m, n, num)

            ac = get_accuracy(output, label)

            Experiment_result[0, iter_num] = ac[-1] * 100  # OA
            Experiment_result[1, iter_num] = np.mean(ac[:-2]) * 100  # AA
            Experiment_result[2, iter_num] = ac[-2] * 100  # Kappa
            Experiment_result[3, iter_num] = train_time2 - train_time1
            Experiment_result[4, iter_num] = tes_time2 - tes_time1
            Experiment_result[5:, iter_num] = ac[:-2] * 100

            print('########### Experiment {}，Model assessment Finished！ ###########'.format(iter_num))

            ########## mean value & standard deviation #############

        Experiment_result[:, -2] = np.mean(Experiment_result[:, 0:-2], axis=1)  # 计算均值
        Experiment_result[:, -1] = np.std(Experiment_result[:, 0:-2], axis=1)  # 计算平均差

        print('OA_std={}'.format(Experiment_result[0, -1]))
        print('AA_std={}'.format(Experiment_result[1, -1]))
        print('Kappa_std={}'.format(Experiment_result[2, -1]))
        print('time training cost_std{:.4f} secs'.format(Experiment_result[3, -1]))
        print('time testing cost_std{:.4f} secs'.format(Experiment_result[4, -1]))
        for i in range(Experiment_result.shape[0]):
            if i > 4:
                print('Class_{}: accuracy_std {:.4f}.'.format(i - 4, Experiment_result[i, -1]))  # 均差

        day = datetime.datetime.now()
        day_str = day.strftime('%m_%d_%H_%M')

        f = open('./record/' + str(day_str) + '_' + dataname + '_' + model_name + '_' + str(num) + 'num.txt', 'w')
        for i in range(Experiment_result.shape[0]):
            f.write(str(i + 1) + ':' + str(round(Experiment_result[i, -2],2)) + '+/-' + str(round(Experiment_result[i, -1],2)) + '\n')
        for i in range(Experiment_result.shape[1] - 2):
            f.write('Experiment_num' + str(i) + '_OA:' + str(Experiment_result[0, i]) + '\n')
        f.close()


