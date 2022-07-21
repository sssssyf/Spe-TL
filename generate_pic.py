"""
@author: sssssyf
@blog: https://github.com/sssssyf
@email: sincere_sunyf@163.com
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0

def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):

        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([255,182,193]) / 255.
        if item == 2:
            y[index] = np.array([60,179,113]) / 255.
        if item == 3:
            y[index] = np.array([255,165,0]) / 255.
        if item == 4:
            y[index] = np.array([65,105,225]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 6:
            y[index] = np.array([148,0,211]) / 255.
        if item == 7:
            y[index] = np.array([139,69,19]) / 255.
        if item == 8:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 9:
            y[index] = np.array([0,255,255])/255.
        if item == 10:
            y[index] = np.array([128, 128, 0])/255.
        if item == 11:
            y[index] = np.array([255,255,0])/255.
        if item == 12:
            y[index] = np.array([121,255,49])/255.
        if item == 13:
            y[index] = np.array([255,49,183])/255.
        if item == 14:
            y[index] = np.array([112, 192, 188])/255.
        if item == 15:
            y[index] = np.array([183,121,121])/255.
        if item == 16:
            y[index] = np.array([13,0,100])/255.

    return y



def generate_png(gt_hsi,pred_test,flag,h,w,num):


    gt = gt_hsi.flatten()


    for i in range(len(pred_test)):
        pred_test[i] = pred_test[i] + 1



    y_list = list_to_colormap(pred_test)
    y_gt = list_to_colormap(gt)


    y_re = np.reshape(y_list, (h, w, 3))
    gt_re = np.reshape(y_gt, (h, w, 3))

    day = datetime.datetime.now()
    day_str = day.strftime('%m_%d_%H_%M')

    path = './maps/'
    classification_map(y_re, gt_re, 600,
                       path + '_' + 'Time_'+str(day_str)+'_'+str(flag)+'_'+str(num)+'num.eps')
    classification_map(gt_re, gt_re, 600,
                       path + 'Time_gt'+str(day_str)+'_'+str(flag)+'.eps')
    print('------Get classification maps successful-------')

