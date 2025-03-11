import numpy as np 
import random
import os
import torch

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)

def per_class_acc(hist):
    tp = np.diag(hist)
    fp = hist.sum(1) - tp
    fn = hist.sum(0) - tp
    total_tp = tp.sum()
    total = tp.sum() + fp.sum() + 1e-15
    return total_tp / total

def fast_hist_crop(output, target, label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(label) + 1)
    hist = hist[label, :]
    hist = hist[:, label]
    return hist

def per_class_iu(hist):
    '''
        hist :(预测)   0                  1
         （真实）0   静态背景标签         错误标签（将静止物体预测为运动物体）
                1   错误标签            目标情况  
    '''
    tp = np.diag(hist)   # 正确的标签   1 -> 1
    fp = hist.sum(1) - tp # 错误的标签  0 -> 1
    fn = hist.sum(0) - tp # 1 -> 0
    # return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    # print('hist',hist)
    # print('tp',tp)
    # print('fp',fp)
    # print('fn',fn)
    # print('hist shape',hist.shape)
    # print('iu shape',tp.shape)
    return tp / (tp + fp + fn)


def analysis_of_movable(movable_hist,movable_hist_moving):
# movable_hist :     0      1
#               0    a      b
#               1    c      d

# movable_hist_moving :      0      1
#                       0    e      f
#                       1    g      h
    
# gt: moving/movable    0         1
#                  00   I         J
#                  01   K         L
#                  11   M         N
# I+K=e  J+L=f  M=g N=h       K+M=c  L+N=d  I=a  J=b    
    letter_N =movable_hist_moving[1,1]  # h
    letter_L = movable_hist[1,1] - letter_N #  d - h
    sum_of_JLNM = (movable_hist_moving.sum(0) - np.diag(movable_hist_moving)).sum()+letter_N  # f + g + h
    result = {
        "movable_and_moving":letter_N / sum_of_JLNM *100,
        "movable_and_static":letter_L/sum_of_JLNM*100 , 
        "movable_as_moving_error": (sum_of_JLNM - letter_L - letter_N)/sum_of_JLNM * 100
    }
    return result

def set_seed(seed=999):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # If we need to reproduce the results, increase the training speed
    #    set benchmark = False
    # If we don’t need to reproduce the results, improve the network performance as much as possible
    #    set benchmark = True
    return seed