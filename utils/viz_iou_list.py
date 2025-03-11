# developed by jxLiang

import time
from matplotlib import pyplot as plt
import numpy as np

iou_list = np.fromfile('iou_list/moving_iou_list.npy',dtype=np.float64)
print("moving:",iou_list)
plt.plot(iou_list, color='black')
plt.xlabel('frame')
plt.ylabel('iou')
moving_file_name = "./iou_list/"+f'moving_{iou_list[-1]*100:.4f}.png'
plt.savefig(moving_file_name)
print(moving_file_name)
# plt.show()

# iou_list = np.fromfile('iou_list/movable_iou_list.npy',dtype=np.float64)
# print("movable",iou_list)
# plt.plot(iou_list, color='black')
# plt.xlabel('frame')
# plt.ylabel('iou')
# plt.savefig("./iou_list/"+f'movable_{iou_list[-1]*100:.4f}.png')
# # plt.show()