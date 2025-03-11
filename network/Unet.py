# -*- coding: utf-8 -*-
# Developed by Liang jx

from dropblock import DropBlock2D
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.basic_blocks import double_conv_circular, inconv, outconv, down, up,single_conv,  single_conv_circular, down2
from network.BaseBlocks import add_down
from einops import rearrange
from network.mambavss import Block_VSS, C2f_VSS, VSSBlock, hswish
from network.mambavss2 import Block_mamba

__all__ = ('UNet_5', # 淘汰
           'UNet_6', # 淘汰
           'UNet_7', # 淘汰
           'UNet_8', # 新的模型结构
           'UNet_PR', # 基于原版MotionBEV模型，加入pixelshuffle，加入残差连接，加入RV残差图
           'UNet_8_pr', # 新的模型结构，加入了新的注意力机制模块到上采样过程中，加入RV残差图
           'UNet_8_pr_up', # 新的模型结构，加入RV残差图,修改up
           "UNet_8_pr2", # 新的模型结构，加入RV残差图,同时将点云从RV转换为BEV进行补充
           "UNet_8_pr_VSS4",
           'UNet_9_pr',# 新的模型结构，加入RV残差图，r2p加入vss
)



class UNet_5(nn.Module): # 全新的up
    ''' 全新的up 53.591ms
        '''
    def __init__(self, moving_n_class, movable_n_class,n_height, residual, dilation, group_conv, input_batch_norm, dropout, circular_padding,
                 *args,**kwargs):
        super(UNet_5, self).__init__()
        
        print('unet-5-全新的up')
        self.inc = inconv(n_height, 32, dilation, input_batch_norm, circular_padding)
        self.down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.down4 = down(256, 256, dilation, group_conv, circular_padding)

        self.res_inc = inconv(residual, 16, dilation, input_batch_norm, circular_padding)
        self.res_down1 = down(16, 32, dilation, group_conv, circular_padding)
        self.res_down2 = down(32, 64, dilation, group_conv, circular_padding)
        self.res_down3 = down(64, 128, dilation, group_conv, circular_padding)
        self.res_down4 = down(512, 512, dilation, group_conv, circular_padding)
        '''
        self.CGM0 = CAG(channel_a=32, channel_m=16, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM1 = CAG(channel_a=64, channel_m=32, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM2 = CAG(channel_a=128, channel_m=64, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM3 = CAG(channel_a=256, channel_m=128, circular_padding=circular_padding, group_conv=group_conv)
        #self.CGM4 = CAG(channel_a=512, channel_m=512, circular_padding=circular_padding, group_conv=group_conv)
        '''
        self.CGM0 = CAG(channel_a=32, channel_m=16, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM1 = CAG(channel_a=64, channel_m=32*2, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM2 = CAG(channel_a=128, channel_m=64*2+32, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM3 = CAG(channel_a=256, channel_m=128*2+64+32, circular_padding=circular_padding, group_conv=group_conv)
        self.add_down00 = add_down(16,32,2)
        self.add_down01 = add_down(16,32,4)
        self.add_down02 = add_down(16,32,8)
        self.add_down11 = add_down(32,64,2)
        self.add_down12 = add_down(32,64,4)
        self.add_down22 = add_down(64,128,2)


        # self.up3 = up(512, 128, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.up3 = newup(256,128)
        # self.up2 = up(256, 64, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.up2 = newup(128,64)
        # self.up1 = up(128, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.up1 = newup(64,32)
        # self.up0 = up(64, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.up0 = newup(32,32)

        # self.movable_up3 = up(512, 128, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.movable_up3 = newup(256,128)
        # self.movable_up2 = up(256, 64, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.movable_up2 = newup(128,64)
        # self.movable_up1 = up(128, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.movable_up1 = newup(64,32)
        # self.movable_up0 = up(64, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.movable_up0 = newup(32,32)

        self.moving_dropout = nn.Dropout(p=dropout)
        self.moving_outc = outconv(32, moving_n_class)

        self.movable_dropout = nn.Dropout(p=dropout)
        self.movable_outc = outconv(32,movable_n_class)# 2*n_height(32)

    def forward(self, x, res):
        c0_motion = self.res_inc(res)
        c1_motion = self.res_down1(c0_motion)
        c2_motion = self.res_down2(c1_motion)
        c3_motion = self.res_down3(c2_motion)
        #c4_motion = self.res_down4(c3_motion)

        c0_appearance = self.inc(x)  # ([4, 32, 480, 360])
        c0_appearance_fused, _ = self.CGM0(c0_appearance, c0_motion) # ([4, 32, 480, 360])

        c1_appearance = self.down1(c0_appearance_fused)  # ([4, 64, 240, 180])
        # c1_appearance, _ = self.CGM1(c1_appearance, c1_motion)
        c1_motion_add= torch.cat([c1_motion,self.add_down00(c0_motion)],dim=1)
        c1_appearance_fused,_ = self.CGM1(c1_appearance,c1_motion_add) # ([4, 64, 240, 180])

        c2_appearance = self.down2(c1_appearance_fused) # ([4, 128, 120, 90])
        # c2_appearance, _ = self.CGM2(c2_appearance, c2_motion)
        c2_motion_add= torch.cat([c2_motion,self.add_down11(c1_motion),self.add_down01(c0_motion)],dim=1)
        c2_appearance_fused,_ = self.CGM2(c2_appearance,c2_motion_add) # ([4, 128, 120, 90])

        c3_appearance = self.down3(c2_appearance_fused)  # ([4, 256, 60, 45])
        # c3_appearance, _ = self.CGM3(c3_appearance, c3_motion)
        c3_motion_add= torch.cat([c3_motion,self.add_down22(c2_motion),self.add_down12(c1_motion),self.add_down02(c0_motion)],dim=1)
        c3_appearance_fused,_ = self.CGM3(c3_appearance,c3_motion_add) # ([4, 256, 60, 45])

        c4_appearance = self.down4(c3_appearance_fused)  # ([4, 256, 30, 22])
        #c4_appearance, g4_motion = self.CGM4(c4_appearance, c4_motion)

        output = self.up3(c4_appearance, c3_appearance_fused)
        # torch.Size([4, 128, 60, 45])    ([4, 256, 30, 22])   ([4, 256, 60, 45])
        output = self.up2(output, c2_appearance_fused)
        # torch.Size([4, 64, 120, 90])    ([4, 128, 60, 45])   ([4, 128, 120, 90])
        output = self.up1(output, c1_appearance_fused)
        # torch.Size([4, 32, 240, 180])   ([4, 64, 120, 90])   ([4, 64, 240, 180])
        output = self.up0(output, c0_appearance_fused)
        # torch.Size([4, 32, 480, 360])   ([4, 32, 240, 180])   ([4, 32, 480, 360])

        # output3 torch.Size([4, 128, 60, 45])
        # output2 torch.Size([4, 64, 120, 90])
        # output1 torch.Size([4, 32, 240, 180])
        # output0 torch.Size([4, 32, 480, 360])

        movable_out = self.movable_up3(c4_appearance,c3_appearance)
        movable_out = self.movable_up2(movable_out,c2_appearance)
        movable_out = self.movable_up1(movable_out,c1_appearance)
        movable_out = self.movable_up0(movable_out,c0_appearance)

        moving_output = self.moving_outc(self.moving_dropout(output))
        #moving_output=F.softmax(moving_output, dim=1)

        movable_output = self.movable_outc(self.movable_dropout(movable_out))
        #movable_output=F.softmax(movable_output, dim=1)
        return moving_output,movable_output
    

class UNet_6(nn.Module): # 全新的down
    '''全新的down 67.268ms
    '''
    def __init__(self, moving_n_class, movable_n_class,n_height, residual, dilation, group_conv, input_batch_norm, dropout, circular_padding,
                 dropblock,PixelShuffle,using_UpBlock,drop_out,):
        super(UNet_6, self).__init__()
        print('全新的down')
        self.inc = inconv(n_height, 32, dilation, input_batch_norm, circular_padding)
        # self.down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.down1 = newdown(32,64,2)
        # self.down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.down2 = newdown(64,128,2)
        # self.down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.down3 = newdown(128,256,2)
        # self.down4 = down(256, 256, dilation, group_conv, circular_padding)
        self.down4 = newdown(256,256,2)

        self.res_inc = inconv(residual, 16, dilation, input_batch_norm, circular_padding)
        # self.res_down1 = down(16, 32, dilation, group_conv, circular_padding)
        self.res_down1 = newdown(16,32,2)
        # self.res_down2 = down(32, 64, dilation, group_conv, circular_padding)
        self.res_down2 = newdown(32,64,2)
        # self.res_down3 = down(64, 128, dilation, group_conv, circular_padding)
        self.res_down3 = newdown(64,128,2)
        # self.res_down4 = down(512, 512, dilation, group_conv, circular_padding)
        '''
        self.CGM0 = CAG(channel_a=32, channel_m=16, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM1 = CAG(channel_a=64, channel_m=32, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM2 = CAG(channel_a=128, channel_m=64, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM3 = CAG(channel_a=256, channel_m=128, circular_padding=circular_padding, group_conv=group_conv)
        #self.CGM4 = CAG(channel_a=512, channel_m=512, circular_padding=circular_padding, group_conv=group_conv)
        '''
        self.CGM0 = CAG(channel_a=32, channel_m=16, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM1 = CAG(channel_a=64, channel_m=32*2, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM2 = CAG(channel_a=128, channel_m=64*2+32, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM3 = CAG(channel_a=256, channel_m=128*2+64+32, circular_padding=circular_padding, group_conv=group_conv)
        self.add_down00 = add_down(16,32,2)
        self.add_down01 = add_down(16,32,4)
        self.add_down02 = add_down(16,32,8)
        self.add_down11 = add_down(32,64,2)
        self.add_down12 = add_down(32,64,4)
        self.add_down22 = add_down(64,128,2)


        self.up3 = up(512, 128, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.up2 = up(256, 64, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.up1 = up(128, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.up0 = up(64, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)

        self.movable_up3 = up(512, 128, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.movable_up2 = up(256, 64, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.movable_up1 = up(128, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.movable_up0 = up(64, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)

        self.moving_dropout = nn.Dropout(p=dropout)
        self.moving_outc = outconv(32, moving_n_class)

        self.movable_dropout = nn.Dropout(p=dropout)
        self.movable_outc = outconv(32,movable_n_class)# 2*n_height(32)

    def forward(self, x, res):
        c0_motion = self.res_inc(res)
        c1_motion = self.res_down1(c0_motion)
        c2_motion = self.res_down2(c1_motion)
        c3_motion = self.res_down3(c2_motion)
        #c4_motion = self.res_down4(c3_motion)

        c0_appearance = self.inc(x)  # ([4, 32, 480, 360])
        c0_appearance_fused, _ = self.CGM0(c0_appearance, c0_motion) # ([4, 32, 480, 360])

        c1_appearance = self.down1(c0_appearance_fused)  # ([4, 64, 240, 180])
        # c1_appearance, _ = self.CGM1(c1_appearance, c1_motion)
        c1_motion_add= torch.cat([c1_motion,self.add_down00(c0_motion)],dim=1)
        c1_appearance_fused,_ = self.CGM1(c1_appearance,c1_motion_add) # ([4, 64, 240, 180])

        c2_appearance = self.down2(c1_appearance_fused) # ([4, 128, 120, 90])
        # c2_appearance, _ = self.CGM2(c2_appearance, c2_motion)
        c2_motion_add= torch.cat([c2_motion,self.add_down11(c1_motion),self.add_down01(c0_motion)],dim=1)
        c2_appearance_fused,_ = self.CGM2(c2_appearance,c2_motion_add) # ([4, 128, 120, 90])

        c3_appearance = self.down3(c2_appearance_fused)  # ([4, 256, 60, 45])
        # c3_appearance, _ = self.CGM3(c3_appearance, c3_motion)
        c3_motion_add= torch.cat([c3_motion,self.add_down22(c2_motion),self.add_down12(c1_motion),self.add_down02(c0_motion)],dim=1)
        c3_appearance_fused,_ = self.CGM3(c3_appearance,c3_motion_add) # ([4, 256, 60, 45])

        c4_appearance = self.down4(c3_appearance_fused)  # ([4, 256, 30, 22])
        #c4_appearance, g4_motion = self.CGM4(c4_appearance, c4_motion)

        output = self.up3(c4_appearance, c3_appearance_fused)
        # torch.Size([4, 128, 60, 45])    ([4, 256, 30, 22])   ([4, 256, 60, 45])
        output = self.up2(output, c2_appearance_fused)
        # torch.Size([4, 64, 120, 90])    ([4, 128, 60, 45])   ([4, 128, 120, 90])
        output = self.up1(output, c1_appearance_fused)
        # torch.Size([4, 32, 240, 180])   ([4, 64, 120, 90])   ([4, 64, 240, 180])
        output = self.up0(output, c0_appearance_fused)
        # torch.Size([4, 32, 480, 360])   ([4, 32, 240, 180])   ([4, 32, 480, 360])

        # output3 torch.Size([4, 128, 60, 45])
        # output2 torch.Size([4, 64, 120, 90])
        # output1 torch.Size([4, 32, 240, 180])
        # output0 torch.Size([4, 32, 480, 360])

        movable_out = self.movable_up3(c4_appearance,c3_appearance)
        movable_out = self.movable_up2(movable_out,c2_appearance)
        movable_out = self.movable_up1(movable_out,c1_appearance)
        movable_out = self.movable_up0(movable_out,c0_appearance)

        moving_output = self.moving_outc(self.moving_dropout(output))
        #moving_output=F.softmax(moving_output, dim=1)

        movable_output = self.movable_outc(self.movable_dropout(movable_out))
        #movable_output=F.softmax(movable_output, dim=1)
        return moving_output,movable_output
    


class UNet_7(nn.Module): # 全新的down 和 up
    ''' 全新的down 和 up,75.477ms
    '''
    def __init__(self, moving_n_class, movable_n_class,n_height, residual, dilation, group_conv, input_batch_norm, dropout, circular_padding,
                 dropblock,PixelShuffle,using_UpBlock,drop_out,):
        super(UNet_7, self).__init__()
        print('全新的down 和 up')
        self.inc = inconv(n_height, 32, dilation, input_batch_norm, circular_padding)
        # self.down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.down1 = newdown(32,64,2)
        # self.down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.down2 = newdown(64,128,2)
        # self.down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.down3 = newdown(128,256,2)
        # self.down4 = down(256, 256, dilation, group_conv, circular_padding)
        self.down4 = newdown(256,256,2)

        self.res_inc = inconv(residual, 16, dilation, input_batch_norm, circular_padding)
        # self.res_down1 = down(16, 32, dilation, group_conv, circular_padding)
        self.res_down1 = newdown(16,32,2)
        # self.res_down2 = down(32, 64, dilation, group_conv, circular_padding)
        self.res_down2 = newdown(32,64,2)
        # self.res_down3 = down(64, 128, dilation, group_conv, circular_padding)
        self.res_down3 = newdown(64,128,2)
        # self.res_down4 = down(512, 512, dilation, group_conv, circular_padding)
        '''
        self.CGM0 = CAG(channel_a=32, channel_m=16, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM1 = CAG(channel_a=64, channel_m=32, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM2 = CAG(channel_a=128, channel_m=64, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM3 = CAG(channel_a=256, channel_m=128, circular_padding=circular_padding, group_conv=group_conv)
        #self.CGM4 = CAG(channel_a=512, channel_m=512, circular_padding=circular_padding, group_conv=group_conv)
        '''
        self.CGM0 = CAG(channel_a=32, channel_m=16, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM1 = CAG(channel_a=64, channel_m=32*2, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM2 = CAG(channel_a=128, channel_m=64*2+32, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM3 = CAG(channel_a=256, channel_m=128*2+64+32, circular_padding=circular_padding, group_conv=group_conv)
        self.add_down00 = add_down(16,32,2)
        self.add_down01 = add_down(16,32,4)
        self.add_down02 = add_down(16,32,8)
        self.add_down11 = add_down(32,64,2)
        self.add_down12 = add_down(32,64,4)
        self.add_down22 = add_down(64,128,2)


        # self.up3 = up(512, 128, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.up3 = newup(256,128)
        # self.up2 = up(256, 64, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.up2 = newup(128,64)
        # self.up1 = up(128, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.up1 = newup(64,32)
        # self.up0 = up(64, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.up0 = newup(32,32)

        # self.movable_up3 = up(512, 128, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.movable_up3 = newup(256,128)
        # self.movable_up2 = up(256, 64, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.movable_up2 = newup(128,64)
        # self.movable_up1 = up(128, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.movable_up1 = newup(64,32)
        # self.movable_up0 = up(64, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.movable_up0 = newup(32,32)

        self.moving_dropout = nn.Dropout(p=dropout)
        self.moving_outc = outconv(32, moving_n_class)

        self.movable_dropout = nn.Dropout(p=dropout)
        self.movable_outc = outconv(32,movable_n_class)# 2*n_height(32)

    def forward(self, x, res):
        c0_motion = self.res_inc(res)
        c1_motion = self.res_down1(c0_motion)
        c2_motion = self.res_down2(c1_motion)
        c3_motion = self.res_down3(c2_motion)
        #c4_motion = self.res_down4(c3_motion)

        c0_appearance = self.inc(x)  # ([4, 32, 480, 360])
        c0_appearance_fused, _ = self.CGM0(c0_appearance, c0_motion) # ([4, 32, 480, 360])

        c1_appearance = self.down1(c0_appearance_fused)  # ([4, 64, 240, 180])
        # c1_appearance, _ = self.CGM1(c1_appearance, c1_motion)
        c1_motion_add= torch.cat([c1_motion,self.add_down00(c0_motion)],dim=1)
        c1_appearance_fused,_ = self.CGM1(c1_appearance,c1_motion_add) # ([4, 64, 240, 180])

        c2_appearance = self.down2(c1_appearance_fused) # ([4, 128, 120, 90])
        # c2_appearance, _ = self.CGM2(c2_appearance, c2_motion)
        c2_motion_add= torch.cat([c2_motion,self.add_down11(c1_motion),self.add_down01(c0_motion)],dim=1)
        c2_appearance_fused,_ = self.CGM2(c2_appearance,c2_motion_add) # ([4, 128, 120, 90])

        c3_appearance = self.down3(c2_appearance_fused)  # ([4, 256, 60, 45])
        # c3_appearance, _ = self.CGM3(c3_appearance, c3_motion)
        c3_motion_add= torch.cat([c3_motion,self.add_down22(c2_motion),self.add_down12(c1_motion),self.add_down02(c0_motion)],dim=1)
        c3_appearance_fused,_ = self.CGM3(c3_appearance,c3_motion_add) # ([4, 256, 60, 45])

        c4_appearance = self.down4(c3_appearance_fused)  # ([4, 256, 30, 22])
        #c4_appearance, g4_motion = self.CGM4(c4_appearance, c4_motion)

        output = self.up3(c4_appearance, c3_appearance_fused)
        # torch.Size([4, 128, 60, 45])    ([4, 256, 30, 22])   ([4, 256, 60, 45])
        output = self.up2(output, c2_appearance_fused)
        # torch.Size([4, 64, 120, 90])    ([4, 128, 60, 45])   ([4, 128, 120, 90])
        output = self.up1(output, c1_appearance_fused)
        # torch.Size([4, 32, 240, 180])   ([4, 64, 120, 90])   ([4, 64, 240, 180])
        output = self.up0(output, c0_appearance_fused)
        # torch.Size([4, 32, 480, 360])   ([4, 32, 240, 180])   ([4, 32, 480, 360])

        # output3 torch.Size([4, 128, 60, 45])
        # output2 torch.Size([4, 64, 120, 90])
        # output1 torch.Size([4, 32, 240, 180])
        # output0 torch.Size([4, 32, 480, 360])

        movable_out = self.movable_up3(c4_appearance,c3_appearance)
        movable_out = self.movable_up2(movable_out,c2_appearance)
        movable_out = self.movable_up1(movable_out,c1_appearance)
        movable_out = self.movable_up0(movable_out,c0_appearance)

        moving_output = self.moving_outc(self.moving_dropout(output))
        #moving_output=F.softmax(moving_output, dim=1)

        movable_output = self.movable_outc(self.movable_dropout(movable_out))
        #movable_output=F.softmax(movable_output, dim=1)
        return moving_output,movable_output

class UNet_PR(nn.Module):
    '''基于原版MotionBEV模型，加入pixelshuffle，加入残差连接，加入RV残差图
    '''
    def __init__(self, moving_n_class, movable_n_class,n_height, residual, dilation, group_conv, input_batch_norm, dropout, circular_padding,
                 dropblock,PixelShuffle,using_UpBlock,drop_out,):
        super(UNet_PR, self).__init__()
        print('UNet')
        self.inc = inconv(n_height, 32, dilation, input_batch_norm, circular_padding)
        self.down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.down4 = down(256, 256, dilation, group_conv, circular_padding)

        self.res_inc = inconv(residual, 16, dilation, input_batch_norm, circular_padding)
        self.res_down1 = down(16, 32, dilation, group_conv, circular_padding)
        self.res_down2 = down(32, 64, dilation, group_conv, circular_padding)
        self.res_down3 = down(64, 128, dilation, group_conv, circular_padding)
        # self.res_down4 = down(512, 512, dilation, group_conv, circular_padding)

        self.res_inc_range = inconv(residual, 16, dilation, input_batch_norm, circular_padding)
        self.res_down1_range = down(16, 32, dilation, group_conv, circular_padding)
        self.res_down2_range = down(32, 64, dilation, group_conv, circular_padding)
        self.res_down3_range = down2(64, 128, dilation, group_conv, circular_padding)
        '''
        self.CGM0 = CAG(channel_a=32, channel_m=16, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM1 = CAG(channel_a=64, channel_m=32, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM2 = CAG(channel_a=128, channel_m=64, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM3 = CAG(channel_a=256, channel_m=128, circular_padding=circular_padding, group_conv=group_conv)
        #self.CGM4 = CAG(channel_a=512, channel_m=512, circular_padding=circular_padding, group_conv=group_conv)
        '''
        self.CGM0 = CAG(channel_a=32, channel_m=16, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM1 = CAG(channel_a=64, channel_m=32*2, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM2 = CAG(channel_a=128, channel_m=64*2+32, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM3 = CAG(channel_a=256, channel_m=128*2+64+32, circular_padding=circular_padding, group_conv=group_conv)
        self.add_down00 = add_down(16,32,2)
        self.add_down01 = add_down(16,32,4)
        self.add_down02 = add_down(16,32,8)
        self.add_down11 = add_down(32,64,2)
        self.add_down12 = add_down(32,64,4)
        self.add_down22 = add_down(64,128,2)

        self.flow_l2_r2p = R2B_flow_new3_nres(32)
        self.flow_l3_r2p = R2B_flow_new3_nres(64)
        self.flow_l4_r2p = R2B_flow_new3_nres(128)

        #dropblock = False
        if not using_UpBlock:
            self.up3 = up(512, 128, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
            self.up2 = up(256, 64, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
            self.up1 = up(128, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
            self.up0 = up(64, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)

            self.movable_up3 = up(512, 128, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
            self.movable_up2 = up(256, 64, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
            self.movable_up1 = up(128, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
            self.movable_up0 = up(64, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        else:
            print("network using UpBlock ")
            self.up3 = UpBlock(512, 128,drop_out = drop_out,drop_p=dropout)
            self.up2 = UpBlock(256, 64, drop_out = drop_out,drop_p=dropout)
            self.up1 = UpBlock(128, 32, drop_out = drop_out,drop_p=dropout)
            self.up0 = UpBlock(64, 32,drop_out = drop_out,drop_p=dropout)

        self.moving_dropout = nn.Dropout(p=dropout)
        self.moving_outc = outconv(32, moving_n_class)

        self.movable_dropout = nn.Dropout(p=dropout)
        self.movable_outc = outconv(32,movable_n_class)# 2*n_height(32)

    def forward(self, x, res, r2p_matrix, p2r_matrix, range_res):
        self.r2p_matrix = r2p_matrix
        # x : B 32 480 360
        # res : B 8 480 360
        c0_motion = self.res_inc(res) # B 8 480 360 -> B 16 480 360
        c1_motion = self.res_down1(c0_motion) # B 16 480 360 -> B 32 240 180
        c2_motion = self.res_down2(c1_motion) # B 32 240 180 -> B 64 120 90
        c3_motion = self.res_down3(c2_motion) # B 64 120 90 -> B 128 60 45
        #c4_motion = self.res_down4(c3_motion)
        c0_motion_range = self.res_inc_range(range_res)
        c1_motion_range = self.res_down1_range(c0_motion_range)
        c2_motion_range = self.res_down2_range(c1_motion_range)
        c3_motion_range = self.res_down3_range(c2_motion_range)

        c0_appearance = self.inc(x)  # B 32 480 360 -> B 32 480 360
        c0_appearance_fused, _ = self.CGM0(c0_appearance, c0_motion) # B 32 480 360 | B 16 480 360 -> B 32 480 360 | B 16 480 360

        c1_appearance = self.down1(c0_appearance_fused)  # B 32 480 360 -> B 64 240 180
        # c1_appearance, _ = self.CGM1(c1_appearance, c1_motion)
        c1_motion_f = self.flow_l2_r2p(self.r2p_matrix, c1_motion, c1_motion_range)
        c1_motion_add= torch.cat([c1_motion_f,self.add_down00(c0_motion)],dim=1) # -> B 64 240 180
        c1_appearance_fused,_ = self.CGM1(c1_appearance,c1_motion_add) # B 64 240 180 | B 64 240 180 -> B 64 240 180 | B 64 240 180

        c2_appearance = self.down2(c1_appearance_fused)  # B 64 240 180 -> B 128 120 90
        # c2_appearance, _ = self.CGM2(c2_appearance, c2_motion)
        c2_motion_f = self.flow_l3_r2p(self.r2p_matrix, c2_motion, c2_motion_range)
        c2_motion_add= torch.cat([c2_motion_f,self.add_down11(c1_motion),self.add_down01(c0_motion)],dim=1) # -> B 160 120 90
        c2_appearance_fused,_ = self.CGM2(c2_appearance,c2_motion_add) # B 128 120 90 | B 160 120 90 -> B 128 120 90 | B 160 120 90

        c3_appearance = self.down3(c2_appearance_fused)  # B 128 120 90 -> B 256 60 45
        # c3_appearance, _ = self.CGM3(c3_appearance, c3_motion)
        c3_motion_f = self.flow_l4_r2p(self.r2p_matrix, c3_motion, c3_motion_range)
        c3_motion_add= torch.cat([c3_motion_f,self.add_down22(c2_motion),self.add_down12(c1_motion),self.add_down02(c0_motion)],dim=1) # -> B 352 60 45
        c3_appearance_fused,_ = self.CGM3(c3_appearance,c3_motion_add) # B 256 60 45 | B 352 60 45 -> B 256 60 45 | B 352 60 45

        c4_appearance = self.down4(c3_appearance_fused)  # B 256 60 45 -> B 256 30 22
        #c4_appearance, g4_motion = self.CGM4(c4_appearance, c4_motion)

        output = self.up3(c4_appearance, c3_appearance_fused) # B 256 30 22 | B 256 60 45 -> B 64 120 90
        output = self.up2(output, c2_appearance_fused) # B 64 120 90 | B 128 120 90 -> B 32 240 180
        output = self.up1(output, c1_appearance_fused)
        output = self.up0(output, c0_appearance_fused)

        movable_out = self.movable_up3(c4_appearance,c3_appearance)
        movable_out = self.movable_up2(movable_out,c2_appearance)
        movable_out = self.movable_up1(movable_out,c1_appearance)
        movable_out = self.movable_up0(movable_out,c0_appearance)

        moving_output = self.moving_outc(self.moving_dropout(output))
        #moving_output=F.softmax(moving_output, dim=1)

        movable_output = self.movable_outc(self.movable_dropout(movable_out))
        #movable_output=F.softmax(movable_output, dim=1)
        return moving_output,movable_output

class UNet_8(nn.Module):
    '''模型结构逻辑换了，加入了新的注意力机制模块到上采样过程中
    '''
    def __init__(self, moving_n_class, movable_n_class,n_height, residual, dilation, group_conv, input_batch_norm, dropout, circular_padding,
                 dropblock,*args):
        super(UNet_8, self).__init__()
        self.cat = False
        self.cat_block = False
        self.mgmfa = True
        self.cca_out = False  # 对输出进行直接的特征提取
        self.newup2 = False
        print("是否是cat:",self.cat)
        print("是否是cca:",self.cat_block)
        print("是否是mgmfa:",self.mgmfa)
        print("是否是cca_out:",self.cca_out)        
        print("newup2",self.newup2 )
        self.inc = inconv(n_height, 32, dilation, input_batch_norm, circular_padding)
        self.down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.down4 = down(256, 256, dilation, group_conv, circular_padding)

        self.res_inc = inconv(residual, 32, dilation, input_batch_norm, circular_padding)
        self.res_down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.res_down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.res_down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.res_down4 = down(256, 256, dilation, group_conv, circular_padding)

        # self.res_inc = inconv(residual, 16, dilation, input_batch_norm, circular_padding)
        # self.res_down1 = down(16, 32, dilation, group_conv, circular_padding)
        # self.res_down2 = down(32, 64, dilation, group_conv, circular_padding)
        # self.res_down3 = down(64, 128, dilation, group_conv, circular_padding)
        # self.res_down4 = down(128, 128, dilation, group_conv, circular_padding)

        self.CGM0 = CAG(channel_a=32, channel_m=32, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM1 = CAG(channel_a=64, channel_m=64, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM2 = CAG(channel_a=128, channel_m=128, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM3 = CAG(channel_a=256, channel_m=256, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM4 = CAG(channel_a=256, channel_m=256, circular_padding=circular_padding, group_conv=group_conv) 

        # self.CGM0 = CAG(channel_a=16, channel_m=32, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM1 = CAG(channel_a=32, channel_m=64, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM2 = CAG(channel_a=64, channel_m=128, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM3 = CAG(channel_a=128, channel_m=256, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM4 = CAG(channel_a=128, channel_m=256, circular_padding=circular_padding, group_conv=group_conv)
        if self.cat:
            if self.newup2 :
                self.up3 = newup2(256,384,out_ch=192,use_dropblock=dropblock,drop_p=dropout)
                self.up2 = newup2(192,192,out_ch=96,use_dropblock=dropblock,drop_p=dropout)
                self.up1 = newup2(96,96,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
                self.up0 = newup2(64,64,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
            else:
                self.up3 = up2(256,384,out_ch=192,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up2 = up2(192,192,out_ch=96,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up1 = up2(96,96,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up0 = up2(64,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        elif self.cat_block:
            self.up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.CCA3 = CrossCoAttention(256,128)
            self.CCA2 = CrossCoAttention(128,64)
            self.CCA1 = CrossCoAttention(64,32)
            self.CCA0 = CrossCoAttention(32,32)
        elif self.mgmfa:
            self.up3 = up2(256,384,out_ch=192,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(192,192,out_ch=96,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(96,96,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(64,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.MGMFA3 = MGMFA(256,128)
            self.MGMFA2 = MGMFA(128,64)
            self.MGMFA1 = MGMFA(64,32)
            self.MGMFA0 = MGMFA(32,32)
        else : 
            self.up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)

        # self.up3 = up2(128,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up2 = up2(128,128,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up1 = up2(128,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up0 = up2(64,48,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        if self.newup2 and self.cat:
            self.movable_up3 = newup2(256,256,out_ch=128,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up2 = newup2(128,128,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up1 = newup2(64,64,out_ch=32,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up0 = newup2(32,32,out_ch=32,use_dropblock=dropblock,drop_p=dropout)
        else:
            self.movable_up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)

        self.moving_dropout = nn.Dropout(p=dropout)
        if self.cat or self.mgmfa:
            self.moving_outc = outconv(64, moving_n_class)
        else: self.moving_outc = outconv(32, moving_n_class)

        self.movable_dropout = nn.Dropout(p=dropout)
        self.movable_outc = outconv(32,movable_n_class)# 2*n_height(32)
        if self.cca_out:
            self.CCA_OUT = CrossCoAttention(64,64)
    def forward(self, x, res):
        # x : B 32 480 360
        # res : B 8 480 360
        c0_appearance = self.inc(x)                # B 32 480 360 -> B 32 480 360
        c1_appearance = self.down1(c0_appearance)  # B 32 480 360 -> B 64 240 180
        c2_appearance = self.down2(c1_appearance)  # B 64 240 180 -> B 128 120 90
        c3_appearance = self.down3(c2_appearance)  # B 128 120 90 -> B 256 60 45
        c4_appearance = self.down4(c3_appearance)  # B 256 60 45  -> B 256 30 22

        c0_motion = self.res_inc(res) # B 8 480 360 -> B 16 480 360
        c0_motion , _ =  self.CGM0(c0_motion,c0_appearance)# B 16 480 360 | B 32 480 360 -> B 16 480 360 | B 32 480 360  

        c1_motion = self.res_down1(c0_motion) # B 16 480 360 -> B 32 240 180
        c1_motion , _ =  self.CGM1(c1_motion,c1_appearance)# B 32 240 180 | B 64 240 180 -> B 32 240 180 | B 64 240 180 

        c2_motion = self.res_down2(c1_motion) # B 32 240 180 -> B 64 120 90
        c2_motion , _ =  self.CGM2(c2_motion,c2_appearance)# B 64 120 90 | B 128 120 90 -> B 64 120 90 | B 128 120 90 

        c3_motion = self.res_down3(c2_motion) # B 64 120 90 -> B 128 60 45
        c3_motion , _ =  self.CGM3(c3_motion,c3_appearance)# B 128 60 45 |  B 256 60 45 -> B 128 60 45 |  B 256 60 45 

        c4_motion = self.res_down4(c3_motion) # B 128 60 45 -> B 128 30 22 
        c4_motion , _ =  self.CGM4(c4_motion,c4_appearance)#  B 128 30 22 | B 256 30 22  ->  B 128 30 22  B 256 30 22 

        movable_out3 = self.movable_up3(c4_appearance,c3_appearance) # B 256 30 22 | B 256 60 45 -> B 128 60 45
        movable_out2 = self.movable_up2(movable_out3,c2_appearance) # B 128 60 45 | B 128 120 90 -> B 64 120 90
        movable_out1 = self.movable_up1(movable_out2,c1_appearance) # B 64 120 90 | B 64 240 180 -> B 32 240 180
        movable_out0 = self.movable_up0(movable_out1,c0_appearance) # B 32 240 180 | B 32 480 360 -> B 32 480 360
        movable_out = self.movable_outc(self.movable_dropout(movable_out0))

        if self.cat:
            c3_motion = torch.cat([c3_motion,movable_out3],dim=1)
        elif self.cat_block:
            c3_motion = self.CCA3(c3_motion,movable_out3)
        elif self.mgmfa:
            c3_motion = self.MGMFA3(c3_motion,movable_out3)
        else :
            c3_motion = c3_motion
        moving_out3 = self.up3(c4_motion, c3_motion)

        if self.cat:
            c2_motion = torch.cat([c2_motion,movable_out2],dim=1)
        elif self.cat_block:
            c2_motion = self.CCA2(c2_motion,movable_out2)
        elif self.mgmfa:
            c2_motion = self.MGMFA2(c2_motion,movable_out2)
        else:
            c2_motion = c2_motion
        moving_out2 = self.up2(moving_out3, c2_motion)
        
        if self.cat:
            c1_motion = torch.cat([c1_motion,movable_out1],dim=1)
        elif self.cat_block:
            c1_motion = self.CCA1(c1_motion,movable_out1)
        elif self.mgmfa:
            c1_motion = self.MGMFA1(c1_motion,movable_out1)
        else :
            c1_motion = c1_motion
        moving_out1 = self.up1(moving_out2, c1_motion)

        if self.cat:
            c0_motion = torch.cat([c0_motion,movable_out0],dim=1)
        elif self.cat_block:
            c0_motion = self.CCA0(c0_motion,movable_out0)  
        elif self.mgmfa:
            c0_motion = self.MGMFA0(c0_motion,movable_out0)
        else:
            c0_motion = c0_motion      
        moving_out0 = self.up0(moving_out1, c0_motion)


        moving_out = self.moving_outc(self.moving_dropout(moving_out0))
        
        if self.cca_out:
            moving_out = self.CCA_OUT(moving_out,moving_out)
        return moving_out,movable_out
    
class UNet_8_pr(nn.Module):
    '''模型结构逻辑换了，加入了新的注意力机制模块到上采样过程中，加入RV残差图
    '''
    def __init__(self, moving_n_class, movable_n_class,n_height, residual, dilation, group_conv, input_batch_norm, dropout, circular_padding,
                 dropblock,*args):
        super(UNet_8_pr, self).__init__()
        self.cat = True
        self.cat_block = False
        self.mgmfa = False
        self.cca_out = False  # 对输出进行直接的特征提取
        self.newup2 = False
        print("是否是cat:",self.cat)
        print("是否是cca:",self.cat_block)
        print("是否是mgmfa:",self.mgmfa)
        print("是否是cca_out:",self.cca_out)        
        print("newup2",self.newup2 )
        self.inc = inconv(n_height, 32, dilation, input_batch_norm, circular_padding)
        self.down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.down4 = down(256, 256, dilation, group_conv, circular_padding)

        self.res_inc = inconv(residual, 32, dilation, input_batch_norm, circular_padding)
        self.res_down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.res_down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.res_down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.res_down4 = down(256, 256, dilation, group_conv, circular_padding)

        self.res_inc_range = inconv(residual, 32, dilation, input_batch_norm, circular_padding)
        self.res_down1_range = down(32, 64, dilation, group_conv, circular_padding)
        self.res_down2_range = down(64, 128, dilation, group_conv, circular_padding)
        self.res_down3_range = down2(128, 256, dilation, group_conv, circular_padding)
        self.res_down4_range = down2(256, 256, dilation, group_conv, circular_padding)

        self.flow_l2_r2p = R2B_flow_new3(64)
        self.flow_l3_r2p = R2B_flow_new3(128)
        self.flow_l4_r2p = R2B_flow_new3(256)
        self.flow_l5_r2p = R2B_flow_new3(256)

        # self.vss2 = Block_mamba(64,8)
        # self.vss3 = Block_mamba(128,8)
        # self.vss4 = Block_mamba(256,4)
        # self.vss5 = Block_mamba(256,4)

        self.CGM0 = CAG(channel_a=32, channel_m=32, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM1 = CAG(channel_a=64, channel_m=64, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM2 = CAG(channel_a=128, channel_m=128, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM3 = CAG(channel_a=256, channel_m=256, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM4 = CAG(channel_a=256, channel_m=256, circular_padding=circular_padding, group_conv=group_conv) 

        # self.CGM0 = CAG(channel_a=16, channel_m=32, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM1 = CAG(channel_a=32, channel_m=64, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM2 = CAG(channel_a=64, channel_m=128, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM3 = CAG(channel_a=128, channel_m=256, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM4 = CAG(channel_a=128, channel_m=256, circular_padding=circular_padding, group_conv=group_conv)
        if self.cat:
            if self.newup2 :
                self.up3 = newup2(256,384,out_ch=192,use_dropblock=dropblock,drop_p=dropout)
                self.up2 = newup2(192,192,out_ch=96,use_dropblock=dropblock,drop_p=dropout)
                self.up1 = newup2(96,96,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
                self.up0 = newup2(64,64,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
            else:
                self.up3 = up2(256,384,out_ch=192,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up2 = up2(192,192,out_ch=96,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up1 = up2(96,96,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up0 = up2(64,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        elif self.cat_block:
            self.up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.CCA3 = CrossCoAttention(256,128)
            self.CCA2 = CrossCoAttention(128,64)
            self.CCA1 = CrossCoAttention(64,32)
            self.CCA0 = CrossCoAttention(32,32)
        elif self.mgmfa:
            self.up3 = up2(256,384,out_ch=192,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(192,192,out_ch=96,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(96,96,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(64,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.MGMFA3 = MGMFA(256,128)
            self.MGMFA2 = MGMFA(128,64)
            self.MGMFA1 = MGMFA(64,32)
            self.MGMFA0 = MGMFA(32,32)
        else : 
            self.up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)

        # self.up3 = up2(128,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up2 = up2(128,128,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up1 = up2(128,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up0 = up2(64,48,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        if self.newup2 and self.cat:
            self.movable_up3 = newup2(256,256,out_ch=128,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up2 = newup2(128,128,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up1 = newup2(64,64,out_ch=32,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up0 = newup2(32,32,out_ch=32,use_dropblock=dropblock,drop_p=dropout)
        else:
            self.movable_up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)

        self.moving_dropout = nn.Dropout(p=dropout)
        if self.cat or self.mgmfa:
            self.moving_outc = outconv(64, moving_n_class)
        else: self.moving_outc = outconv(32, moving_n_class)

        self.movable_dropout = nn.Dropout(p=dropout)
        self.movable_outc = outconv(32,movable_n_class)# 2*n_height(32)
        if self.cca_out:
            self.CCA_OUT = CrossCoAttention(64,64)
            
    def forward(self, x, res, r2p_matrix, p2r_matrix, range_res):
        self.r2p_matrix = r2p_matrix
        # x : B 32 480 360
        # res : B 8 480 360
        c0_appearance = self.inc(x)                # B 32 480 360 -> B 32 480 360
        c1_appearance = self.down1(c0_appearance)  # B 32 480 360 -> B 64 240 180
        c2_appearance = self.down2(c1_appearance)  # B 64 240 180 -> B 128 120 90
        c3_appearance = self.down3(c2_appearance)  # B 128 120 90 -> B 256 60 45
        c4_appearance = self.down4(c3_appearance)  # B 256 60 45  -> B 256 30 22

        c0_motion_range = self.res_inc_range(range_res)
        c1_motion_range = self.res_down1_range(c0_motion_range)
        c2_motion_range = self.res_down2_range(c1_motion_range)
        c3_motion_range = self.res_down3_range(c2_motion_range)
        c4_motion_range = self.res_down4_range(c3_motion_range)
        
        c0_motion = self.res_inc(res) # B 8 480 360 -> B 16 480 360
        c0_motion , _ =  self.CGM0(c0_motion,c0_appearance)# B 16 480 360 | B 32 480 360 -> B 16 480 360 | B 32 480 360  

        c1_motion = self.res_down1(c0_motion) # B 16 480 360 -> B 32 240 180
        c1_motion = self.flow_l2_r2p(self.r2p_matrix, c1_motion, c1_motion_range)
        # c1_motion = self.vss2(c1_motion,240,180)
        c1_motion , _ =  self.CGM1(c1_motion,c1_appearance)# B 32 240 180 | B 64 240 180 -> B 32 240 180 | B 64 240 180 

        c2_motion = self.res_down2(c1_motion) # B 32 240 180 -> B 64 120 90
        c2_motion = self.flow_l3_r2p(self.r2p_matrix, c2_motion, c2_motion_range)
        # c2_motion = self.vss3(c2_motion,120,90)
        c2_motion , _ =  self.CGM2(c2_motion,c2_appearance)# B 64 120 90 | B 128 120 90 -> B 64 120 90 | B 128 120 90 

        c3_motion = self.res_down3(c2_motion) # B 64 120 90 -> B 128 60 45
        c3_motion = self.flow_l4_r2p(self.r2p_matrix, c3_motion, c3_motion_range)
        # c3_motion = self.vss4(c3_motion,60,45)
        c3_motion , _ =  self.CGM3(c3_motion,c3_appearance)# B 128 60 45 |  B 256 60 45 -> B 128 60 45 |  B 256 60 45 

        c4_motion = self.res_down4(c3_motion) # B 128 60 45 -> B 128 30 22 
        c4_motion = self.flow_l5_r2p(self.r2p_matrix, c4_motion, c4_motion_range)
        c4_motion = self.vss4(c4_motion,30,22)
        c4_motion , _ =  self.CGM4(c4_motion,c4_appearance)#  B 128 30 22 | B 256 30 22  ->  B 128 30 22  B 256 30 22 

        movable_out3 = self.movable_up3(c4_appearance,c3_appearance) # B 256 30 22 | B 256 60 45 -> B 128 60 45
        movable_out2 = self.movable_up2(movable_out3,c2_appearance) # B 128 60 45 | B 128 120 90 -> B 64 120 90
        movable_out1 = self.movable_up1(movable_out2,c1_appearance) # B 64 120 90 | B 64 240 180 -> B 32 240 180
        movable_out0 = self.movable_up0(movable_out1,c0_appearance) # B 32 240 180 | B 32 480 360 -> B 32 480 360
        movable_out = self.movable_outc(self.movable_dropout(movable_out0))

        if self.cat:
            c3_motion = torch.cat([c3_motion,movable_out3],dim=1)
        elif self.cat_block:
            c3_motion = self.CCA3(c3_motion,movable_out3)
        elif self.mgmfa:
            c3_motion = self.MGMFA3(c3_motion,movable_out3)
        else :
            c3_motion = c3_motion
        moving_out3 = self.up3(c4_motion, c3_motion)

        if self.cat:
            c2_motion = torch.cat([c2_motion,movable_out2],dim=1)
        elif self.cat_block:
            c2_motion = self.CCA2(c2_motion,movable_out2)
        elif self.mgmfa:
            c2_motion = self.MGMFA2(c2_motion,movable_out2)
        else:
            c2_motion = c2_motion
        moving_out2 = self.up2(moving_out3, c2_motion)
        
        if self.cat:
            c1_motion = torch.cat([c1_motion,movable_out1],dim=1)
        elif self.cat_block:
            c1_motion = self.CCA1(c1_motion,movable_out1)
        elif self.mgmfa:
            c1_motion = self.MGMFA1(c1_motion,movable_out1)
        else :
            c1_motion = c1_motion
        moving_out1 = self.up1(moving_out2, c1_motion)

        if self.cat:
            c0_motion = torch.cat([c0_motion,movable_out0],dim=1)
        elif self.cat_block:
            c0_motion = self.CCA0(c0_motion,movable_out0)  
        elif self.mgmfa:
            c0_motion = self.MGMFA0(c0_motion,movable_out0)
        else:
            c0_motion = c0_motion      
        moving_out0 = self.up0(moving_out1, c0_motion)


        moving_out = self.moving_outc(self.moving_dropout(moving_out0))
        
        if self.cca_out:
            moving_out = self.CCA_OUT(moving_out,moving_out)
        return moving_out,movable_out


class UNet_8_pr_up(nn.Module):
    '''模型结构逻辑换了，加入了新的注意力机制模块到上采样过程中，加入RV残差图
    '''
    def __init__(self, moving_n_class, movable_n_class,n_height, residual, dilation, group_conv, input_batch_norm, dropout, circular_padding,
                 dropblock,*args):
        super(UNet_8_pr_up, self).__init__()
        self.cat = True
        self.cat_block = False
        self.mgmfa = False
        self.cca_out = False  # 对输出进行直接的特征提取
        self.newup2 = False
        print("是否是cat:",self.cat)
        print("是否是cca:",self.cat_block)
        print("是否是mgmfa:",self.mgmfa)
        print("是否是cca_out:",self.cca_out)        
        print("newup2",self.newup2 )
        self.inc = inconv(n_height, 32, dilation, input_batch_norm, circular_padding)
        self.down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.down4 = down(256, 256, dilation, group_conv, circular_padding)

        self.res_inc = inconv(residual, 32, dilation, input_batch_norm, circular_padding)
        self.res_down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.res_down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.res_down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.res_down4 = down(256, 256, dilation, group_conv, circular_padding)

        self.res_inc_range = inconv(residual, 32, dilation, input_batch_norm, circular_padding)
        self.res_down1_range = down(32, 64, dilation, group_conv, circular_padding)
        self.res_down2_range = down(64, 128, dilation, group_conv, circular_padding)
        self.res_down3_range = down2(128, 256, dilation, group_conv, circular_padding)
        self.res_down4_range = down2(256, 256, dilation, group_conv, circular_padding)

        self.flow_l2_r2p = R2B_flow_new3(64)
        self.flow_l3_r2p = R2B_flow_new3(128)
        self.flow_l4_r2p = R2B_flow_new3(256)
        self.flow_l5_r2p = R2B_flow_new3(256)

        self.CGM0 = CAG(channel_a=32, channel_m=32, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM1 = CAG(channel_a=64, channel_m=64, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM2 = CAG(channel_a=128, channel_m=128, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM3 = CAG(channel_a=256, channel_m=256, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM4 = CAG(channel_a=256, channel_m=256, circular_padding=circular_padding, group_conv=group_conv) 

        # self.CGM0 = CAG(channel_a=16, channel_m=32, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM1 = CAG(channel_a=32, channel_m=64, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM2 = CAG(channel_a=64, channel_m=128, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM3 = CAG(channel_a=128, channel_m=256, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM4 = CAG(channel_a=128, channel_m=256, circular_padding=circular_padding, group_conv=group_conv)
        if self.cat:
            if self.newup2 :
                self.up3 = newup2(256,384,out_ch=192,use_dropblock=dropblock,drop_p=dropout)
                self.up2 = newup2(192,192,out_ch=96,use_dropblock=dropblock,drop_p=dropout)
                self.up1 = newup2(96,96,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
                self.up0 = newup2(64,64,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
            else:
                # self.up3 = up2(256,384,out_ch=192,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                # self.up2 = up2(192,192,out_ch=96,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                # self.up1 = up2(96,96,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                # self.up0 = up2(64,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up3 = up2(256,256,out_ch=192,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up2 = up2(192,128,out_ch=96,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up1 = up2(96,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up0 = up2(64,32,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        elif self.cat_block:
            self.up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.CCA3 = CrossCoAttention(256,128)
            self.CCA2 = CrossCoAttention(128,64)
            self.CCA1 = CrossCoAttention(64,32)
            self.CCA0 = CrossCoAttention(32,32)
        elif self.mgmfa:
            self.up3 = up2(256,384,out_ch=192,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(192,192,out_ch=96,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(96,96,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(64,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.MGMFA3 = MGMFA(256,128)
            self.MGMFA2 = MGMFA(128,64)
            self.MGMFA1 = MGMFA(64,32)
            self.MGMFA0 = MGMFA(32,32)
        else : 
            self.up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)

        # self.up3 = up2(128,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up2 = up2(128,128,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up1 = up2(128,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up0 = up2(64,48,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        if self.newup2 and self.cat:
            self.movable_up3 = newup2(256,256,out_ch=128,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up2 = newup2(128,128,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up1 = newup2(64,64,out_ch=32,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up0 = newup2(32,32,out_ch=32,use_dropblock=dropblock,drop_p=dropout)
        else:
            self.movable_up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)

        self.catup0 = catblock(32,32)
        self.catup1 = catblock(64,32)
        self.catup2 = catblock(128,64)  
        self.catup3 = catblock(256,128)  
        
        self.moving_dropout = nn.Dropout(p=dropout)
        if self.cat or self.mgmfa:
            self.moving_outc = outconv(64, moving_n_class)
        else: self.moving_outc = outconv(32, moving_n_class)

        self.movable_dropout = nn.Dropout(p=dropout)
        self.movable_outc = outconv(32,movable_n_class)# 2*n_height(32)
        if self.cca_out:
            self.CCA_OUT = CrossCoAttention(64,64)
            
    def forward(self, x, res, r2p_matrix, p2r_matrix, range_res):
        self.r2p_matrix = r2p_matrix
        # x : B 32 480 360
        # res : B 8 480 360
        c0_appearance = self.inc(x)                # B 32 480 360 -> B 32 480 360
        c1_appearance = self.down1(c0_appearance)  # B 32 480 360 -> B 64 240 180
        c2_appearance = self.down2(c1_appearance)  # B 64 240 180 -> B 128 120 90
        c3_appearance = self.down3(c2_appearance)  # B 128 120 90 -> B 256 60 45
        c4_appearance = self.down4(c3_appearance)  # B 256 60 45  -> B 256 30 22

        c0_motion_range = self.res_inc_range(range_res)
        c1_motion_range = self.res_down1_range(c0_motion_range)
        c2_motion_range = self.res_down2_range(c1_motion_range)
        c3_motion_range = self.res_down3_range(c2_motion_range)
        c4_motion_range = self.res_down4_range(c3_motion_range)
        
        c0_motion = self.res_inc(res) # B 8 480 360 -> B 16 480 360
        c0_motion , _ =  self.CGM0(c0_motion,c0_appearance)# B 16 480 360 | B 32 480 360 -> B 16 480 360 | B 32 480 360  

        c1_motion = self.res_down1(c0_motion) # B 16 480 360 -> B 32 240 180
        c1_motion = self.flow_l2_r2p(self.r2p_matrix, c1_motion, c1_motion_range)
        c1_motion , _ =  self.CGM1(c1_motion,c1_appearance)# B 32 240 180 | B 64 240 180 -> B 32 240 180 | B 64 240 180 

        c2_motion = self.res_down2(c1_motion) # B 32 240 180 -> B 64 120 90
        c2_motion = self.flow_l3_r2p(self.r2p_matrix, c2_motion, c2_motion_range)
        c2_motion , _ =  self.CGM2(c2_motion,c2_appearance)# B 64 120 90 | B 128 120 90 -> B 64 120 90 | B 128 120 90 

        c3_motion = self.res_down3(c2_motion) # B 64 120 90 -> B 128 60 45
        c3_motion = self.flow_l4_r2p(self.r2p_matrix, c3_motion, c3_motion_range)
        c3_motion , _ =  self.CGM3(c3_motion,c3_appearance)# B 128 60 45 |  B 256 60 45 -> B 128 60 45 |  B 256 60 45 

        c4_motion = self.res_down4(c3_motion) # B 128 60 45 -> B 128 30 22 
        c4_motion = self.flow_l5_r2p(self.r2p_matrix, c4_motion, c4_motion_range)
        c4_motion , _ =  self.CGM4(c4_motion,c4_appearance)#  B 128 30 22 | B 256 30 22  ->  B 128 30 22  B 256 30 22 

        movable_out3 = self.movable_up3(c4_appearance,c3_appearance) # B 256 30 22 | B 256 60 45 -> B 128 60 45
        movable_out2 = self.movable_up2(movable_out3,c2_appearance) # B 128 60 45 | B 128 120 90 -> B 64 120 90
        movable_out1 = self.movable_up1(movable_out2,c1_appearance) # B 64 120 90 | B 64 240 180 -> B 32 240 180
        movable_out0 = self.movable_up0(movable_out1,c0_appearance) # B 32 240 180 | B 32 480 360 -> B 32 480 360
        movable_out = self.movable_outc(self.movable_dropout(movable_out0))

        if self.cat:
            # c3_motion = torch.cat([c3_motion,movable_out3],dim=1)
            c3_motion = self.catup3(c3_motion,movable_out3)    # 256, 128 -> 256
        elif self.cat_block:
            c3_motion = self.CCA3(c3_motion,movable_out3)
        elif self.mgmfa:
            c3_motion = self.MGMFA3(c3_motion,movable_out3)
        else :
            c3_motion = c3_motion
        moving_out3 = self.up3(c4_motion, c3_motion)

        if self.cat:
            # c2_motion = torch.cat([c2_motion,movable_out2],dim=1)
            c2_motion = self.catup2(c2_motion,movable_out2)     # 128, 64 -> 128
        elif self.cat_block:
            c2_motion = self.CCA2(c2_motion,movable_out2)
        elif self.mgmfa:
            c2_motion = self.MGMFA2(c2_motion,movable_out2)
        else:
            c2_motion = c2_motion
        moving_out2 = self.up2(moving_out3, c2_motion)
        
        if self.cat:
            # c1_motion = torch.cat([c1_motion,movable_out1],dim=1)
            c1_motion = self.catup1(c1_motion,movable_out1)    # 64, 32 -> 64
        elif self.cat_block:
            c1_motion = self.CCA1(c1_motion,movable_out1)
        elif self.mgmfa:
            c1_motion = self.MGMFA1(c1_motion,movable_out1)
        else :
            c1_motion = c1_motion
        moving_out1 = self.up1(moving_out2, c1_motion)

        if self.cat:
            # c0_motion = torch.cat([c0_motion,movable_out0],dim=1)
            c0_motion = self.catup0(c0_motion,movable_out0)    # 32, 32 ->32
        elif self.cat_block:
            c0_motion = self.CCA0(c0_motion,movable_out0)  
        elif self.mgmfa:
            c0_motion = self.MGMFA0(c0_motion,movable_out0)
        else:
            c0_motion = c0_motion      
        moving_out0 = self.up0(moving_out1, c0_motion)


        moving_out = self.moving_outc(self.moving_dropout(moving_out0))
        
        if self.cca_out:
            moving_out = self.CCA_OUT(moving_out,moving_out)
        return moving_out,movable_out

class UNet_8_pr2(nn.Module):
    '''
    '''
    def __init__(self, moving_n_class, movable_n_class,n_height, residual, dilation, group_conv, input_batch_norm, dropout, circular_padding,
                 dropblock,*args):
        super(UNet_8_pr2, self).__init__()
        self.cat = True
        self.cat_block = False
        self.mgmfa = False
        self.cca_out = False  # 对输出进行直接的特征提取
        self.newup2 = False
        print("是否是cat:",self.cat)
        print("是否是cca:",self.cat_block)
        print("是否是mgmfa:",self.mgmfa)
        print("是否是cca_out:",self.cca_out)        
        print("newup2",self.newup2 )
        self.inc = inconv(n_height, 32, dilation, input_batch_norm, circular_padding)
        self.down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.down4 = down(256, 256, dilation, group_conv, circular_padding)

        self.res_inc = inconv(residual, 32, dilation, input_batch_norm, circular_padding)
        self.res_down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.res_down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.res_down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.res_down4 = down(256, 256, dilation, group_conv, circular_padding)

        self.res_inc_range = inconv(residual, 32, dilation, input_batch_norm, circular_padding)
        self.res_down1_range = down(32, 64, dilation, group_conv, circular_padding)
        self.res_down2_range = down(64, 128, dilation, group_conv, circular_padding)
        self.res_down3_range = down2(128, 256, dilation, group_conv, circular_padding)
        self.res_down4_range = down2(256, 256, dilation, group_conv, circular_padding)

        self.flow_l2_r2p = R2B_flow_new3(64)
        self.flow_l3_r2p = R2B_flow_new3(128)
        self.flow_l4_r2p = R2B_flow_new3(256)
        self.flow_l5_r2p = R2B_flow_new3(256)
        self.flow_l2_r2p_cur = R2B_flow_new3(64)
        self.flow_l3_r2p_cur = R2B_flow_new3(128)
        self.flow_l4_r2p_cur = R2B_flow_new3(256)
        self.flow_l5_r2p_cur = R2B_flow_new3(256)
        
        self.CGM0 = CAG(channel_a=32, channel_m=32, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM1 = CAG(channel_a=64, channel_m=64, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM2 = CAG(channel_a=128, channel_m=128, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM3 = CAG(channel_a=256, channel_m=256, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM4 = CAG(channel_a=256, channel_m=256, circular_padding=circular_padding, group_conv=group_conv) 

        # self.CGM0 = CAG(channel_a=16, channel_m=32, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM1 = CAG(channel_a=32, channel_m=64, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM2 = CAG(channel_a=64, channel_m=128, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM3 = CAG(channel_a=128, channel_m=256, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM4 = CAG(channel_a=128, channel_m=256, circular_padding=circular_padding, group_conv=group_conv)
        if self.cat:
            if self.newup2 :
                self.up3 = newup2(256,384,out_ch=192,use_dropblock=dropblock,drop_p=dropout)
                self.up2 = newup2(192,192,out_ch=96,use_dropblock=dropblock,drop_p=dropout)
                self.up1 = newup2(96,96,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
                self.up0 = newup2(64,64,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
            else:
                self.up3 = up2(256,384,out_ch=192,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up2 = up2(192,192,out_ch=96,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up1 = up2(96,96,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up0 = up2(64,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        elif self.cat_block:
            self.up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.CCA3 = CrossCoAttention(256,128)
            self.CCA2 = CrossCoAttention(128,64)
            self.CCA1 = CrossCoAttention(64,32)
            self.CCA0 = CrossCoAttention(32,32)
        elif self.mgmfa:
            self.up3 = up2(256,384,out_ch=192,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(192,192,out_ch=96,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(96,96,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(64,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.MGMFA3 = MGMFA(256,128)
            self.MGMFA2 = MGMFA(128,64)
            self.MGMFA1 = MGMFA(64,32)
            self.MGMFA0 = MGMFA(32,32)
        else : 
            self.up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)

        # self.up3 = up2(128,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up2 = up2(128,128,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up1 = up2(128,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up0 = up2(64,48,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        if self.newup2 and self.cat:
            self.movable_up3 = newup2(256,256,out_ch=128,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up2 = newup2(128,128,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up1 = newup2(64,64,out_ch=32,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up0 = newup2(32,32,out_ch=32,use_dropblock=dropblock,drop_p=dropout)
        else:
            self.movable_up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)

        self.moving_dropout = nn.Dropout(p=dropout)
        if self.cat or self.mgmfa:
            self.moving_outc = outconv(64, moving_n_class)
        else: self.moving_outc = outconv(32, moving_n_class)

        self.movable_dropout = nn.Dropout(p=dropout)
        self.movable_outc = outconv(32,movable_n_class)# 2*n_height(32)
        if self.cca_out:
            self.CCA_OUT = CrossCoAttention(64,64)
            
    def forward(self, x, res, r2p_matrix, p2r_matrix, range_res):
        self.r2p_matrix = r2p_matrix
        # x : B 32 480 360
        # res : B 8 480 360
        c0_appearance = self.inc(x)                # B 32 480 360 -> B 32 480 360
        c1_appearance = self.down1(c0_appearance)  # B 32 480 360 -> B 64 240 180
        c2_appearance = self.down2(c1_appearance)  # B 64 240 180 -> B 128 120 90
        c3_appearance = self.down3(c2_appearance)  # B 128 120 90 -> B 256 60 45
        c4_appearance = self.down4(c3_appearance)  # B 256 60 45  -> B 256 30 22

        c0_motion_range = self.res_inc_range(range_res)
        c1_motion_range = self.res_down1_range(c0_motion_range)
        c2_motion_range = self.res_down2_range(c1_motion_range)
        c3_motion_range = self.res_down3_range(c2_motion_range)
        c4_motion_range = self.res_down4_range(c3_motion_range)
        
        c0_motion = self.res_inc(res) # B 8 480 360 -> B 16 480 360
        c0_motion , _ =  self.CGM0(c0_motion,c0_appearance)# B 16 480 360 | B 32 480 360 -> B 16 480 360 | B 32 480 360  

        c1_motion = self.res_down1(c0_motion) # B 16 480 360 -> B 32 240 180
        c1_motion = self.flow_l2_r2p(self.r2p_matrix, c1_motion, c1_motion_range)
        c1_appearance = self.flow_l2_r2p_cur(self.r2p_matrix, c1_appearance, c1_motion_range)
        c1_motion , _ =  self.CGM1(c1_motion,c1_appearance)# B 32 240 180 | B 64 240 180 -> B 32 240 180 | B 64 240 180 

        c2_motion = self.res_down2(c1_motion) # B 32 240 180 -> B 64 120 90
        c2_motion = self.flow_l3_r2p(self.r2p_matrix, c2_motion, c2_motion_range)
        c2_appearance = self.flow_l3_r2p_cur(self.r2p_matrix, c2_appearance, c2_motion_range)
        c2_motion , _ =  self.CGM2(c2_motion,c2_appearance)# B 64 120 90 | B 128 120 90 -> B 64 120 90 | B 128 120 90 

        c3_motion = self.res_down3(c2_motion) # B 64 120 90 -> B 128 60 45
        c3_motion = self.flow_l4_r2p(self.r2p_matrix, c3_motion, c3_motion_range)
        c3_appearance = self.flow_l4_r2p_cur(self.r2p_matrix, c3_appearance, c3_motion_range)
        c3_motion , _ =  self.CGM3(c3_motion,c3_appearance)# B 128 60 45 |  B 256 60 45 -> B 128 60 45 |  B 256 60 45 

        c4_motion = self.res_down4(c3_motion) # B 128 60 45 -> B 128 30 22 
        c4_motion = self.flow_l5_r2p(self.r2p_matrix, c4_motion, c4_motion_range)
        c4_appearance = self.flow_l5_r2p_cur(self.r2p_matrix, c4_appearance, c4_motion_range)
        c4_motion , _ =  self.CGM4(c4_motion,c4_appearance)#  B 128 30 22 | B 256 30 22  ->  B 128 30 22  B 256 30 22 

        movable_out3 = self.movable_up3(c4_appearance,c3_appearance) # B 256 30 22 | B 256 60 45 -> B 128 60 45
        movable_out2 = self.movable_up2(movable_out3,c2_appearance) # B 128 60 45 | B 128 120 90 -> B 64 120 90
        movable_out1 = self.movable_up1(movable_out2,c1_appearance) # B 64 120 90 | B 64 240 180 -> B 32 240 180
        movable_out0 = self.movable_up0(movable_out1,c0_appearance) # B 32 240 180 | B 32 480 360 -> B 32 480 360
        movable_out = self.movable_outc(self.movable_dropout(movable_out0))

        if self.cat:
            c3_motion = torch.cat([c3_motion,movable_out3],dim=1)
        elif self.cat_block:
            c3_motion = self.CCA3(c3_motion,movable_out3)
        elif self.mgmfa:
            c3_motion = self.MGMFA3(c3_motion,movable_out3)
        else :
            c3_motion = c3_motion
        moving_out3 = self.up3(c4_motion, c3_motion)

        if self.cat:
            c2_motion = torch.cat([c2_motion,movable_out2],dim=1)
        elif self.cat_block:
            c2_motion = self.CCA2(c2_motion,movable_out2)
        elif self.mgmfa:
            c2_motion = self.MGMFA2(c2_motion,movable_out2)
        else:
            c2_motion = c2_motion
        moving_out2 = self.up2(moving_out3, c2_motion)
        
        if self.cat:
            c1_motion = torch.cat([c1_motion,movable_out1],dim=1)
        elif self.cat_block:
            c1_motion = self.CCA1(c1_motion,movable_out1)
        elif self.mgmfa:
            c1_motion = self.MGMFA1(c1_motion,movable_out1)
        else :
            c1_motion = c1_motion
        moving_out1 = self.up1(moving_out2, c1_motion)

        if self.cat:
            c0_motion = torch.cat([c0_motion,movable_out0],dim=1)
        elif self.cat_block:
            c0_motion = self.CCA0(c0_motion,movable_out0)  
        elif self.mgmfa:
            c0_motion = self.MGMFA0(c0_motion,movable_out0)
        else:
            c0_motion = c0_motion      
        moving_out0 = self.up0(moving_out1, c0_motion)


        moving_out = self.moving_outc(self.moving_dropout(moving_out0))
        
        if self.cca_out:
            moving_out = self.CCA_OUT(moving_out,moving_out)
        return moving_out,movable_out

class UNet_8_pr2_LKA(nn.Module):
    '''
    '''
    def __init__(self, moving_n_class, movable_n_class,n_height, residual, dilation, group_conv, input_batch_norm, dropout, circular_padding,
                 dropblock,*args):
        super(UNet_8_pr2, self).__init__()
        self.cat = True
        self.cat_block = False
        self.mgmfa = False
        self.cca_out = False  # 对输出进行直接的特征提取
        self.newup2 = False
        print("是否是cat:",self.cat)
        print("是否是cca:",self.cat_block)
        print("是否是mgmfa:",self.mgmfa)
        print("是否是cca_out:",self.cca_out)        
        print("newup2",self.newup2 )
        self.inc = inconv(n_height, 32, dilation, input_batch_norm, circular_padding)
        self.down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.down4 = down(256, 256, dilation, group_conv, circular_padding)

        self.res_inc = inconv(residual, 32, dilation, input_batch_norm, circular_padding)
        self.res_down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.res_down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.res_down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.res_down4 = down(256, 256, dilation, group_conv, circular_padding)

        self.res_inc_range = inconv(residual, 32, dilation, input_batch_norm, circular_padding)
        self.res_down1_range = down(32, 64, dilation, group_conv, circular_padding)
        self.res_down2_range = down(64, 128, dilation, group_conv, circular_padding)
        self.res_down3_range = down2(128, 256, dilation, group_conv, circular_padding)
        self.res_down4_range = down2(256, 256, dilation, group_conv, circular_padding)

        self.flow_l2_r2p = R2B_flow_new3(64)
        self.flow_l3_r2p = R2B_flow_new3(128)
        self.flow_l4_r2p = R2B_flow_new3(256)
        self.flow_l5_r2p = R2B_flow_new3(256)
        self.flow_l2_r2p_cur = R2B_flow_new3(64)
        self.flow_l3_r2p_cur = R2B_flow_new3(128)
        self.flow_l4_r2p_cur = R2B_flow_new3(256)
        self.flow_l5_r2p_cur = R2B_flow_new3(256)
        
        self.CGM0 = CAG(channel_a=32, channel_m=32, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM1 = CAG(channel_a=64, channel_m=64, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM2 = CAG(channel_a=128, channel_m=128, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM3 = CAG(channel_a=256, channel_m=256, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM4 = CAG(channel_a=256, channel_m=256, circular_padding=circular_padding, group_conv=group_conv) 

        # self.CGM0 = CAG(channel_a=16, channel_m=32, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM1 = CAG(channel_a=32, channel_m=64, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM2 = CAG(channel_a=64, channel_m=128, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM3 = CAG(channel_a=128, channel_m=256, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM4 = CAG(channel_a=128, channel_m=256, circular_padding=circular_padding, group_conv=group_conv)
        if self.cat:
            if self.newup2 :
                self.up3 = newup2(256,384,out_ch=192,use_dropblock=dropblock,drop_p=dropout)
                self.up2 = newup2(192,192,out_ch=96,use_dropblock=dropblock,drop_p=dropout)
                self.up1 = newup2(96,96,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
                self.up0 = newup2(64,64,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
            else:
                self.up3 = up2(256,384,out_ch=192,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up2 = up2(192,192,out_ch=96,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up1 = up2(96,96,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up0 = up2(64,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        elif self.cat_block:
            self.up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.CCA3 = CrossCoAttention(256,128)
            self.CCA2 = CrossCoAttention(128,64)
            self.CCA1 = CrossCoAttention(64,32)
            self.CCA0 = CrossCoAttention(32,32)
        elif self.mgmfa:
            self.up3 = up2(256,384,out_ch=192,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(192,192,out_ch=96,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(96,96,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(64,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.MGMFA3 = MGMFA(256,128)
            self.MGMFA2 = MGMFA(128,64)
            self.MGMFA1 = MGMFA(64,32)
            self.MGMFA0 = MGMFA(32,32)
        else : 
            self.up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)

        # self.up3 = up2(128,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up2 = up2(128,128,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up1 = up2(128,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up0 = up2(64,48,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        if self.newup2 and self.cat:
            self.movable_up3 = newup2(256,256,out_ch=128,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up2 = newup2(128,128,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up1 = newup2(64,64,out_ch=32,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up0 = newup2(32,32,out_ch=32,use_dropblock=dropblock,drop_p=dropout)
        else:
            self.movable_up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)

        self.moving_dropout = nn.Dropout(p=dropout)
        if self.cat or self.mgmfa:
            self.moving_outc = outconv(64, moving_n_class)
        else: self.moving_outc = outconv(32, moving_n_class)

        self.movable_dropout = nn.Dropout(p=dropout)
        self.movable_outc = outconv(32,movable_n_class)# 2*n_height(32)
        if self.cca_out:
            self.CCA_OUT = CrossCoAttention(64,64)
            
    def forward(self, x, res, r2p_matrix, p2r_matrix, range_res):
        self.r2p_matrix = r2p_matrix
        # x : B 32 480 360
        # res : B 8 480 360
        c0_appearance = self.inc(x)                # B 32 480 360 -> B 32 480 360
        c1_appearance = self.down1(c0_appearance)  # B 32 480 360 -> B 64 240 180
        c2_appearance = self.down2(c1_appearance)  # B 64 240 180 -> B 128 120 90
        c3_appearance = self.down3(c2_appearance)  # B 128 120 90 -> B 256 60 45
        c4_appearance = self.down4(c3_appearance)  # B 256 60 45  -> B 256 30 22

        c0_motion_range = self.res_inc_range(range_res)
        c1_motion_range = self.res_down1_range(c0_motion_range)
        c2_motion_range = self.res_down2_range(c1_motion_range)
        c3_motion_range = self.res_down3_range(c2_motion_range)
        c4_motion_range = self.res_down4_range(c3_motion_range)
        
        c0_motion = self.res_inc(res) # B 8 480 360 -> B 16 480 360
        c0_motion , _ =  self.CGM0(c0_motion,c0_appearance)# B 16 480 360 | B 32 480 360 -> B 16 480 360 | B 32 480 360  

        c1_motion = self.res_down1(c0_motion) # B 16 480 360 -> B 32 240 180
        c1_motion = self.flow_l2_r2p(self.r2p_matrix, c1_motion, c1_motion_range)
        c1_appearance = self.flow_l2_r2p_cur(self.r2p_matrix, c1_appearance, c1_motion_range)
        c1_motion , _ =  self.CGM1(c1_motion,c1_appearance)# B 32 240 180 | B 64 240 180 -> B 32 240 180 | B 64 240 180 

        c2_motion = self.res_down2(c1_motion) # B 32 240 180 -> B 64 120 90
        c2_motion = self.flow_l3_r2p(self.r2p_matrix, c2_motion, c2_motion_range)
        c2_appearance = self.flow_l3_r2p_cur(self.r2p_matrix, c2_appearance, c2_motion_range)
        c2_motion , _ =  self.CGM2(c2_motion,c2_appearance)# B 64 120 90 | B 128 120 90 -> B 64 120 90 | B 128 120 90 

        c3_motion = self.res_down3(c2_motion) # B 64 120 90 -> B 128 60 45
        c3_motion = self.flow_l4_r2p(self.r2p_matrix, c3_motion, c3_motion_range)
        c3_appearance = self.flow_l4_r2p_cur(self.r2p_matrix, c3_appearance, c3_motion_range)
        c3_motion , _ =  self.CGM3(c3_motion,c3_appearance)# B 128 60 45 |  B 256 60 45 -> B 128 60 45 |  B 256 60 45 

        c4_motion = self.res_down4(c3_motion) # B 128 60 45 -> B 128 30 22 
        c4_motion = self.flow_l5_r2p(self.r2p_matrix, c4_motion, c4_motion_range)
        c4_appearance = self.flow_l5_r2p_cur(self.r2p_matrix, c4_appearance, c4_motion_range)
        c4_motion , _ =  self.CGM4(c4_motion,c4_appearance)#  B 128 30 22 | B 256 30 22  ->  B 128 30 22  B 256 30 22 

        movable_out3 = self.movable_up3(c4_appearance,c3_appearance) # B 256 30 22 | B 256 60 45 -> B 128 60 45
        movable_out2 = self.movable_up2(movable_out3,c2_appearance) # B 128 60 45 | B 128 120 90 -> B 64 120 90
        movable_out1 = self.movable_up1(movable_out2,c1_appearance) # B 64 120 90 | B 64 240 180 -> B 32 240 180
        movable_out0 = self.movable_up0(movable_out1,c0_appearance) # B 32 240 180 | B 32 480 360 -> B 32 480 360
        movable_out = self.movable_outc(self.movable_dropout(movable_out0))

        if self.cat:
            c3_motion = torch.cat([c3_motion,movable_out3],dim=1)
        elif self.cat_block:
            c3_motion = self.CCA3(c3_motion,movable_out3)
        elif self.mgmfa:
            c3_motion = self.MGMFA3(c3_motion,movable_out3)
        else :
            c3_motion = c3_motion
        moving_out3 = self.up3(c4_motion, c3_motion)

        if self.cat:
            c2_motion = torch.cat([c2_motion,movable_out2],dim=1)
        elif self.cat_block:
            c2_motion = self.CCA2(c2_motion,movable_out2)
        elif self.mgmfa:
            c2_motion = self.MGMFA2(c2_motion,movable_out2)
        else:
            c2_motion = c2_motion
        moving_out2 = self.up2(moving_out3, c2_motion)
        
        if self.cat:
            c1_motion = torch.cat([c1_motion,movable_out1],dim=1)
        elif self.cat_block:
            c1_motion = self.CCA1(c1_motion,movable_out1)
        elif self.mgmfa:
            c1_motion = self.MGMFA1(c1_motion,movable_out1)
        else :
            c1_motion = c1_motion
        moving_out1 = self.up1(moving_out2, c1_motion)

        if self.cat:
            c0_motion = torch.cat([c0_motion,movable_out0],dim=1)
        elif self.cat_block:
            c0_motion = self.CCA0(c0_motion,movable_out0)  
        elif self.mgmfa:
            c0_motion = self.MGMFA0(c0_motion,movable_out0)
        else:
            c0_motion = c0_motion      
        moving_out0 = self.up0(moving_out1, c0_motion)


        moving_out = self.moving_outc(self.moving_dropout(moving_out0))
        
        if self.cca_out:
            moving_out = self.CCA_OUT(moving_out,moving_out)
        return moving_out,movable_out
                
class UNet_9_pr(nn.Module):
    '''模型结构逻辑换了，加入了新的注意力机制模块到上采样过程中，新的VSS-r2p
    '''
    def __init__(self, moving_n_class, movable_n_class,n_height, residual, dilation, group_conv, input_batch_norm, dropout, circular_padding,
                 dropblock,*args):
        super(UNet_9_pr, self).__init__()
        self.cat = True
        self.cat_block = False
        self.mgmfa = False
        self.cca_out = False  # 对输出进行直接的特征提取
        self.newup2 = False
        print("是否是cat:",self.cat)
        print("是否是cca:",self.cat_block)
        print("是否是mgmfa:",self.mgmfa)
        print("是否是cca_out:",self.cca_out)        
        print("newup2",self.newup2 )
        self.inc = inconv(n_height, 32, dilation, input_batch_norm, circular_padding)
        self.down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.down4 = down(256, 256, dilation, group_conv, circular_padding)

        self.res_inc = inconv(residual, 32, dilation, input_batch_norm, circular_padding)
        self.res_down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.res_down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.res_down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.res_down4 = down(256, 256, dilation, group_conv, circular_padding)

        self.res_inc_range = inconv(residual, 32, dilation, input_batch_norm, circular_padding)
        self.res_down1_range = down(32, 64, dilation, group_conv, circular_padding)
        self.res_down2_range = down(64, 128, dilation, group_conv, circular_padding)
        self.res_down3_range = down2(128, 256, dilation, group_conv, circular_padding)
        self.res_down4_range = down2(256, 256, dilation, group_conv, circular_padding)

        self.flow_l2_r2p = R2B_flow_new3_VSS(64)
        self.flow_l3_r2p = R2B_flow_new3_VSS(128)
        self.flow_l4_r2p = R2B_flow_new3_VSS(256)
        self.flow_l5_r2p = R2B_flow_new3_VSS(256)

        self.CGM0 = CAG(channel_a=32, channel_m=32, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM1 = CAG(channel_a=64, channel_m=64, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM2 = CAG(channel_a=128, channel_m=128, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM3 = CAG(channel_a=256, channel_m=256, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM4 = CAG(channel_a=256, channel_m=256, circular_padding=circular_padding, group_conv=group_conv) 

        # self.CGM0 = CAG(channel_a=16, channel_m=32, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM1 = CAG(channel_a=32, channel_m=64, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM2 = CAG(channel_a=64, channel_m=128, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM3 = CAG(channel_a=128, channel_m=256, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM4 = CAG(channel_a=128, channel_m=256, circular_padding=circular_padding, group_conv=group_conv)
        if self.cat:
            if self.newup2 :
                self.up3 = newup2(256,384,out_ch=192,use_dropblock=dropblock,drop_p=dropout)
                self.up2 = newup2(192,192,out_ch=96,use_dropblock=dropblock,drop_p=dropout)
                self.up1 = newup2(96,96,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
                self.up0 = newup2(64,64,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
            else:
                self.up3 = up2(256,384,out_ch=192,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up2 = up2(192,192,out_ch=96,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up1 = up2(96,96,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up0 = up2(64,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        elif self.cat_block:
            self.up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.CCA3 = CrossCoAttention(256,128)
            self.CCA2 = CrossCoAttention(128,64)
            self.CCA1 = CrossCoAttention(64,32)
            self.CCA0 = CrossCoAttention(32,32)
        elif self.mgmfa:
            self.up3 = up2(256,384,out_ch=192,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(192,192,out_ch=96,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(96,96,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(64,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.MGMFA3 = MGMFA(256,128)
            self.MGMFA2 = MGMFA(128,64)
            self.MGMFA1 = MGMFA(64,32)
            self.MGMFA0 = MGMFA(32,32)
        else : 
            self.up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)

        # self.up3 = up2(128,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up2 = up2(128,128,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up1 = up2(128,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up0 = up2(64,48,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        if self.newup2 and self.cat:
            self.movable_up3 = newup2(256,256,out_ch=128,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up2 = newup2(128,128,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up1 = newup2(64,64,out_ch=32,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up0 = newup2(32,32,out_ch=32,use_dropblock=dropblock,drop_p=dropout)
        else:
            self.movable_up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)

        self.moving_dropout = nn.Dropout(p=dropout)
        if self.cat or self.mgmfa:
            self.moving_outc = outconv(64, moving_n_class)
        else: self.moving_outc = outconv(32, moving_n_class)

        self.movable_dropout = nn.Dropout(p=dropout)
        self.movable_outc = outconv(32,movable_n_class)# 2*n_height(32)
        if self.cca_out:
            self.CCA_OUT = CrossCoAttention(64,64)
            
    def forward(self, x, res, r2p_matrix, p2r_matrix, range_res):
        self.r2p_matrix = r2p_matrix
        # x : B 32 480 360
        # res : B 8 480 360
        c0_appearance = self.inc(x)                # B 32 480 360 -> B 32 480 360
        c1_appearance = self.down1(c0_appearance)  # B 32 480 360 -> B 64 240 180
        c2_appearance = self.down2(c1_appearance)  # B 64 240 180 -> B 128 120 90
        c3_appearance = self.down3(c2_appearance)  # B 128 120 90 -> B 256 60 45
        c4_appearance = self.down4(c3_appearance)  # B 256 60 45  -> B 256 30 22

        c0_motion_range = self.res_inc_range(range_res)
        c1_motion_range = self.res_down1_range(c0_motion_range)
        c2_motion_range = self.res_down2_range(c1_motion_range)
        c3_motion_range = self.res_down3_range(c2_motion_range)
        c4_motion_range = self.res_down4_range(c3_motion_range)
        
        c0_motion = self.res_inc(res) # B 8 480 360 -> B 16 480 360
        c0_motion , _ =  self.CGM0(c0_motion,c0_appearance)# B 16 480 360 | B 32 480 360 -> B 16 480 360 | B 32 480 360  

        c1_motion = self.res_down1(c0_motion) # B 16 480 360 -> B 32 240 180
        c1_motion = self.flow_l2_r2p(self.r2p_matrix, c1_motion, c1_motion_range)
        c1_motion , _ =  self.CGM1(c1_motion,c1_appearance)# B 32 240 180 | B 64 240 180 -> B 32 240 180 | B 64 240 180 

        c2_motion = self.res_down2(c1_motion) # B 32 240 180 -> B 64 120 90
        c2_motion = self.flow_l3_r2p(self.r2p_matrix, c2_motion, c2_motion_range)
        c2_motion , _ =  self.CGM2(c2_motion,c2_appearance)# B 64 120 90 | B 128 120 90 -> B 64 120 90 | B 128 120 90 

        c3_motion = self.res_down3(c2_motion) # B 64 120 90 -> B 128 60 45
        c3_motion = self.flow_l4_r2p(self.r2p_matrix, c3_motion, c3_motion_range)
        c3_motion , _ =  self.CGM3(c3_motion,c3_appearance)# B 128 60 45 |  B 256 60 45 -> B 128 60 45 |  B 256 60 45 

        c4_motion = self.res_down4(c3_motion) # B 128 60 45 -> B 128 30 22 
        c4_motion = self.flow_l5_r2p(self.r2p_matrix, c4_motion, c4_motion_range)
        c4_motion , _ =  self.CGM4(c4_motion,c4_appearance)#  B 128 30 22 | B 256 30 22  ->  B 128 30 22  B 256 30 22 

        movable_out3 = self.movable_up3(c4_appearance,c3_appearance) # B 256 30 22 | B 256 60 45 -> B 128 60 45
        movable_out2 = self.movable_up2(movable_out3,c2_appearance) # B 128 60 45 | B 128 120 90 -> B 64 120 90
        movable_out1 = self.movable_up1(movable_out2,c1_appearance) # B 64 120 90 | B 64 240 180 -> B 32 240 180
        movable_out0 = self.movable_up0(movable_out1,c0_appearance) # B 32 240 180 | B 32 480 360 -> B 32 480 360
        movable_out = self.movable_outc(self.movable_dropout(movable_out0))

        if self.cat:
            c3_motion = torch.cat([c3_motion,movable_out3],dim=1)
        elif self.cat_block:
            c3_motion = self.CCA3(c3_motion,movable_out3)
        elif self.mgmfa:
            c3_motion = self.MGMFA3(c3_motion,movable_out3)
        else :
            c3_motion = c3_motion
        moving_out3 = self.up3(c4_motion, c3_motion)

        if self.cat:
            c2_motion = torch.cat([c2_motion,movable_out2],dim=1)
        elif self.cat_block:
            c2_motion = self.CCA2(c2_motion,movable_out2)
        elif self.mgmfa:
            c2_motion = self.MGMFA2(c2_motion,movable_out2)
        else:
            c2_motion = c2_motion
        moving_out2 = self.up2(moving_out3, c2_motion)
        
        if self.cat:
            c1_motion = torch.cat([c1_motion,movable_out1],dim=1)
        elif self.cat_block:
            c1_motion = self.CCA1(c1_motion,movable_out1)
        elif self.mgmfa:
            c1_motion = self.MGMFA1(c1_motion,movable_out1)
        else :
            c1_motion = c1_motion
        moving_out1 = self.up1(moving_out2, c1_motion)

        if self.cat:
            c0_motion = torch.cat([c0_motion,movable_out0],dim=1)
        elif self.cat_block:
            c0_motion = self.CCA0(c0_motion,movable_out0)  
        elif self.mgmfa:
            c0_motion = self.MGMFA0(c0_motion,movable_out0)
        else:
            c0_motion = c0_motion      
        moving_out0 = self.up0(moving_out1, c0_motion)


        moving_out = self.moving_outc(self.moving_dropout(moving_out0))
        
        if self.cca_out:
            moving_out = self.CCA_OUT(moving_out,moving_out)
        return moving_out,movable_out

class UNet_8_pr_VSS4(nn.Module):
    '''模型结构逻辑换了，加入了新的注意力机制模块到上采样过程中
    '''
    def __init__(self, moving_n_class, movable_n_class,n_height, residual, dilation, group_conv, input_batch_norm, dropout, circular_padding,
                 dropblock,*args):
        super(UNet_8_pr_VSS4, self).__init__()
        self.cat = True
        self.cat_block = False
        self.mgmfa = False
        self.cca_out = False  # 对输出进行直接的特征提取
        self.newup2 = False
        print("是否是cat:",self.cat)
        print("是否是cca:",self.cat_block)
        print("是否是mgmfa:",self.mgmfa)
        print("是否是cca_out:",self.cca_out)        
        print("newup2",self.newup2 )
        self.inc = inconv(n_height, 32, dilation, input_batch_norm, circular_padding)
        self.down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.down4 = down(256, 256, dilation, group_conv, circular_padding)

        self.res_inc = inconv(residual, 32, dilation, input_batch_norm, circular_padding)
        self.res_down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.res_down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.res_down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.res_down4 = down(256, 256, dilation, group_conv, circular_padding)

        self.res_inc_range = inconv(residual, 32, dilation, input_batch_norm, circular_padding)
        self.res_down1_range = down(32, 64, dilation, group_conv, circular_padding)
        self.res_down2_range = down(64, 128, dilation, group_conv, circular_padding)
        self.res_down3_range = down(128, 256, dilation, group_conv, circular_padding)
        self.res_down4_range = down2(256, 256, dilation, group_conv, circular_padding)

        self.flow_l2_r2p = R2B_flow_new3(64)
        self.flow_l3_r2p = R2B_flow_new3(128)
        self.flow_l4_r2p = R2B_flow_new3(256)
        self.flow_l5_r2p = R2B_flow_new3(256)
        # self.vss1 = Block_VSS(3,64,192,64,hswish(),VSSBlock(64),1)
        # self.vss2 = Block_VSS(3,96,256,96,hswish(),VSSBlock(96),1)
        # self.vss3 = Block_VSS(3,192,512,192,hswish(),VSSBlock(192),1)
        # self.vss4 = Block_VSS(3,256,512,256,hswish(),VSSBlock(256),1)

        self.CGM0 = CAG(channel_a=32, channel_m=32, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM1 = CAG_VSS(channel_a=64, channel_m=64, circular_padding=circular_padding, group_conv=group_conv,n=1)
        self.CGM2 = CAG_VSS(channel_a=128, channel_m=128, circular_padding=circular_padding, group_conv=group_conv,n=1)
        self.CGM3 = CAG_VSS(channel_a=256, channel_m=256, circular_padding=circular_padding, group_conv=group_conv,n=1)
        self.CGM4 = CAG(channel_a=256, channel_m=256, circular_padding=circular_padding, group_conv=group_conv) 

        # self.CGM0 = CAG(channel_a=16, channel_m=32, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM1 = CAG(channel_a=32, channel_m=64, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM2 = CAG(channel_a=64, channel_m=128, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM3 = CAG(channel_a=128, channel_m=256, circular_padding=circular_padding, group_conv=group_conv)
        # self.CGM4 = CAG(channel_a=128, channel_m=256, circular_padding=circular_padding, group_conv=group_conv)
        if self.cat:
            if self.newup2 :
                self.up3 = newup2(256,384,out_ch=192,use_dropblock=dropblock,drop_p=dropout)
                self.up2 = newup2(192,192,out_ch=96,use_dropblock=dropblock,drop_p=dropout)
                self.up1 = newup2(96,96,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
                self.up0 = newup2(64,64,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
            else:
                self.up3 = up2(256,384,out_ch=192,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up2 = up2(192,192,out_ch=96,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up1 = up2(96,96,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
                self.up0 = up2(64,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        elif self.cat_block:
            self.up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.CCA3 = CrossCoAttention(256,128)
            self.CCA2 = CrossCoAttention(128,64)
            self.CCA1 = CrossCoAttention(64,32)
            self.CCA0 = CrossCoAttention(32,32)
        elif self.mgmfa:
            self.up3 = up2(256,384,out_ch=192,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(192,192,out_ch=96,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(96,96,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(64,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.MGMFA3 = MGMFA(256,128)
            self.MGMFA2 = MGMFA(128,64)
            self.MGMFA1 = MGMFA(64,32)
            self.MGMFA0 = MGMFA(32,32)
        else : 
            self.up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)

        # self.up3 = up2(128,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up2 = up2(128,128,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up1 = up2(128,64,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        # self.up0 = up2(64,48,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
        if self.newup2 and self.cat:
            self.movable_up3 = newup2(256,256,out_ch=128,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up2 = newup2(128,128,out_ch=64,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up1 = newup2(64,64,out_ch=32,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up0 = newup2(32,32,out_ch=32,use_dropblock=dropblock,drop_p=dropout)
        else:
            self.movable_up3 = up2(256,256,out_ch=128,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up2 = up2(128,128,out_ch=64,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up1 = up2(64,64,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)
            self.movable_up0 = up2(32,32,out_ch=32,circular_padding=circular_padding,group_conv=group_conv,use_dropblock=dropblock,drop_p=dropout)

        self.moving_dropout = nn.Dropout(p=dropout)
        if self.cat or self.mgmfa:
            self.moving_outc = outconv(64, moving_n_class)
        else: self.moving_outc = outconv(32, moving_n_class)

        self.movable_dropout = nn.Dropout(p=dropout)
        self.movable_outc = outconv(32,movable_n_class)# 2*n_height(32)
        if self.cca_out:
            self.CCA_OUT = CrossCoAttention(64,64)
            
    def forward(self, x, res, r2p_matrix, p2r_matrix, range_res):
        self.r2p_matrix = r2p_matrix
        # x : B 32 480 360
        # res : B 8 480 360
        c0_appearance = self.inc(x)                # B 32 480 360 -> B 32 480 360
        c1_appearance = self.down1(c0_appearance)  # B 32 480 360 -> B 64 240 180
        c2_appearance = self.down2(c1_appearance)  # B 64 240 180 -> B 128 120 90
        c3_appearance = self.down3(c2_appearance)  # B 128 120 90 -> B 256 60 45
        c4_appearance = self.down4(c3_appearance)  # B 256 60 45  -> B 256 30 22

        c0_motion_range = self.res_inc_range(range_res)
        c1_motion_range = self.res_down1_range(c0_motion_range)
        c2_motion_range = self.res_down2_range(c1_motion_range)
        c3_motion_range = self.res_down3_range(c2_motion_range)
        c4_motion_range = self.res_down4_range(c3_motion_range)
        
        c0_motion = self.res_inc(res) # B 8 480 360 -> B 16 480 360
        c0_motion , _ =  self.CGM0(c0_motion,c0_appearance)# B 16 480 360 | B 32 480 360 -> B 16 480 360 | B 32 480 360  

        c1_motion = self.res_down1(c0_motion) # B 16 480 360 -> B 32 240 180
        c1_motion = self.flow_l2_r2p(self.r2p_matrix, c1_motion, c1_motion_range)
        c1_motion , _ =  self.CGM1(c1_motion,c1_appearance)# B 32 240 180 | B 64 240 180 -> B 32 240 180 | B 64 240 180 

        c2_motion = self.res_down2(c1_motion) # B 32 240 180 -> B 64 120 90
        c2_motion = self.flow_l3_r2p(self.r2p_matrix, c2_motion, c2_motion_range)
        c2_motion , _ =  self.CGM2(c2_motion,c2_appearance)# B 64 120 90 | B 128 120 90 -> B 64 120 90 | B 128 120 90 

        c3_motion = self.res_down3(c2_motion) # B 64 120 90 -> B 128 60 45
        c3_motion = self.flow_l4_r2p(self.r2p_matrix, c3_motion, c3_motion_range)
        c3_motion , _ =  self.CGM3(c3_motion,c3_appearance)# B 128 60 45 |  B 256 60 45 -> B 128 60 45 |  B 256 60 45 

        c4_motion = self.res_down4(c3_motion) # B 128 60 45 -> B 128 30 22 
        c4_motion = self.flow_l5_r2p(self.r2p_matrix, c4_motion, c4_motion_range)
        c4_motion , _ =  self.CGM4(c4_motion,c4_appearance)#  B 128 30 22 | B 256 30 22  ->  B 128 30 22  B 256 30 22 

        movable_out3 = self.movable_up3(c4_appearance,c3_appearance) # B 256 30 22 | B 256 60 45 -> B 128 60 45
        movable_out2 = self.movable_up2(movable_out3,c2_appearance) # B 128 60 45 | B 128 120 90 -> B 64 120 90
        movable_out1 = self.movable_up1(movable_out2,c1_appearance) # B 64 120 90 | B 64 240 180 -> B 32 240 180
        movable_out0 = self.movable_up0(movable_out1,c0_appearance) # B 32 240 180 | B 32 480 360 -> B 32 480 360
        movable_out = self.movable_outc(self.movable_dropout(movable_out0))

        if self.cat:
            c3_motion = torch.cat([c3_motion,movable_out3],dim=1)
        elif self.cat_block:
            c3_motion = self.CCA3(c3_motion,movable_out3)
        elif self.mgmfa:
            c3_motion = self.MGMFA3(c3_motion,movable_out3)
        else :
            c3_motion = c3_motion
        moving_out3 = self.up3(c4_motion, c3_motion)
        # moving_out3 = self.vss3(moving_out3)

        if self.cat:
            c2_motion = torch.cat([c2_motion,movable_out2],dim=1)
        elif self.cat_block:
            c2_motion = self.CCA2(c2_motion,movable_out2)
        elif self.mgmfa:
            c2_motion = self.MGMFA2(c2_motion,movable_out2)
        else:
            c2_motion = c2_motion
        moving_out2 = self.up2(moving_out3, c2_motion)
        # moving_out2 = self.vss2(moving_out2)
        
        if self.cat:
            c1_motion = torch.cat([c1_motion,movable_out1],dim=1)
        elif self.cat_block:
            c1_motion = self.CCA1(c1_motion,movable_out1)
        elif self.mgmfa:
            c1_motion = self.MGMFA1(c1_motion,movable_out1)
        else :
            c1_motion = c1_motion
        moving_out1 = self.up1(moving_out2, c1_motion)
        # moving_out1 = self.vss1(moving_out1)

        if self.cat:
            c0_motion = torch.cat([c0_motion,movable_out0],dim=1)
        elif self.cat_block:
            c0_motion = self.CCA0(c0_motion,movable_out0)  
        elif self.mgmfa:
            c0_motion = self.MGMFA0(c0_motion,movable_out0)
        else:
            c0_motion = c0_motion      
        moving_out0 = self.up0(moving_out1, c0_motion)


        moving_out = self.moving_outc(self.moving_dropout(moving_out0))
        
        if self.cca_out:
            moving_out = self.CCA_OUT(moving_out,moving_out)
        return moving_out,movable_out


class up2(nn.Module):
    def __init__(self, x1_in_ch,x2_in_ch, out_ch, circular_padding,group_conv=False, use_dropblock=False,
                 drop_p=0.5):
        super(up2, self).__init__()

        self.up = nn.Sequential(
                nn.Conv2d(x1_in_ch,x1_in_ch*4,kernel_size=(1,1)),
                nn.PixelShuffle(2)                
        )
        
        if circular_padding:
            self.conv = double_conv_circular(x1_in_ch+x2_in_ch, out_ch, group_conv=group_conv)


        self.use_dropblock = use_dropblock
        if self.use_dropblock:
            self.dropblock = DropBlock2D(block_size=7, drop_prob=drop_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is BCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.use_dropblock:
            x = self.dropblock(x)
        return x

class R2B_flow_new3(nn.Module):

    def __init__(self, fea_dim):
        super(R2B_flow_new3, self).__init__()
        self.fea_dim = fea_dim

        self.fusion = nn.Sequential(
            nn.Conv2d(fea_dim * 2, fea_dim, kernel_size=3, padding=(1, 0), bias=False),
            nn.BatchNorm2d(fea_dim),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=(1, 0), bias=False),
            nn.BatchNorm2d(fea_dim),
            nn.Softmax(dim=1)
        )
        # self.conv1x1x1 = torch.nn.Conv3d(in_channels=fea_dim, out_channels=fea_dim // 32, kernel_size=(1, 1, 1))
        self.conv1x1x1 = nn.Sequential(
            torch.nn.Conv3d(in_channels=fea_dim, out_channels=fea_dim // 32, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(fea_dim // 32),
            nn.ReLU(inplace=True)
        )

    def forward(self, flow_matrix, polar_fea, range_fea):
        """
        range_fea: [N, C1, 64, 2048]
        polar_fea: [N, C2, 480, 360]
        flow_matrix: [N, 480, 360, 32, 2]
        """
        range_fea_5d = range_fea.unsqueeze(2)   # (N,C1,1,H,W)   1，H，W分别代表深度，高度和宽度

        _, _, H, W = polar_fea.shape
        N, C, _, _ = range_fea.shape

        flow_matrix = torch.flip(flow_matrix, dims=[-1])
        N, h, w, K, c = flow_matrix.shape
        flow_matrix = flow_matrix.view(N, h, w, K * c).permute(0, 3, 1, 2).contiguous()
        flow_matrix_scaled = F.interpolate(flow_matrix.float(), (H, W), mode='nearest')
        flow_matrix_scaled = flow_matrix_scaled.permute(0, 2, 3, 1).view(N, H, W, K, c)
        flow_matrix_scaled = F.pad(flow_matrix_scaled, pad=(0, 1), mode='constant', value=0.0) #(N, H, W, K, c+1)
        flow_fea = F.grid_sample(range_fea_5d, flow_matrix_scaled, padding_mode='zeros', align_corners=False) # N*C*H*W*K

        # 对张量进行3D卷积操作
        flow_fea = self.conv1x1x1(flow_fea)  # [N,c/K,H,W,K]

        # 转换为[N, K, H, W]
        #flow_fea = flow_fea.squeeze(dim=1).permute(0, 3, 1, 2).contiguous()
        flow_fea = rearrange(flow_fea, 'b d h w c -> b (d c) h w')
        #flow_fea = torch.max(flow_fea, dim=-1)[0] # N*C*H*W

        # fea = torch.cat((polar_fea, flow_fea), dim=1)
        try:
            fea = torch.cat((polar_fea, flow_fea), dim=1)
        except Exception as e:
            print(f"Error during torch.cat operation: {e}")
            print(f"polar_fea shape: {polar_fea.shape}")
            print(f"flow_fea shape: {flow_fea.shape}")
            
        fea = F.pad(fea, (1, 1, 0, 0), mode='circular')
        res = self.fusion(fea)
        res = res * self.attention(F.pad(res, (1, 1, 0, 0), mode='circular'))
        fea  = polar_fea + res

        return fea      
      
class R2B_flow_new3_nres(nn.Module):

    def __init__(self, fea_dim):
        super(R2B_flow_new3_nres, self).__init__()
        self.fea_dim = fea_dim

        self.fusion = nn.Sequential(
            nn.Conv2d(fea_dim * 2, fea_dim, kernel_size=3, padding=(1, 0), bias=False),
            nn.BatchNorm2d(fea_dim),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=(1, 0), bias=False),
            nn.BatchNorm2d(fea_dim),
            nn.Softmax(dim=1)
        )
        # self.conv1x1x1 = torch.nn.Conv3d(in_channels=fea_dim, out_channels=fea_dim // 32, kernel_size=(1, 1, 1))
        self.conv1x1x1 = nn.Sequential(
            torch.nn.Conv3d(in_channels=fea_dim, out_channels=fea_dim // 32, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(fea_dim // 32),
            nn.ReLU(inplace=True)
        )

    def forward(self, flow_matrix, polar_fea, range_fea):
        """
        range_fea: [N, C1, 64, 2048]
        polar_fea: [N, C2, 480, 360]
        flow_matrix: [N, 480, 360, 32, 2]
        """
        range_fea_5d = range_fea.unsqueeze(2)   # (N,C1,1,H,W)   1，H，W分别代表深度，高度和宽度

        _, _, H, W = polar_fea.shape
        N, C, _, _ = range_fea.shape

        flow_matrix = torch.flip(flow_matrix, dims=[-1])
        N, h, w, K, c = flow_matrix.shape
        flow_matrix = flow_matrix.view(N, h, w, K * c).permute(0, 3, 1, 2).contiguous()
        flow_matrix_scaled = F.interpolate(flow_matrix.float(), (H, W), mode='nearest')
        flow_matrix_scaled = flow_matrix_scaled.permute(0, 2, 3, 1).view(N, H, W, K, c)
        flow_matrix_scaled = F.pad(flow_matrix_scaled, pad=(0, 1), mode='constant', value=0.0) #(N, H, W, K, c+1)
        flow_fea = F.grid_sample(range_fea_5d, flow_matrix_scaled, padding_mode='zeros', align_corners=False) # N*C*H*W*K

        # 对张量进行3D卷积操作
        flow_fea = self.conv1x1x1(flow_fea)  # [N,c/K,H,W,K]

        # 转换为[N, K, H, W]
        #flow_fea = flow_fea.squeeze(dim=1).permute(0, 3, 1, 2).contiguous()
        flow_fea = rearrange(flow_fea, 'b d h w c -> b (d c) h w')
        #flow_fea = torch.max(flow_fea, dim=-1)[0] # N*C*H*W

        # fea = torch.cat((polar_fea, flow_fea), dim=1)
        try:
            fea = torch.cat((polar_fea, flow_fea), dim=1)
        except Exception as e:
            print(f"Error during torch.cat operation: {e}")
            print(f"polar_fea shape: {polar_fea.shape}")
            print(f"flow_fea shape: {flow_fea.shape}")
            
        fea = F.pad(fea, (1, 1, 0, 0), mode='circular')
        res = self.fusion(fea)
        res = res * self.attention(F.pad(res, (1, 1, 0, 0), mode='circular'))
        #fea  = polar_fea + res

        return res   

class R2B_flow_new3_VSS(nn.Module):

    def __init__(self, fea_dim):
        super(R2B_flow_new3_VSS, self).__init__()
        self.fea_dim = fea_dim

        # self.fusion = nn.Sequential(
        #     nn.Conv2d(fea_dim * 2, fea_dim, kernel_size=3, padding=(1, 0), bias=False),
        #     nn.BatchNorm2d(fea_dim),
        #     nn.ReLU(inplace=True)
        # )
        # self.attention = nn.Sequential(
        #     nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=(1, 0), bias=False),
        #     nn.BatchNorm2d(fea_dim),
        #     nn.Softmax(dim=1)
        # )
        # self.conv1x1x1 = torch.nn.Conv3d(in_channels=fea_dim, out_channels=fea_dim // 32, kernel_size=(1, 1, 1))
        self.conv1x1x1 = nn.Sequential(
            torch.nn.Conv3d(in_channels=fea_dim, out_channels=fea_dim // 32, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(fea_dim // 32),
            nn.ReLU(inplace=True)
        )
        # self.vss = C2f_LVMB(fea_dim*2,fea_dim,2)
        self.vss = C2f_VSS(fea_dim*2,fea_dim,2)
        #self.vss = VSSBlock(fea_dim*2)

    def forward(self, flow_matrix, polar_fea, range_fea):
        """
        range_fea: [N, C1, 64, 2048]
        polar_fea: [N, C2, 480, 360]
        flow_matrix: [N, 480, 360, 32, 2]
        """
        range_fea_5d = range_fea.unsqueeze(2)   # (N,C1,1,H,W)   1，H，W分别代表深度，高度和宽度

        _, _, H, W = polar_fea.shape
        N, C, _, _ = range_fea.shape

        flow_matrix = torch.flip(flow_matrix, dims=[-1])
        N, h, w, K, c = flow_matrix.shape
        flow_matrix = flow_matrix.view(N, h, w, K * c).permute(0, 3, 1, 2).contiguous()
        flow_matrix_scaled = F.interpolate(flow_matrix.float(), (H, W), mode='nearest')
        flow_matrix_scaled = flow_matrix_scaled.permute(0, 2, 3, 1).view(N, H, W, K, c)
        flow_matrix_scaled = F.pad(flow_matrix_scaled, pad=(0, 1), mode='constant', value=0.0) #(N, H, W, K, c+1)
        flow_fea = F.grid_sample(range_fea_5d, flow_matrix_scaled, padding_mode='zeros', align_corners=False) # N*C*H*W*K

        # 对张量进行3D卷积操作
        flow_fea = self.conv1x1x1(flow_fea)  # [N,c/K,H,W,K]

        # 转换为[N, K, H, W]
        #flow_fea = flow_fea.squeeze(dim=1).permute(0, 3, 1, 2).contiguous()
        flow_fea = rearrange(flow_fea, 'b d h w c -> b (d c) h w')
        #flow_fea = torch.max(flow_fea, dim=-1)[0] # N*C*H*W

        # fea = torch.cat((polar_fea, flow_fea), dim=1)
        try:
            fea = torch.cat((polar_fea, flow_fea), dim=1)
        except Exception as e:
            print(f"Error during torch.cat operation: {e}")
            print(f"polar_fea shape: {polar_fea.shape}")
            print(f"flow_fea shape: {flow_fea.shape}")
        
        fea = self.vss(fea)

        # fea = F.pad(fea, (1, 1, 0, 0), mode='circular')
        # res = self.fusion(fea)
        # res = res * self.attention(F.pad(res, (1, 1, 0, 0), mode='circular'))
        fea  = polar_fea + fea

        return fea

class newup2(nn.Module):
    def __init__(self, x1_in_ch,x2_in_ch,out_ch,use_dropblock,drop_p):
        super(newup2, self).__init__()
        print("using newup2")
        self.up = nn.Sequential(
                nn.Conv2d(x1_in_ch,x1_in_ch*4,kernel_size=(1,1)),
                nn.PixelShuffle(2)                
        )
        self.conv  = nn.Sequential(# BasicBlock((x1_in_ch+x2_in_ch),(x1_in_ch+x2_in_ch),stride=1,relu=False),
                                    BasicBlock((x1_in_ch+x2_in_ch),out_ch,stride=1,relu=True),)
        self.use_dropblock = use_dropblock
        if self.use_dropblock:
            self.dropblock = DropBlock2D(block_size=7, drop_prob=drop_p)
    def forward(self, x1, x2):  
        x1 = self.up(x1)
        # input is BCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.use_dropblock:
            x = self.dropblock(x)
        return x
    
class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1,relu=False):
        super(BasicBlock, self).__init__()
        self.relu = relu 
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
                )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        if self.relu:
            out = self.relu(out)
        return out

class newdown(nn.Module):
    def __init__(self,in_chan,out_chan,stride):
        super(newdown,self).__init__()
        self.down  = nn.Sequential(BasicBlock(in_chan,in_chan*2,stride=1,relu=False),
                                    BasicBlock(in_chan*2,out_chan,stride=1,relu=False),
                                    BasicBlock(out_chan,out_chan,stride=stride,relu=True),)

    def forward(self,x):
        return self.down(x)
    
class newup(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(newup, self).__init__()

        self.up = nn.Sequential(
                    nn.Conv2d(in_chan,in_chan*4,kernel_size=(1,1)),
                    nn.PixelShuffle(2)                
        )

        self.conv  = nn.Sequential(BasicBlock(in_chan*2,in_chan,stride=1,relu=False),
                                    BasicBlock(in_chan,out_chan,stride=1,relu=False),
                                    BasicBlock(out_chan,out_chan,stride=1,relu=True),)

    def forward(self, x1, x2):  
        x1 = self.up(x1)
        # input is BCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x








class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        x_channel = x.mul(out)  # out * x
        return x_channel


class SpatialAttention(nn.Module):
    def __init__(self, circular_padding):
        super(SpatialAttention, self).__init__()
        self.circular_padding = circular_padding
        if circular_padding:
            padding = (1, 0)
        else:
            padding = 1

        self.conv1 = nn.Conv2d(2, 1, 3, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        if self.circular_padding:
            out = F.pad(out, (1, 1, 0, 0), mode='circular')
        out = self.conv1(out)
        x_spatial = self.sigmoid(out)
        return x_spatial


class attention_module_MGA_tmc(nn.Module):
    def __init__(self, channel_a, channel_m):
        super(attention_module_MGA_tmc, self).__init__()
        self.conv1x1_channel_wise = nn.Conv2d(channel_a, channel_a, 1, bias=True)
        self.conv1x1_spatial = nn.Conv2d(channel_m, 1, 1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, img_feat, flow_feat):
        """
            flow_feat_map:  [bsize, 1, h, w]
            feat_vec:       [bsize, channel, 1, 1]
            channel_attentioned_img_feat:  [bsize, channel, h, w]
        """
        # spatial attention
        flow_feat_map = self.conv1x1_spatial(flow_feat)
        flow_feat_map = nn.Sigmoid()(flow_feat_map)
        spatial_attentioned_img_feat = flow_feat_map * img_feat

        # channel-wise attention
        feat_vec = self.avg_pool(spatial_attentioned_img_feat)
        feat_vec = self.conv1x1_channel_wise(feat_vec)
        feat_vec = nn.Softmax(dim=1)(feat_vec) * feat_vec.shape[1]
        channel_attentioned_img_feat = spatial_attentioned_img_feat * feat_vec

        final_feat = channel_attentioned_img_feat + img_feat
        return final_feat

class CAG_VSS(nn.Module):  # Co-Attention Gate Module
    def __init__(self, channel_a, channel_m, circular_padding=False, group_conv=False,n=2):
        super(CAG_VSS, self).__init__()
        self.circular_padding = circular_padding
        self.channel_a = channel_a
        self.channel_m = channel_m
        self.vss = Block_VSS(3,channel_a + channel_m, channel_a + channel_m, channel_a, hswish(),VSSBlock(channel_a),1,n)

        if circular_padding:
            self.fuse_feature1 = single_conv_circular(channel_a + channel_m, 32, group_conv)
            self.fuse_feature2 = nn.Conv2d(32, 2, kernel_size=3, padding=(1, 0))
        else:
            self.fuse_feature1 = single_conv(channel_a + channel_m, 32, group_conv)
            self.fuse_feature2 = nn.Conv2d(32, 2, kernel_size=3, padding=1)
        self.motion_channel_attention = ChannelAttention(channel_m)
        self.motion_spatial_attention = SpatialAttention(circular_padding)
        self.attention_module = attention_module_MGA_tmc(channel_a, channel_m)

    def forward(self, f_a, f_m):
        g_a = self.vss(torch.cat((f_a, f_m), dim=1))
        # g_am = self.fuse_feature1(torch.cat((f_a, f_m), dim=1))
        # if self.circular_padding:
        #     g_am = F.pad(g_am, (1, 1, 0, 0), mode='circular')
        # g_am = self.fuse_feature2(g_am)
        # g_am = F.adaptive_avg_pool2d(torch.sigmoid(g_am), 1)
        # g_a = g_am[:, 0, :, :].unsqueeze(1).repeat(1, self.channel_a, 1, 1) * f_a
        # g_m = g_am[:, 1, :, :].unsqueeze(1).repeat(1, self.channel_m, 1, 1) * f_m

        e_am = self.attention_module(g_a, f_m)

        return e_am, f_m

class CAG(nn.Module):  # Co-Attention Gate Module
    def __init__(self, channel_a, channel_m, circular_padding=False, group_conv=False):
        super(CAG, self).__init__()
        self.circular_padding = circular_padding
        self.channel_a = channel_a
        self.channel_m = channel_m

        if circular_padding:
            self.fuse_feature1 = single_conv_circular(channel_a + channel_m, 32, group_conv)
            self.fuse_feature2 = nn.Conv2d(32, 2, kernel_size=3, padding=(1, 0))
        else:
            self.fuse_feature1 = single_conv(channel_a + channel_m, 32, group_conv)
            self.fuse_feature2 = nn.Conv2d(32, 2, kernel_size=3, padding=1)
        self.motion_channel_attention = ChannelAttention(channel_m)
        self.motion_spatial_attention = SpatialAttention(circular_padding)
        self.attention_module = attention_module_MGA_tmc(channel_a, channel_m)

    def forward(self, f_a, f_m):
        g_am = self.fuse_feature1(torch.cat((f_a, f_m), dim=1))
        if self.circular_padding:
            g_am = F.pad(g_am, (1, 1, 0, 0), mode='circular')
        g_am = self.fuse_feature2(g_am)
        g_am = F.adaptive_avg_pool2d(torch.sigmoid(g_am), 1)
        g_a = g_am[:, 0, :, :].unsqueeze(1).repeat(1, self.channel_a, 1, 1) * f_a
        g_m = g_am[:, 1, :, :].unsqueeze(1).repeat(1, self.channel_m, 1, 1) * f_m

        e_am = self.attention_module(g_a, g_m)

        return e_am, g_m
    

# Global-local Attention Context (GAC)
class GAC(nn.Module):
    def __init__(self, input_channels, eps=1e-5):
        super(GAC, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, input_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, input_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, input_channels, 1, 1))
        self.epsilon = eps

    def forward(self, x):
        Nl = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon)
        glo = Nl.pow(0.5) * self.alpha
        Nc = (glo.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        cal = self.gamma / Nc

        v_fea = x * 1. + x * torch.tanh(glo * cal + self.beta)
        return v_fea
    
# Global-local Attention Fusion (GAF)
class GAF(nn.Module):
    def __init__(self, channels=64, r=4):
        super(GAF, self).__init__()
        # r = channels // 16
        inter_channels = int(channels // r)

        # local attention
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # global attention
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # local attention
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # global attention
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        xa = x1 + x2
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x1 * wei + x2 * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x1 * wei2 + x2 * (1 - wei2)
        return xo

# Cross Co-Attention Module (CCAM)
class CrossCoAttention(nn.Module):
    def __init__(self, f1_channel,f2_channel):
        super(CrossCoAttention, self).__init__()

        self.gct = GAC(f1_channel)
        self.faf = GAF(f1_channel)
        if f1_channel != f2_channel:
            self.conv = nn.Conv2d(f2_channel,f1_channel,1,1)

    def forward(self, f1, f2):
        a1 = self.gct(f1)
        c1 = f1.shape[1]
        c2 = f2.shape[1]
        if c1!= c2:
            f2=self.conv(f2)
        a2 = self.gct(f2)
        aff = self.faf(a1, a2)
        return aff
    

class MGMFA(nn.Module):
    # Movable Guided MultiPath Fused Attention
    def __init__(self, motion_ch , movable_ch) -> None:
        super().__init__()
        total_ch = motion_ch + movable_ch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(total_ch,total_ch,1,1),
            nn.BatchNorm2d(total_ch),
            nn.ReLU()
        )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_hw = nn.Sequential(
                    nn.Conv2d(total_ch,total_ch,(3,1),1,(1,0)),
                    nn.BatchNorm2d(total_ch),
                    nn.ReLU()    
        )
        self.conv_pool_hw = nn.Sequential(
                    nn.Conv2d(total_ch,total_ch,1,1),
                    nn.BatchNorm2d(total_ch),
                    nn.ReLU()    
        )
    
    def forward(self, motion,movable):
        fused = torch.cat([motion,movable],dim=1)
        _, _, h, w = fused.size()
        fused_pool_h, fused_pool_w, fused_pool_ch = self.pool_h(fused), self.pool_w(fused).permute(0, 1, 3, 2), self.gap(fused)
        fused_pool_hw = torch.cat([fused_pool_h, fused_pool_w], dim=2)
        fused_pool_hw = self.conv_hw(fused_pool_hw)
        fused_pool_h, fused_pool_w = torch.split(fused_pool_hw, [h, w], dim=2)
        fused_pool_hw_weight = self.conv_pool_hw(fused_pool_hw).sigmoid()
        fused_pool_h_weight, fused_pool_w_weight = torch.split(fused_pool_hw_weight, [h, w], dim=2)
        fused_pool_h, fused_pool_w = fused_pool_h * fused_pool_h_weight, fused_pool_w * fused_pool_w_weight
        fused_pool_ch = fused_pool_ch * torch.mean(fused_pool_hw_weight, dim=2, keepdim=True)
        return fused * fused_pool_h.sigmoid() * fused_pool_w.permute(0, 1, 3, 2).sigmoid() * fused_pool_ch.sigmoid()    
    
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class MPCA(nn.Module):
    # MultiPath Coordinate Attention
    def __init__(self, channels) -> None:
        super().__init__()
        
        # self.gap = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     Conv(channels, channels)
        # )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_hw = Conv(channels, channels, (3, 1))
        self.conv_pool_hw = Conv(channels, channels, 1)
    
    def forward(self, x):
        _, _, h, w = x.size()
        x_pool_h = self.pool_h(x)
        x_pool_w = self.pool_w(x).permute(0, 1, 3, 2)
        #x_pool_ch = self.gap(x)
        x_pool_hw = torch.cat([x_pool_h, x_pool_w], dim=2)
        x_pool_hw = self.conv_hw(x_pool_hw)
        x_pool_h, x_pool_w = torch.split(x_pool_hw, [h, w], dim=2)
        x_pool_hw_weight = self.conv_pool_hw(x_pool_hw).sigmoid()
        x_pool_h_weight, x_pool_w_weight = torch.split(x_pool_hw_weight, [h, w], dim=2)
        x_pool_h, x_pool_w = x_pool_h * x_pool_h_weight, x_pool_w * x_pool_w_weight
        #x_pool_ch = x_pool_ch * torch.mean(x_pool_hw_weight, dim=2, keepdim=True)
        return x * x_pool_h.sigmoid() * x_pool_w.permute(0, 1, 3, 2).sigmoid() #* x_pool_ch.sigmoid()

class R2B_flow_new3_MPCA2d(nn.Module):

    def __init__(self, fea_dim):
        super(R2B_flow_new3_MPCA2d, self).__init__()
        self.fea_dim = fea_dim

        self.fusion = nn.Sequential(
            nn.Conv2d(fea_dim * 2, fea_dim, kernel_size=3, padding=(1, 0), bias=False),
            nn.BatchNorm2d(fea_dim),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=(1, 0), bias=False),
            nn.BatchNorm2d(fea_dim),
            nn.Softmax(dim=1)
        )
        # self.conv1x1x1 = torch.nn.Conv3d(in_channels=fea_dim, out_channels=fea_dim // 32, kernel_size=(1, 1, 1))
        self.mpca=MPCA(fea_dim)
        self.conv1x1x1 = nn.Sequential(
            torch.nn.Conv3d(in_channels=fea_dim, out_channels=fea_dim // 32, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(fea_dim // 32),
            nn.ReLU(inplace=True)
        )

    def forward(self, flow_matrix, polar_fea, range_fea):
        """
        range_fea: [N, C1, 64, 2048]
        polar_fea: [N, C2, 480, 360]
        flow_matrix: [N, 480, 360, 32, 2]
        """
        range_fea_5d = range_fea.unsqueeze(2)   # (N,C1,1,H,W)   1，H，W分别代表深度，高度和宽度

        _, _, H, W = polar_fea.shape
        N, C, _, _ = range_fea.shape

        flow_matrix = torch.flip(flow_matrix, dims=[-1])
        N, h, w, K, c = flow_matrix.shape
        flow_matrix = flow_matrix.view(N, h, w, K * c).permute(0, 3, 1, 2).contiguous()
        flow_matrix_scaled = F.interpolate(flow_matrix.float(), (H, W), mode='nearest')
        flow_matrix_scaled = flow_matrix_scaled.permute(0, 2, 3, 1).view(N, H, W, K, c)
        flow_matrix_scaled = F.pad(flow_matrix_scaled, pad=(0, 1), mode='constant', value=0.0) #(N, H, W, K, c+1)
        flow_fea = F.grid_sample(range_fea_5d, flow_matrix_scaled, padding_mode='zeros', align_corners=False) # N*C*H*W*K
        # 对张量进行3D卷积操作
        flow_fea = self.conv1x1x1(flow_fea)  # [N,c/K,H,W,K]

        # 转换为[N, K, H, W]
        #flow_fea = flow_fea.squeeze(dim=1).permute(0, 3, 1, 2).contiguous()
        flow_fea = rearrange(flow_fea, 'b d h w c -> b (d c) h w')
        #flow_fea = torch.max(flow_fea, dim=-1)[0] # N*C*H*W

        # flow_fea = self.mpca(flow_fea)
        # fea = torch.cat((polar_fea, flow_fea), dim=1)
        try:
            fea = torch.cat((polar_fea, flow_fea), dim=1)
        except Exception as e:
            print(f"Error during torch.cat operation: {e}")
            print(f"polar_fea shape: {polar_fea.shape}")
            print(f"flow_fea shape: {flow_fea.shape}")
            
        fea = F.pad(fea, (1, 1, 0, 0), mode='circular')
        res = self.fusion(fea)
        res = self.mpca(res)
        res = res * self.attention(F.pad(res, (1, 1, 0, 0), mode='circular'))
        fea  = polar_fea + res

        return fea
    
class catblock(nn.Module):

    def __init__(self, fea_dim,fea_dim2):
        super(catblock, self).__init__()
        self.fea_dim = fea_dim

        self.fusion = nn.Sequential(
            nn.Conv2d(fea_dim + fea_dim2, fea_dim, kernel_size=3, padding=(1, 0), bias=False),
            nn.BatchNorm2d(fea_dim),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=(1, 0), bias=False),
            nn.BatchNorm2d(fea_dim),
            nn.Softmax(dim=1)
        )
        # self.conv1x1x1 = torch.nn.Conv3d(in_channels=fea_dim, out_channels=fea_dim // 32, kernel_size=(1, 1, 1))
        #self.mpca=MPCA(fea_dim)
        self.conv1x1x1 = nn.Sequential(
            torch.nn.Conv3d(in_channels=fea_dim, out_channels=fea_dim // 32, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(fea_dim // 32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):

        fea = torch.cat((x1, x2), dim=1)

        fea = F.pad(fea, (1, 1, 0, 0), mode='circular')
        res = self.fusion(fea)
        #res = self.mpca(res)
        res = res * self.attention(F.pad(res, (1, 1, 0, 0), mode='circular'))
        fea  = x1 + res

        return fea