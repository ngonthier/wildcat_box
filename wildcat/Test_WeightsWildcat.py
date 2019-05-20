#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Goal of this script is to see if the kernel learn by Wildcat evole in 
# the same way during the optimization, i.e. if the gradient apply to the 
# different maps of a same class are the same (due to the average class-wise pooling)
#  
#  Copyright 2019 gonthier <gonthier@Morisot>  
#  

import torch
import numpy as np

from wildcat.models import resnet101_wildcat
from wildcat.models import resnet101_wildcat

def main(args):
    
    # /home/gonthier/Travail_Local/Icono_Art/TravauxProches/wildcat.pytorch/expes/models/IconArt_v1/model_im224_bs16_lrp0.1_lr0.01_ep20_k25_a0.7_m8_0.pth.tar
# /home/gonthier/Travail_Local/Icono_Art/TravauxProches/wildcat.pytorch/expes/models/IconArt_v1/model_im224_bs16_lrp0.1_lr0.01_ep20_k25_a0.7_m8_0_initModel.pth.tar
# /home/gonthier/Travail_Local/Icono_Art/TravauxProches/wildcat.pytorch/expes/models/IconArt_v1/model_im224_bs16_lrp0.1_lr0.01_ep20_k25_a0.7_m8_1.pth.tar
# /home/gonthier/Travail_Local/Icono_Art/TravauxProches/wildcat.pytorch/expes/models/IconArt_v1/model_im224_bs16_lrp0.1_lr0.01_ep20_k25_a0.7_m8_1_initModel.pth.tar
# /home/gonthier/Travail_Local/Icono_Art/TravauxProches/wildcat.pytorch/expes/models/IconArt_v1/model_im224_bs16_lrp0.1_lr0.01_ep20_k25_a0.7_m8_SameKernel_0.pth.tar
# /home/gonthier/Travail_Local/Icono_Art/TravauxProches/wildcat.pytorch/expes/models/IconArt_v1/model_im224_bs16_lrp0.1_lr0.01_ep20_k25_a0.7_m8_SameKernel_0_initModel.pth.tar
# /home/gonthier/Travail_Local/Icono_Art/TravauxProches/wildcat.pytorch/expes/models/IconArt_v1/model_im224_bs16_lrp0.1_lr0.01_ep20_k25_a0.7_m8_SameKernel_1.pth.tar
# /home/gonthier/Travail_Local/Icono_Art/TravauxProches/wildcat.pytorch/expes/models/IconArt_v1/model_im224_bs16_lrp0.1_lr0.01_ep20_k25_a0.7_m8_SameKernel_1_initModel.pth.tar
    
    list_model_compared_weights = ['/home/gonthier/Travail_Local/Icono_Art/TravauxProches/wildcat.pytorch/expes/models/IconArt_v1/model_im224_bs16_lrp0.1_lr0.01_ep20_k25_a0.7_m8_0.pth.tar',
    '/home/gonthier/Travail_Local/Icono_Art/TravauxProches/wildcat.pytorch/expes/models/IconArt_v1/model_im224_bs16_lrp0.1_lr0.01_ep20_k25_a0.7_m8_1.pth.tar',
    '/home/gonthier/Travail_Local/Icono_Art/TravauxProches/wildcat.pytorch/expes/models/IconArt_v1/model_im224_bs16_lrp0.1_lr0.01_ep20_k25_a0.7_m8_SameKernel_0.pth.tar',
    '/home/gonthier/Travail_Local/Icono_Art/TravauxProches/wildcat.pytorch/expes/models/IconArt_v1/model_im224_bs16_lrp0.1_lr0.01_ep20_k25_a0.7_m8_SameKernel_1.pth.tar',
    '/home/gonthier/Travail_Local/Icono_Art/TravauxProches/wildcat.pytorch/expes/models/IconArt_v1/model_im600_bs8_lrp0.1_lr0.01_ep20_k25_a0.7_m8_SameKernel_1.pth.tar',
    '/home/gonthier/Travail_Local/Icono_Art/TravauxProches/wildcat.pytorch/expes/models/IconArt_v1/model_im600_bs8_lrp0.1_lr0.01_ep20_k25_a0.7_m8_SameKernel_0.pth.tar',
    '/home/gonthier/Travail_Local/Icono_Art/TravauxProches/wildcat.pytorch/expes/models/IconArt_v1/model_im600_bs8_lrp0.1_lr0.01_ep20_k25_a0.7_m8_1.pth.tar',
    '/home/gonthier/Travail_Local/Icono_Art/TravauxProches/wildcat.pytorch/expes/models/IconArt_v1/model_im600_bs8_lrp0.1_lr0.01_ep20_k25_a0.7_m8_0.pth.tar']
    
    for model_name in list_model_compared_weights:
        
        model_last = model_name.split('/')[-1]
        model_tab = model_last.split('_')
        # print(model_tab)
        im = model_tab[1]
        image_size = int(im.replace('im',''))
        bs = model_tab[2]
        batch_size = int(bs.replace('bs',''))
        ep = model_tab[5]
        epochs = int(ep.replace('ep',''))
        kk = model_tab[6]
        k = int(kk.replace('k',''))
        aa = model_tab[7]
        alpha = float(aa.replace('a',''))
        mm = model_tab[8]
        maps = int(mm.replace('m',''))
        typeKernel = model_tab[9]
        if typeKernel=='SameKernel':
            num = int(model_tab[10].replace('.pth.tar',''))
        else:
            typeKernel = ''
            num = int(model_tab[9].split('.')[0])
            
        model_init =  model_name.split('.')[0] + '_initModel.pth.tar'
        dict_kernels ={}
        
        with_gt = False
        multiscale = False
        num_classes = 7
        kernel_size = 1
        
        for name,model_to_read in zip(['init','end'],[model_init,model_name]):
            evaluate = True
            # PATH =  'expes/models/IconArt_v1/'+model_name
            PATH = model_name
            state = {'batch_size': batch_size, 'image_size': image_size, 'max_epochs': epochs,
                     'evaluate': evaluate, 'resume': PATH}
            state['difficult_examples'] = True
            state['save_model_path'] = 'expes/models/IconArt_v1/'
            use_gpu = torch.cuda.is_available()
            state['use_gpu'] = use_gpu


            if typeKernel=='SameKernel':
                same_kernel=True
            else:
                same_kernel = False
            model = resnet101_wildcat(num_classes, pretrained=True, kmax=k,\
             alpha=alpha, num_maps=maps,kernel_size=kernel_size,same_kernel=same_kernel)
            
            model.train(False)
            state_dict_all = torch.load(PATH)
            best_epoch = state_dict_all["epoch"]
            best_score = state_dict_all["best_score"]
            state_dict = state_dict_all["state_dict"]
            model.load_state_dict(state_dict)
            
            
            last_kernel_conv2D = None
            for m in model.modules():
                # print(m)
                if isinstance(m, torch.nn.Conv2d):
                    # print('conv2D')
                    last_kernel_conv2D = m.weight.data.cpu().numpy()
            dict_kernels[name] = last_kernel_conv2D  
        
        kernel_conv_init = dict_kernels['init']
        kernel_conv_end = dict_kernels['end']
        
        print("For model imsize =",image_size,"restart =",num,"type kernel :",typeKernel)

        for c in range(num_classes):
            kernels_c_init = kernel_conv_init[c:(c+1)*maps,:,0,0] 
            kernels_c_end = kernel_conv_end[c:(c+1)*maps,:,0,0] 
            diff_c = kernels_c_end - kernels_c_init
            max_diff = 0.
            for ki in range(maps):
                for kj in range(maps):
                    if not(ki==kj):
                        # print('Indices :',ki,kj)
                        diff_i_j = diff_c[ki,:] - diff_c[kj,:]
                        max_diff_ki_kj = np.max(np.abs(diff_i_j))
                        if max_diff < max_diff_ki_kj:
                            max_diff = max_diff_ki_kj
                        # print('max abs(a-b)',np.max(np.abs(diff_i_j)))
                        # print('max abs(a-b)/abs(a)',np.max(np.abs(diff_i_j)/np.abs(diff_c[ki,:])))
            print('Class :',c,' max = {0:.5f}'.format(max_diff))
        

    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
