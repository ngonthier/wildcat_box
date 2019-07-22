#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ClusterRun.py
#  
#  Copyright 2019 gonthier <gonthier@Morisot>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

from wildcat.demo_voc2007_new_pooling import get_parser,train_or_test_VOC07
                
def main():
    k = 20
    for mode in ['']:
        for k in [20,0.13]: # 13 % of the region take for max and min
            for classif in [False,True]:
                # Training or Testing
                parser = get_parser()
                parser.set_defaults(data='../data/voc',image_size=448,batch_size=16,lrp=0.1,lr=0.01,\
                    epochs=20,k=k,maps=4,alpha=0.7,\
                    save_init_model=True,test=False,classif=classif,mode=mode) # pas de test de detection
                args = parser.parse_args()
                train_or_test_VOC07(args)

def mainDirect():
    k = 20
    for mode in ['LCPPReLU','LCPRReLU']:
        for k in [20,0.13]: # 13 % of the region take for max and min
            for classif in [False,True]:
                # Training or Testing
                parser = get_parser()
                parser.set_defaults(data='../data/voc',image_size=448,batch_size=16,lrp=0.1,lr=0.01,\
                    epochs=20,k=k,maps=4,alpha=0.7,\
                    save_init_model=True,test=False,classif=classif,mode=mode) # pas de test de detection
                args = parser.parse_args()
                train_or_test_VOC07(args)
                
def mainAll():
    k = 20
    for init in ['xavier_uniform','kaiming_uniform','orthogonal']: # deja fait 'uniform_div_std_maps'
        for mode in ['LCP','Direct','LCPPReLU']:
            for kernel_size in [1,3]:
                if not mode in ['','Direct']:
                    kernel_size_lcp_tab = [1,3]
                else:
                    kernel_size_lcp_tab = [1]
                for kernel_size_lcp in kernel_size_lcp_tab:
                    for classif in [False,True]:
                        # Training or Testing
                        parser = get_parser()
                        parser.set_defaults(data='../data/voc',image_size=448,batch_size=16,lrp=0.1,lr=0.01,\
                            epochs=40,k=k,maps=4,alpha=0.7,\
                            save_init_model=False,test=False,classif=classif,mode=mode
                            ,kernel_size_lcp=kernel_size_lcp,kernel_size=kernel_size,init=init) # pas de test de detection
                        args = parser.parse_args()
                        train_or_test_VOC07(args)
     
if __name__ == '__main__':
    mainAll()
