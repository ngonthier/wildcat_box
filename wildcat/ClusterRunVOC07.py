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
                
def mainDirect():
    k = 20
    for mode in ['Direct','LCP']:
        for lrm in [1.,10.]: # 13 % of the region take for max and min
            for classif in [False,True]:
                # Training or Testing
                parser = get_parser()
                parser.set_defaults(data='../data/voc',image_size=448,batch_size=16,lrp=lrm*0.1,lr=lrm*0.01,\
                    epochs=20,k=k,maps=4,alpha=0.7,\
                    save_init_model=True,test=False,classif=classif,mode=mode) # pas de test de detection
                args = parser.parse_args()
                train_or_test_VOC07(args)
     
if __name__ == '__main__':
    mainDirect()
