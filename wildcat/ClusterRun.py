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

from wildcat.demo_IconArt_v1 import get_parser,train_or_test_IconArt_v1

def main():
	for same_kernel in [False,True]:
		for i in range(2): # Number of repetition
			for test,classif in zip([False,True],[False,True]):
				# Training or Testing
				parser = get_parser()
				parser.set_defaults(data='../data/',i=600,b=8,lrp=0.1,lr=0.01,\
					epochs=20,k=25,maps=8,alpha=0.7,same_kernel=same_kernel,\
					save_init_model=True,ext=str(i),test=test,classif=classif)
				args = parser.parse_args()
				train_or_test_IconArt_v1(args)
    
    

if __name__ == '__main__':
    main()
