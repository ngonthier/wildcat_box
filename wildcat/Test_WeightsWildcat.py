#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Goal of this script is to see if the kernel learn by Wildcat evole in 
# the same way during the optimization, i.e. if the gradient apply to the 
# different maps of a same class are the same (due to the average class-wise pooling)
#  
#  Copyright 2019 gonthier <gonthier@Morisot>  
#  


def main(args):
	
	list_model_compared_weights = 
	
	PATH =  'expes/models/IconArt_v1/'+model_name
	state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
			 'evaluate': args.evaluate, 'resume': PATH}
	state['difficult_examples'] = True
	state['save_model_path'] = 'expes/models/IconArt_v1/'
	use_gpu = torch.cuda.is_available()
	state['use_gpu'] = use_gpu

	with_gt = False
	multiscale = False
	num_classes = 7
	model = resnet101_wildcat(num_classes, pretrained=True, kmax=args.k,\
	 alpha=args.alpha, num_maps=args.maps,kernel_size=args.kernel_size,same_kernel=args.same_kernel)
	
	model.train(False)
	state_dict_all = torch.load(PATH)
	best_epoch = state_dict_all["epoch"]
	best_score = state_dict_all["best_score"]
	state_dict = state_dict_all["state_dict"]
	model.load_state_dict(state_dict)
	
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
