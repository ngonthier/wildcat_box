## Only work for IconArt

import numpy as np
import torch
from torchvision import models
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
from IntegratedGradientPytorch.utils import calculate_outputs_and_gradients, generate_entrie_images,generate_entrie_images_with_title,calculate_outputs_and_gradients_forWildcat
from IntegratedGradientPytorch.integrated_gradients import random_baseline_integrated_gradients
from IntegratedGradientPytorch.visualization import visualize
import argparse
import os
from torch.autograd import Variable
from PIL import Image, ImageDraw, ImageFont
from wildcat.models import resnet101_wildcat
from wildcat.Attention_models import resnet101_attention


from wildcat.util import AveragePrecisionMeter, Warp
import pandas as pd
import pathlib

object_categories = ['angel','Child_Jesus', 'crucifixion_of_Jesus',
                    'Mary','nudity', 'ruins','Saint_Sebastien']

def get_parser():
    parser = argparse.ArgumentParser(description='integrated-gradients for Wildcat Model for IconArt dataset')
    parser.add_argument('--cuda', action='store_true', help='if use the cuda to do the accelartion')
    parser.add_argument('--model-type', type=str, default='wildcat', help='the type of network')
    parser.add_argument('--img', type=str, default='01.jpg', help='the images name')
    parser.add_argument('--classe', type=str, default='angel', help='the class name if empty string do all the classes')
    # parser.add_argument('data', metavar='DIR',
                        # help='path to dataset (e.g. ../data/')
    parser.add_argument('--image_size', '-i', default=224, type=int,
                        metavar='N', help='image size (default: 224)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                        metavar='LR', help='learning rate for pre-trained layers')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=0, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--k', default=1, type=float,
                        metavar='N', help='number of regions (default: 1)')
    parser.add_argument('--alpha', default=1, type=float,
                        metavar='N', help='weight for the min regions (default: 1)')
    parser.add_argument('--maps', default=1, type=int,
                        metavar='N', help='number of maps per class (default: 1)')
    parser.add_argument('--kernel_size', default=1, type=int,
                        metavar='N', help='kernel size in the last layer (default: 1)')
    parser.add_argument('--kernel_size_lcp', default=1, type=int,
                        metavar='N', help='kernel size in the last layer for LCP pooling (default: 1)')
    parser.add_argument('--test', action="store_true",
                        help='Use this command to eval the detection performance of the model')
    parser.add_argument('--classif', action="store_true",
                        help='Use this command to eval the classification performance of the model')
    parser.add_argument('--plot', action="store_true",
                        help='Use this command to plot the bounding boxes.')
    parser.add_argument('--att', action="store_true",
                        help='Use this command to use the attention model.')
    parser.add_argument('--same_kernel', action="store_true",
                        help='Use this command to have the same kernels weights and biases on all the maps.')
    parser.add_argument('--save_init_model', action="store_true",
                        help='Use this command to save the model before optimization.')
    parser.add_argument('--ext', default='', type=str,
                        help='Extension added to the name of the model saved (default: '')')
    parser.add_argument('--mode', default='', type=str,
                        choices=['','Direct','LCP','LCPPReLU','LCPRReLU'],
                        help='Modification of the default WILDCAT algo to have different kernel learned (default: '')')
    parser.add_argument('--init', default='', type=str,
                        choices=['','uniform_div_std_maps','xavier_uniform','kaiming_uniform','orthogonal'],
                        help='Modification of the default WILDCAT algo to have different kernel learned (default: '')')
    return(parser)

def main():
    global args
    parser = get_parser()
    args = parser.parse_args()
    IntegratedGradient_for_oneImage(args)
    
def IntegratedGradient_for_oneImage(args):
    
    ## Pour l instant que pour IconArt et que pour wildcat !!!
    
    path_to_img = 'data/IconArt_v1/JPEGImages/'
    

    
    # check if have the space to save the results
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists('results/' + args.model_type):
        os.mkdir('results/' + args.model_type)
    
    # start to create models...
    # if args.model_type == 'inception':
        # model = models.inception_v3(pretrained=True)
    # elif args.model_type == 'resnet152':
        # model = models.resnet152(pretrained=True)
    # elif args.model_type == 'resnet18':
        # model = models.resnet18(pretrained=True)
    # elif args.model_type == 'vgg19':
        # model = models.vgg19_bn(pretrained=True) # with batch normalisation
    # elif args.model_type == 
    
    if args.model_type == 'wildcat':
        model_name_base = 'model_im'+str(args.image_size)+'_bs'+str(args.batch_size)+\
        '_lrp'+str(args.lrp)+'_lr'+str(args.lr)+'_ep'+str(args.epochs)+'_k'+str(args.k)+\
        '_a'+str(args.alpha)+'_m'+str(args.maps)
        
        if not(args.mode==''):
            model_name_base += '_'+ args.mode
        
        if args.att:
            model_name_base = 'model_att'+str(args.image_size)+'_bs'+str(args.batch_size)+\
            '_lrp'+str(args.lrp)+'_lr'+str(args.lr)+'_ep'+str(args.epochs)+'_m'+str(args.maps)
        if args.same_kernel:
            model_name_base += '_SameKernel'
        if not(args.kernel_size==1):
            model_name_base += '_ks'+str(args.kernel_size)
        if not(args.kernel_size_lcp==1):
            model_name_base += '_lcpks'+str(args.kernel_size_lcp)
        if not(args.init==''):
            model_name_base += args.init
        model_name_base += args.ext
        model_name = model_name_base+'.pth.tar'
        with_gt = False
        multiscale = False
        num_classes = 7
        if not(args.att):
            model = resnet101_wildcat(num_classes, pretrained=True, kmax=args.k,\
             alpha=args.alpha, num_maps=args.maps,kernel_size=args.kernel_size,\
             same_kernel=args.same_kernel,mode=args.mode,kernel_size_lcp=args.kernel_size_lcp)
        else:
            model = resnet101_attention(num_classes,sizeMaps=sizeMaps, pretrained=True,\
             num_maps=args.maps,kernel_size=args.kernel_size)
        model.train(False)
        PATH =  'expes/models/IconArt_v1/'+model_name
        state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
                 'evaluate': args.evaluate, 'resume': PATH}
        state['difficult_examples'] = True
        state['save_model_path'] = 'expes/models/IconArt_v1/'
        if args.cuda:
            use_gpu = torch.cuda.is_available()
        else:
            use_gpu = False
        state['use_gpu'] = use_gpu
        state_dict_all = torch.load(PATH)
        best_epoch = state_dict_all["epoch"]
        best_score = state_dict_all["best_score"]
        state_dict = state_dict_all["state_dict"]
        model.load_state_dict(state_dict)
        
        normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                                 std=model.image_normalization_std)
        image_transform = transforms.Compose([
                Warp(args.image_size),
                transforms.ToTensor(),
                normalize,])
        image_transformNoTensor = transforms.Compose([
                transforms.ToTensor(),
                normalize,])
        
    print('Model loaded')
    model.eval()
    if args.cuda:
        model.cuda()
        
    classes =  ['angel','Child_Jesus', 'crucifixion_of_Jesus',
                    'Mary','nudity', 'ruins','Saint_Sebastien']
        
    num_random_trials = 10
        
    if args.img=='':
        # We will apply the Integrated Gradient to all the image belonging to a specific class 
        # read the image
        path_file_csv = 'data/IconArt_v1/ImageSets/Main/IconArt_v1_Classification_test.csv'
        df =  pd.read_csv(path_file_csv,sep=",")
        classe_args = args.classe
        if not(classe_args==''):
            classes_concerned = [classe_args]
        else:
            classes_concerned = classes
        for classe in classes_concerned:
            df_c = df[df[classe]==1]
            for ci,c in enumerate(classes):
                if c==classe:
                    #target_label_idx = ci + 1
                    target_label_idx = ci
            for ielt, elt in enumerate(df_c['item']):
                path_results = 'results/'+ args.model_type +'/Images/'
                path_results_data = 'results/'+ args.model_type +'/data/'
                pathlib.Path(path_results).mkdir(parents=True, exist_ok=True)
                pathlib.Path(path_results_data).mkdir(parents=True, exist_ok=True)
                name_output_im = path_results  +elt +'_' + classe+ '.jpg'
                name_output_data = path_results_data  +elt +'_' + classe+'_IntegratedGrad.npy'
                if not os.path.isfile(name_output_data):
                    img_name = elt + '.jpg'
                    print(ielt,'Image :',img_name)
                    img = Image.open(path_to_img + img_name).convert("RGB")
                    img = np.array(img.resize((args.image_size,args.image_size), Image.ANTIALIAS))
                    img_normalized = image_transformNoTensor(img).float()
                    #image_normalized = torch.from_numpy(np.array(image_raw)).permute(2, 0, 1)
                    if use_gpu:
                        #image_normalized= image_normalized.cuda()
                        #image_normalized= image_normalized.float().div(255)
                        img_normalized= img_normalized.cuda()
                    
                    # img = cv2.imread(path_to_img + img_name)
                    # img = cv2.resize(img, (args.image_size, args.image_size))
                    # img_bgr = img.astype(np.float32) 
                    # img = img_bgr[:, :, (2, 1, 0)] # Image in RGB
                    # if args.model_type == 'wildcat':
                        # #image_transformNoTensor(img)
                        # img_normalized = (img - model.image_normalization_mean) / model.image_normalization_std

                    # calculate the gradient and the label index
                    #gradients, label_index = calculate_outputs_and_gradients([img_normalized], model, target_label_idx, args.cuda)
                    gradients, label_index, output,full_output = calculate_outputs_and_gradients_forWildcat([img_normalized], model, target_label_idx, args.cuda) # For Wildcat i.e. need to normalise the image before
                    
                    print('full vecto output',full_output)
                    print('output value for the label index',output)
                    print('label_index',label_index,' : ',classes[label_index])
                    title = elt + ' ' + classes[label_index] +' : '+str(output.detach().cpu().numpy()[0])
                    gradients = np.transpose(gradients[0], (1, 2, 0))
                    img_gradient_overlay = visualize(gradients, img, clip_above_percentile=99, clip_below_percentile=0, overlay=True, mask_mode=True)
                    img_gradient = visualize(gradients, img, clip_above_percentile=99, clip_below_percentile=0, overlay=False)

                    # calculate the integrated gradients 

                    attributions = random_baseline_integrated_gradients(img, model, label_index, calculate_outputs_and_gradients, \
                                                                        steps=50, num_random_trials=num_random_trials, cuda=args.cuda)
                    img_integrated_gradient_overlay = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0, \
                                                                overlay=True, mask_mode=True)
                    img_integrated_gradient = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0, overlay=False)
                    # output_img = generate_entrie_images(img, img_gradient, img_gradient_overlay, img_integrated_gradient, \
                                                        # img_integrated_gradient_overlay)
                    output_img = generate_entrie_images_with_title(img, img_gradient, img_gradient_overlay, img_integrated_gradient, \
                                                        img_integrated_gradient_overlay,title)
                    #print(output_img)

                    cv2.imwrite(name_output_im , np.uint8(output_img))
                    np.save(name_output_data, img_integrated_gradient)
    
    else:
        # We will consider only one image
        img = cv2.imread(path_to_img + args.img)
        # if args.model_type == 'inception':
            # # the input image's size is different
            # img = cv2.resize(img, (299, 299))
        img = cv2.resize(img, (args.image_size, args.image_size))
        img = img.astype(np.float32) 
        img = img[:, :, (2, 1, 0)]
        if args.model_type == 'wildcat':
            image_transform(img)
        print('Image read')
        # calculate the gradient and the label index
        gradients, label_index = calculate_outputs_and_gradients([img], model, None, args.cuda)
        print('label_index',label_index)
        gradients = np.transpose(gradients[0], (1, 2, 0))
        img_gradient_overlay = visualize(gradients, img, clip_above_percentile=99, clip_below_percentile=0, overlay=True, mask_mode=True)
        img_gradient = visualize(gradients, img, clip_above_percentile=99, clip_below_percentile=0, overlay=False)

        # calculate the integrated gradients 
        attributions = random_baseline_integrated_gradients(img, model, label_index, calculate_outputs_and_gradients, \
                                                            steps=50, num_random_trials=num_random_trials, cuda=args.cuda)
        img_integrated_gradient_overlay = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0, \
                                                    overlay=True, mask_mode=True)
        img_integrated_gradient = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0, overlay=False)
        output_img = generate_entrie_images(img, img_gradient, img_gradient_overlay, img_integrated_gradient, \
                                            img_integrated_gradient_overlay)
        cv2.imwrite('results/' + args.model_type + '/' + args.img, np.uint8(output_img))

if __name__ == '__main__':
    main()
    
