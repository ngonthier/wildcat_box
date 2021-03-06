import argparse

import torch
import torch.nn as nn

from wildcat.engine import MultiLabelMAPEngine
from wildcat.models import resnet101_wildcat
from wildcat.voc import Voc2007Classification
from wildcat.boxesPredict import object_localization

from wildcat.tf_faster_rcnn.lib.datasets.factory import get_imdb

from shutil import copyfile
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torch.autograd import Variable
from wildcat.LatexOuput import arrayToLatex
import matplotlib.pyplot as plt
from wildcat.util import draw_bboxes

import torchvision.transforms as transforms
from wildcat.util import AveragePrecisionMeter, Warp

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']


def get_parser():
    parser = argparse.ArgumentParser(description='WILDCAT Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset (e.g. ../data/')
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
                        metavar='N', help='kernel size in the wildcat layer (default: 1)')
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
    global args, best_prec1, use_gpu
    parser = get_parser()
    args = parser.parse_args()
    train_or_test_VOC07(args)

def train_or_test_VOC07(args):

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
        model_name_base += '_' +args.init
    model_name_base += args.ext
    model_name = model_name_base+'.pth.tar'

    if args.save_init_model:
        name_init_model= model_name_base+'_initModel.pth.tar'
    else:
        name_init_model = model_name_base

    use_gpu = torch.cuda.is_available()

    sizeMaps = (args.image_size)//32 +1 # Necessary for the attention model

    num_classes = len(object_categories)

    if not(args.test) and not(args.classif):
        print("=== Training ===")

        # define dataset
        train_dataset = Voc2007Classification(args.data, 'train')
        val_dataset = Voc2007Classification(args.data, 'val')

        # load model
        if not(args.att):
            model = resnet101_wildcat(num_classes, pretrained=True, kmax=args.k,\
             alpha=args.alpha, num_maps=args.maps,kernel_size=args.kernel_size,\
             same_kernel=args.same_kernel,mode=args.mode,kernel_size_lcp=args.kernel_size_lcp,
             initialization=args.init)
        else:
            model = resnet101_attention(num_classes,sizeMaps=sizeMaps, pretrained=True,\
             num_maps=args.maps,kernel_size=args.kernel_size)

        # define loss function (criterion)
        criterion = nn.MultiLabelSoftMarginLoss()

        # define optimizer
        optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
                 'evaluate': args.evaluate, 'resume': args.resume}
        state['difficult_examples'] = True
        state['save_model_path'] = 'expes/models/VOC2007/'
        engine = MultiLabelMAPEngine(state)
        engine.learning(model, criterion, train_dataset, val_dataset, 
            optimizer,name_init_model=name_init_model)

        # Copy the checkpoint with a new name
        
        path = state['save_model_path']
        src = path + 'model_best.pth.tar'
        dst = path + model_name
        copyfile(src, dst)
    else:
        print("=== Testing of ",model_name," ===")
        PATH =  'expes/models/VOC2007/'+model_name
        state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
                 'evaluate': args.evaluate, 'resume': PATH}
        state['difficult_examples'] = True
        state['save_model_path'] = 'expes/models/VOC2007/'
        use_gpu = torch.cuda.is_available()
        state['use_gpu'] = use_gpu

        with_gt = False
        multiscale = False
        
        if not(args.att):
            model = resnet101_wildcat(num_classes, pretrained=True, kmax=args.k,\
             alpha=args.alpha, num_maps=args.maps,kernel_size=args.kernel_size,\
             same_kernel=args.same_kernel,mode=args.mode,kernel_size_lcp=args.kernel_size_lcp)
        else:
            model = resnet101_attention(num_classes,sizeMaps=sizeMaps, pretrained=True,\
             num_maps=args.maps,kernel_size=args.kernel_size)
        model.train(False)
        state_dict_all = torch.load(PATH)
        best_epoch = state_dict_all["epoch"]
        best_score = state_dict_all["best_score"]
        state_dict = state_dict_all["state_dict"]
        model.load_state_dict(state_dict)

        #classwise_feature_maps = []
        #def hook(module, input1, output2):
            #classwise_feature_maps.append(output2)

        #model.spatial_pooling.class_wise.register_forward_hook(hook)

        normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                                 std=model.image_normalization_std)
        image_transform = transforms.Compose([
                Warp(args.image_size),
                transforms.ToTensor(),
                normalize,])

        if use_gpu:
            model = model.to('cuda')
        #model = deepcopy(model).cuda()
        #model.load_state_dict(torch.load(PATH))
        #model.eval()
        #model.get_saved_model()
        #model = utils.load_model_iconart(model_path, multiscale=multiscale, scale=560)

        case = 'WILDCAT detections '
        if multiscale:
            case += ' Multiscale '
        if with_gt:
            case += ' With Ground Truth classification '
        print('===',case,'===')

        val_dataset = Voc2007Classification(args.data, 'test')
        val_dataset.transform = image_transform
        criterion = nn.MultiLabelSoftMarginLoss()

        engine = MultiLabelMAPEngine(state)
        #engine.get_saved_model()
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=state['batch_size'], shuffle=False,
                                                 num_workers=state['workers'])
        if args.classif:
            engine.validate(val_loader, model, criterion)
            if not(args.test):
                return(0)

        # if args.test:
            # print('-------- test detection --------')
            # database = 'IconArt_v1_test'
            # imdb = get_imdb('IconArt_v1_test')
            # imdb.set_use_diff(True)
            # dont_use_07_metric = True
            # imdb.set_force_dont_use_07_metric(dont_use_07_metric)

            # max_per_image = 100
            # num_images_detect =  len(imdb.image_index)
            # all_boxes_order = [[[] for _ in range(num_images_detect)] for _ in range(imdb.num_classes)]
            
            # plot = args.plot
            # if plot:
                # plt.ion()
                # import pathlib
                # folder = '/media/HDD/output_exp/WILDCAT/'+ 'WILDCAT_'+model_name_base+'/'
                # pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
            # Itera = 500
            # for i in range(num_images_detect):
                # image_name = imdb.image_path_at(i)
                # if i%Itera==0:
                    # print(i,' : ',image_name)
                # with torch.no_grad():
                    # image_raw = Image.open(image_name).convert("RGB")
                    # image_normalized = image_transform(image_raw).float()
                    # #image_normalized = torch.from_numpy(np.array(image_raw)).permute(2, 0, 1)
                    # if use_gpu:
                        # #image_normalized= image_normalized.cuda()
                        # #image_normalized= image_normalized.float().div(255)
                        # image_normalized= image_normalized.cuda()
                        # # This model need a RGB image scale between 0 and 1 reduce the mean if ImageNet
                        # #image_normalized = image_normalized.index_select(0, torch.LongTensor([2,1,0]).cuda())
                        # #image_normalized = (image_normalized - torch.Tensor([0.485, 0.456, 0.406]).cuda().view(3, 1, 1))
                    # input_var = Variable(image_normalized.unsqueeze(0))
                    # #input_var = Variable(image_normalized.unsqueeze(0), volatile=True)
                    # if not(with_gt):
                        # gt_labels_minus1 = None
                    # else:
                        # image_name_split = image_name.split('/')[-1]
                        # image_name_split = image_name_split.split('.')[0]
                        # gt_labels = np.unique(imdb._load_pascal_annotation(image_name_split)['gt_classes'])
                        # #print('gt_labels',gt_labels)
                        # gt_labels_minus1 = gt_labels -1 # Background as class 0
                        # #gt_labels_minus = None

                    # #preds, labels = object_localization(model_dict, input_var, location_type='bbox',gt_labels=None)
                    # #print('labels',labels)
                    # #print('labels.cpu().numpy()',labels.cpu().numpy())
                    
                    # preds, labels = object_localization(model, input_var, location_type='bbox',
                        # gt_labels=gt_labels_minus1,size=args.image_size)
                    # #print('gt_labels_minus1',gt_labels_minus1)
                    # #print('labels',labels)
                    # #print('preds',preds)

                    # # We need to resize the boxes to the real size of the image that have been wrap to (args.image_size, args.image_size)
                    # x_,y_,_ = np.array(image_raw).shape # when you use cv2 the x and y are inverted
                    # x_scale = x_ / args.image_size
                    # y_scale = y_ / args.image_size
                    # #print(x_scale,y_scale)
                    # for ii,box in enumerate(preds):
                        # #print(x_,y_)
                        # #print(box)
                        # (classe,origLeft, origTop, origRight, origBottom,score) = box
                        # x = origLeft * x_scale
                        # y = origTop * y_scale
                        # xmax = origRight * x_scale
                        # ymax = origBottom * y_scale
                        # preds[ii] = [classe,x, y, xmax, ymax,score]
                        # #print(preds[ii])

                    # if plot:
                        # preds_np =np.array(preds)
                        # if not(len(preds_np)==0):
                            # inds = np.where(np.array(preds_np)[:,-1]>0.)
                            # if not(len(inds)==0):
                                # inds = inds[0]
                                # preds_plot = preds_np[inds,:]
                                # labels_plot = preds_plot[:,0]
                                # img = Image.open(image_name)
                                # #img_resized = img.resize((args.image_size, args.image_size), Image.ANTIALIAS)
                                # img_draw = draw_bboxes(img, preds_plot, object_categories)
                                # plt.imshow(img_draw)
                                # tmp = image_name.split('/')[-1]
                                # name_output =  folder + tmp.split('.')[0] +'_Regions.jpg'
                                # plt.axis('off')
                                # plt.tight_layout()
                                # #plt.show()
                                # plt.savefig(name_output, dpi=300)
                                # #input('wait')

                    # for j in range(len(preds)):
                        # index_c = preds[j][0]+1
                        # if len(all_boxes_order[index_c][i])==0:
                            # all_boxes_order[index_c][i] = np.array([preds[j][1:]])
                        # else:
                            # all_boxes_order[index_c][i] = np.vstack((preds[j][1:],all_boxes_order[index_c][i]))
                    # #if not(with_gt):
                    # #    for c in labels:
                    # #        all_boxes_order[c+1][i] = np.array(all_boxes_order[c+1][i])
                    # #else: # gt cases
                    # #    for c in labels:
                    # #        all_boxes_order[c+1][i] = np.array(all_boxes_order[c+1][i])

            # output_dir = 'tmp/'
            # aps =  imdb.evaluate_detections(all_boxes_order, output_dir)
            # apsAt05 = aps
            # print("Detection score (thres = 0.5): ",database)
            # print(arrayToLatex(aps,per=True))
            # ovthresh_tab = [0.3,0.1,0.]
            # for ovthresh in ovthresh_tab:
                # aps = imdb.evaluate_localisation_ovthresh(all_boxes_order, output_dir,ovthresh)
                # if ovthresh == 0.1:
                    # apsAt01 = aps
                # print("Detection score with thres at ",ovthresh)
                # print(arrayToLatex(aps,per=True))
            # #imdb.set_use_diff(True) # Modification of the use_diff attribute in the imdb
            # #aps =  imdb.evaluate_detections(all_boxes_order, output_dir)
            # #print("Detection score with the difficult element")
            # #print(arrayToLatex(aps,per=True))
            # #imdb.set_use_diff(False)




if __name__ == '__main__':
    main()
