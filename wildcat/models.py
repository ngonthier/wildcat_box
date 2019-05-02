import torch.nn as nn
import torchvision.models as models
import torch
import numpy as np

from wildcat.pooling import WildcatPool2d, ClassWisePool

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.Tensor(torch.index_select(a, dim, order_index))

class ResNetWSL(nn.Module):

    def __init__(self, model, num_classes,num_maps, pooling=WildcatPool2d(), dense=False,kernel_size=1,same_kernel=False):
        super(ResNetWSL, self).__init__()

        self.num_classes = num_classes
        self.num_maps = num_maps

        self.dense = dense

        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4)

        # classification layer
        num_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(num_features, num_classes*num_maps, kernel_size=kernel_size, stride=1, padding=0, bias=True))

        def init_weights(m):
            import math
            if type(m) == nn.Conv2d:
                n = m.in_channels
                for k in m.kernel_size:
                    n *= k
                stdv = 1. / math.sqrt(n)
                stdv /= 2. * math.sqrt(2.)
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)
                #m.weight.data.fill_(0.0)
                #m.bias.data.fill_(0.0)
                #print(m.weight)
                
        def repeat_weights(m):
            if type(m) == nn.Conv2d:
                weights = m.weight.data
                weights = weights.view(self.num_classes,self.num_maps,num_features)
                weights_c = torch.sum(weights, 1) / self.num_maps
                weights_c = weights_c.view(self.num_classes,1,num_features)
                tile_weights_c = tile(weights_c,dim=1,n_tile=self.num_maps).view(m.weight.data.size())
                m.weight.data.copy_(tile_weights_c)
                biases = m.bias.data
                biases = biases.view(self.num_classes,self.num_maps)
                biases_c = torch.sum(biases, 1) / self.num_maps
                biases_c = biases_c.view(self.num_classes,1)
                tile_biases_c = tile(biases_c,dim=1,n_tile=self.num_maps).view(m.bias.data.size())
                m.bias.data.copy_(tile_biases_c)
        
        if same_kernel:
            self.classifier.apply(repeat_weights)


        
        self.spatial_pooling = pooling

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        if not self.dense:
            x = self.spatial_pooling(x)
            
        ## Code pour voir le gradient et les kernels : uncomment to print kernel values
        #gradconv = self.classifier[0].weight.grad
        #weight = self.classifier[0].weight.clone()
        #weight = weight.detach().cpu().numpy()
        #import numpy as np

        #if not(gradconv is None):
            #gradconv= gradconv.cpu().numpy()
            ##print('weight.shape',weight.shape)
            ##print('gradconv.shape',gradconv.shape)
            #num_classes = 7
            #num_maps = gradconv.shape[0]/num_classes
            #gradconv_class0 = gradconv[0:2,:,0,0]
            ##with np.set_printoptions(threshold=2000):
            ##print(gradconv[:,0:5,0,0])
            #a = gradconv_class0[0,:]
            #b = gradconv_class0[1,:]
            #print('grada',a[0:5])
            #print('gradb',b[0:5])
            ##print(a.shape)
            ##print(b.shape)
            
            #print('Equal grada gradb ? :',np.all(np.equal(a,b)))
            #print('max abs(a-b)',np.max(np.abs(a-b)))
            #print('max abs(a-b)/abs(a)',np.max(np.abs(a-b)/np.abs(a)))
            #print(' norm(a-b)/norm2(a)',np.linalg.norm(a-b)/np.linalg.norm(a))
        #else:
            #print(gradconv)
        #print('kernela',weight[0,0:5,0,0])
        #print('kernelb',weight[1,0:5,0,0])
        #a = weight[0,:,0,0]
        #b = weight[1,:,0,0]
        #print('Equal kernela kernelb ? :',np.all(np.equal(a,b)))
        #print('max abs(a-b)',np.max(np.abs(a-b)))
        #print('max abs(a-b)/abs(a)',np.max(np.abs(a-b)/np.abs(a)))
        #print('max abs(a-b)/norm2(a)',np.max(np.abs(a-b)/np.linalg.norm(a)))
        
        return x

    def get_config_optim(self, lr, lrp):
        return [{'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.classifier.parameters()},
                {'params': self.spatial_pooling.parameters()}]


def resnet50_wildcat(num_classes, pretrained=True, kmax=1, kmin=None, alpha=1, num_maps=1,kernel_size=1,same_kernel=False):
    model = models.resnet50(pretrained)
    pooling = nn.Sequential()
    pooling.add_module('class_wise', ClassWisePool(num_maps))
    pooling.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))
    return ResNetWSL(model, num_classes , num_maps, pooling=pooling,\
        kernel_size=kernel_size,same_kernel=same_kernel)


def resnet101_wildcat(num_classes, pretrained=True, kmax=1, kmin=None, alpha=1, num_maps=1,kernel_size=1,same_kernel=False):
    model = models.resnet101(pretrained)
    pooling = nn.Sequential()
    pooling.add_module('class_wise', ClassWisePool(num_maps))
    pooling.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))
    return ResNetWSL(model, num_classes , num_maps, pooling=pooling,\
        kernel_size=kernel_size,same_kernel=same_kernel)
