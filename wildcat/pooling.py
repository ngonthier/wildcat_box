import sys
import torch
import torch.nn as nn
from torch.autograd import Function, Variable


class WildcatPool2dFunction(Function):
    def __init__(self, kmax, kmin, alpha):
        super(WildcatPool2dFunction, self).__init__()
        self.kmax = kmax
        self.kmin = kmin
        self.alpha = alpha

    def get_positive_k(self, k, n):
        if k <= 0:
            return 0
        elif k < 1:
            return round(k * n)
        elif k > n:
            return int(n)
        else:
            return int(k)

    def forward(self, input):
        batch_size = input.size(0)
        num_channels = input.size(1)
        h = input.size(2)
        w = input.size(3)

        n = h * w  # number of regions

        kmax = self.get_positive_k(self.kmax, n)
        kmin = self.get_positive_k(self.kmin, n)

        sorted, indices = input.new(), input.new().long()
        torch.sort(input.view(batch_size, num_channels, n), dim=2, descending=True, out=(sorted, indices))

        self.indices_max = indices.narrow(2, 0, kmax)
        output = sorted.narrow(2, 0, kmax).sum(2).div_(kmax)

        if kmin > 0 and self.alpha is not 0:
            self.indices_min = indices.narrow(2, n - kmin, kmin)
            output.add_(sorted.narrow(2, n - kmin, kmin).sum(2).mul_(self.alpha / kmin)).div_(2)

        self.save_for_backward(input)
        return output.view(batch_size, num_channels)

    def backward(self, grad_output):

        input, = self.saved_tensors

        batch_size = input.size(0)
        num_channels = input.size(1)
        h = input.size(2)
        w = input.size(3)

        n = h * w  # number of regions

        kmax = self.get_positive_k(self.kmax, n)
        kmin = self.get_positive_k(self.kmin, n)

        grad_output_max = grad_output.view(batch_size, num_channels, 1).expand(batch_size, num_channels, kmax)

        grad_input = grad_output.new().resize_(batch_size, num_channels, n).fill_(0).scatter_(2, self.indices_max,
                                                                                              grad_output_max).div_(
            kmax)

        if kmin > 0 and self.alpha is not 0:
            grad_output_min = grad_output.view(batch_size, num_channels, 1).expand(batch_size, num_channels, kmin)
            grad_input_min = grad_output.new().resize_(batch_size, num_channels, n).fill_(0).scatter_(2,
                                                                                                      self.indices_min,
                                                                                                      grad_output_min).mul_(
                self.alpha / kmin)
            grad_input.add_(grad_input_min).div_(2)

        return grad_input.view(batch_size, num_channels, h, w)
        
class DirectMaxPlusAlphaMinPool2dFunction(Function):
    def __init__(self,num_maps, kmax, kmin, alpha):
        super(DirectMaxPlusAlphaMinPool2dFunction, self).__init__()
        self.num_maps = num_maps
        self.kmax = kmax
        self.kmin = kmin
        self.alpha = alpha

    def get_positive_k(self, k, n):
        if k <= 0:
            return 0
        elif k < 1:
            return round(k * n)
        elif k > n:
            return int(n)
        else:
            return int(k)

    def forward(self, input):
        # input of size batch, self.num_maps *  number classe, h, w
        batch_size, num_channels, h, w = input.size()
        n = h * w * self.num_maps  # number of regions
        num_outputs = int(num_channels / self.num_maps)
        #x = input.view(batch_size, num_outputs, self.num_maps, h, w)
        kmax = self.get_positive_k(self.kmax, n)
        kmin = self.get_positive_k(self.kmin, n)

        if num_channels % self.num_maps != 0:
            print('Error in ClassWisePoolFunction. The number of channels has to be a multiple of the number of maps per class')
            sys.exit(-1)

        sorted, indices = input.new(), input.new().long()
        torch.sort(input.view(batch_size, num_outputs, n), dim=2, descending=True, out=(sorted, indices))

        self.indices_max = indices.narrow(2, 0, kmax)
        output = sorted.narrow(2, 0, kmax).sum(2).div_(kmax)

        if kmin > 0 and self.alpha is not 0:
            self.indices_min = indices.narrow(2, n - kmin, kmin)
            output.add_(sorted.narrow(2, n - kmin, kmin).sum(2).mul_(self.alpha / kmin)).div_(2)

        self.save_for_backward(input)
        return output.view(batch_size, num_outputs) # batch_size, number of classes

    def backward(self, grad_output):
        input, = self.saved_tensors

        batch_size, num_channels, h, w = input.size()

        n = h * w * self.num_maps  # number of regions
        num_outputs = grad_output.size(1) # number of classes
        
        kmax = self.get_positive_k(self.kmax, n)
        kmin = self.get_positive_k(self.kmin, n)
        
        grad_output_max = grad_output.view(batch_size, num_outputs, 1).expand(batch_size, num_outputs, kmax)
        grad_input = grad_output.new().resize_(batch_size, num_outputs, n).fill_(0).scatter_(2, self.indices_max,
                                                                                                grad_output_max).div_(kmax).contiguous()

        # grad_input = grad_output.view(batch_size, num_outputs, 1, h, w).expand(batch_size, num_outputs, self.num_maps,
                                                                               # h, w).contiguous()
        

        if kmin > 0 and self.alpha is not 0:
            grad_output_min = grad_output.view(batch_size, num_outputs, 1).expand(batch_size, num_outputs, kmin)
            grad_input_min = grad_output.new().resize_(batch_size, num_outputs, n).fill_(0).scatter_(2,
                                                                                                      self.indices_min,
                                                                                                      grad_output_min).mul_(
                self.alpha / kmin).contiguous()
            grad_input.add_(grad_input_min).div_(2)

        return grad_input.view(batch_size, num_channels, h, w)


class WildcatPool2d(nn.Module):
    def __init__(self, kmax=1, kmin=None, alpha=1):
        super(WildcatPool2d, self).__init__()
        self.kmax = kmax
        self.kmin = kmin
        if self.kmin is None:
            self.kmin = self.kmax
        self.alpha = alpha

    def forward(self, input):
        return WildcatPool2dFunction(self.kmax, self.kmin, self.alpha)(input)

    def __repr__(self):
        return self.__class__.__name__ + ' (kmax=' + str(self.kmax) + ', kmin=' + str(self.kmin) + ', alpha=' + str(
            self.alpha) + ')'
            
class DirectMaxPlusAlphaMinPool2d(nn.Module):
    def __init__(self,num_maps, kmax=1, kmin=None, alpha=1):
        super(DirectMaxPlusAlphaMinPool2d, self).__init__()
        self.num_maps = num_maps
        self.kmax = kmax
        self.kmin = kmin
        if self.kmin is None:
            self.kmin = self.kmax
        self.alpha = alpha

    def forward(self, input):
        return DirectMaxPlusAlphaMinPool2dFunction(self.num_maps,self.kmax, self.kmin, self.alpha)(input)

    def __repr__(self):
        return self.__class__.__name__ + ' (num_maps=' + str(self.num_maps) + ', kmax=' + str(self.kmax) + ', kmin=' + str(self.kmin) + ', alpha=' + str(
            self.alpha) + ')'


class ClassWisePoolFunction(Function):
    def __init__(self, num_maps):
        super(ClassWisePoolFunction, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        # batch dimension
        batch_size, num_channels, h, w = input.size()

        if num_channels % self.num_maps != 0:
            print('Error in ClassWisePoolFunction. The number of channels has to be a multiple of the number of maps per class')
            sys.exit(-1)

        num_outputs = int(num_channels / self.num_maps)
        x = input.view(batch_size, num_outputs, self.num_maps, h, w)
        output = torch.sum(x, 2)
        self.save_for_backward(input)
        return output.view(batch_size, num_outputs, h, w) / self.num_maps

    def backward(self, grad_output):
        input, = self.saved_tensors

        # batch dimension
        batch_size, num_channels, h, w = input.size()
        num_outputs = grad_output.size(1)

        grad_input = grad_output.view(batch_size, num_outputs, 1, h, w).expand(batch_size, num_outputs, self.num_maps,
                                                                               h, w).contiguous()

        return grad_input.view(batch_size, num_channels, h, w)

def learned_pooling(num_maps,kernel_size):
    conv2d = nn.Sequential(
            nn.Conv2d(num_maps, 1, kernel_size=kernel_size, stride=1, padding=0, bias=True))
            # in_channels, out_channels, kernel_size
    return(conv2d)
    
class LCPPool(nn.Module): # Replace Function by nn.Module
    def __init__(self,num_classes,num_maps,kernel_size=1):
        super(LCPPool, self).__init__()
        self.num_classes = num_classes
        self.num_maps = num_maps
        self.kernel_size = kernel_size
        
        self.convs = nn.Sequential(
            nn.Conv2d(num_maps*num_classes, num_classes, kernel_size=kernel_size, stride=1, padding=0, bias=True,groups=num_classes))
        
        # self.branches = nn.ModuleList([learned_pooling(num_maps,kernel_size) for i in range(self.num_classes)])
        # for i, branch in enumerate(self.branches):
            # self.add_module(str(i), branch)
        # self.learned_pooling = nn.Sequential(
            # nn.Conv2d( self.num_maps, 1, kernel_size=self.kernel_size, stride=1, padding=0, bias=True)) # We keep only one element per image
        # 
        
    def forward(self, input):
        # batch dimension
        batch_size, num_channels, h, w = input.size()

        if num_channels % self.num_maps != 0:
            print('Error in LearnedClassWisePoolFunction. The number of channels has to be a multiple of the number of maps per class')
            sys.exit(-1)

        num_outputs = int(num_channels / self.num_maps)
        
        output = self.convs(input)
        return(output)
        # x = input.view(batch_size,num_outputs, self.num_maps, h, w)
        
        # #print('before cat')
        
        # output = torch.cat([b(x[:,i,:,:]) for i,b in enumerate(self.branches)], 1)
        
        # # print(self.learned_pooling)
        # # print(x)
        # # output = self.learned_pooling(x)
        # #print(output)

        # #self.save_for_backward(input)
        # import numpy as np
        # print(np.shape(output.detach().cpu().numpy()))
        #return output.view(batch_size, self.num_classes, h, w) # Normalisation par le nombre de maps / self.num_maps 

    # def backward(self, grad_output):
        # input, = self.saved_tensors

        # # batch dimension
        # batch_size, num_channels, h, w = input.size()
        # num_outputs = grad_output.size(1)

        # grad_input = grad_output.view(batch_size, num_outputs, 1, h, w).expand(batch_size, num_outputs, self.num_maps,
                                                                               # h, w).contiguous()

        # return grad_input.view(batch_size, num_channels, h, w)
        
    def __repr__(self):
        return self.__class__.__name__ + ' (num_maps={num_maps})'.format(num_maps=self.num_maps)+ ' (kernel_size={kernel_size})'.format(kernel_size=self.kernel_size)

class ClassWisePool(nn.Module):
    def __init__(self, num_maps):
        super(ClassWisePool, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        return ClassWisePoolFunction(self.num_maps)(input)

    def __repr__(self):
        return self.__class__.__name__ + ' (num_maps={num_maps})'.format(num_maps=self.num_maps)
