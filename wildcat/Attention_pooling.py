import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,sizeMaps,num_maps,num_classes):
        super(Attention, self).__init__()
        self.num_maps = num_maps
        #self.L = 500 # Number of neurons of the last layers
        self.D = 128
        self.L = 128
        self.K = 1
        self.num_classes = num_classes 
        self.sizeMaps = sizeMaps
        
        #self.feature_extractor_part1 = nn.Sequential(
            #nn.Conv2d(1, 20, kernel_size=5),
            #nn.ReLU(),
            #nn.MaxPool2d(2, stride=2),
            #nn.Conv2d(20, 50, kernel_size=5),
            #nn.ReLU(),
            #nn.MaxPool2d(2, stride=2)
        #)

        #self.feature_extractor_part2 = nn.Sequential(
            #nn.Linear(self.sizeMaps*self.sizeMaps,self.L),
            #nn.ReLU(),
        #)

        self.attention = nn.Sequential(
            nn.Linear(self.num_classes*self.num_maps, self.L), # Here self.num_classes is the number of features we keep per region
            nn.Tanh(),
            nn.Linear(self.L, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.num_classes*self.num_maps, self.num_classes),
            #nn.Sigmoid() # Because there is a sigmoid in multilabel_soft_margin_loss
        )

    def forward(self, x):
        #print('x',x.shape)
        batch_size = x.size(0)
        num_channels = x.size(1)
        h = x.size(2)
        w = x.size(3)
        # batch, classes, h,w
        
        x = x.view(batch_size,num_channels,h*w) # batch size, self.num_classes, (h/32 +1) *( w/32 +1)
        xx = torch.transpose(x,1,2).contiguous()
        #print('x',x.shape) 
        #print('xx',xx.shape) 
        H = xx.view(-1,num_channels)
        #print('H',H.shape)
        A = self.attention(H)
        #print('A',A.shape)
        A = A.view(batch_size,h*w,-1)
        #print('A',A.shape)
        A = F.softmax(A, dim=1) # Softmax over the regions
        #print('A',A.shape)
        M = torch.bmm(x,A)  # KxL
        #print('M',M.shape)
        M = M.view(batch_size,-1)
        #print('M',M.shape)

        Y_prob = self.classifier(M)
        #print('Y',Y_prob.shape)
        #Y_hat = torch.ge(Y_prob, 0.5).float()

        #return Y_prob, Y_hat, A
        return Y_prob
