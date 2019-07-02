import torch.nn as nn

def init_weights_xavier_uniform(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
def init_weights_kaiming_uniform(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def init_weights_orthogonal(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0.01)
        
def init_weights_uniform_div_std_num_maps(num_maps,m):
    if type(m) == nn.Conv2d:
        n = m.in_channels
        for k in m.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        stdv /= math.sqrt(num_maps)
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)
