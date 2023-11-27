import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import normal

class KPSLoss(nn.Module):
    r"""Implement of KPS Loss :
    Args:
    """

    def __init__(self, cls_num_list, max_m=0.5, weight= None, s=30):
        super(KPSLoss, self).__init__()
        assert s > 0

        s_list = torch.cuda.FloatTensor(cls_num_list)
        s_list = s_list*(50/s_list.min())
        s_list = torch.log(s_list) #torch.log(s_list) #s_list**(1/4) #torch.log(s_list) #s_list**(1/4)#s_list = torch.log(s_list)**2  #s_list**(1/5)
        s_list = s_list*(1/s_list.min()) #s+ s_list #
        self.s_list = s_list
        self.s = s
        
        m_list =  torch.flip(self.s_list, dims=[0])
        m_list = m_list * (max_m / m_list.max())
        self.m_list = m_list
        self.weight = weight
        

    def forward(self, input, label, epoch):

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = input*self.s_list
        phi = cosine - self.m_list
        # --------------------------- convert label to one-hot ---------------------------
        index = torch.zeros_like(input, dtype=torch.uint8)
        index.scatter_(1, label.data.view(-1, 1), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        #output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output = torch.where(index > 0 , phi, cosine)
        if epoch < 120:
            output *= self.s
        else:
            index_float = index.type(torch.cuda.FloatTensor)
            batch_s = torch.flip(self.s_list, dims=[0])*self.s
            batch_s = torch.clamp(batch_s, self.s, 50)    #s过大不好。s过大会发散。          
            batch_s = torch.matmul(batch_s[None, :], index_float.transpose(0,1)) 
            batch_s = batch_s.view((-1, 1))           
            output *= batch_s
        
        return F.cross_entropy(output, label)




