import torch
import shutil
from models.resnet import resnet32, resnet32_rep, resnet32_liner
from models.resnet_bcl import *
from models.moco_gml import MoCo_cls
from models.resnet_large import resnet50_rep,resnet50_liner
from models.res_shike import resnet32_moe_cls,resnet50_moe_cls

import torch.nn as nn


def dataset_argument(args):
    if args.dataset == 'cifar100':
        args.num_class = 100
    elif args.dataset == 'imgnet':
        args.num_class = 1000
    elif args.dataset == 'inat18':
        args.num_class = 8142
    else :
        args.num_class = 10
    return args


def get_rep_layer(args):
    if args.dataset == 'cifar100':
        return resnet32_rep(args.num_class, use_norm= False).cuda()
    else :
        return resnet50_rep().cuda()

def get_liner(args,num_class, loss_name,use_norm = False):
    if args.dataset == 'cifar100':
        if loss_name == 'bcl':
            return bcl_classfir(num_class,use_norm).cuda()
        elif loss_name == 'shike':
            return  resnet32_moe_cls(num_class,use_norm)
        return resnet32_liner(num_class, use_norm= loss_name == 'ldam_drw').cuda()
    else:
        if loss_name == 'bcl':
            return bcl_classfir(num_class,use_norm,dim_in=2048).cuda()
        elif loss_name == 'shike':
            return resnet50_moe_cls(num_class,use_norm)
        return resnet50_liner(num_class, use_norm= loss_name == 'ldam_drw').cuda()

from MultiBackward import Weight_acc
class Net(nn.Module):
    def __init__(self,args, use_norm=False):
        super(Net, self).__init__()
        self.num_class = args.num_class
        self.encoder = get_rep_layer(args)
        self.decoders =  torch.nn.ModuleDict()
        self.unnorm_task = ['bcl','shike']

        for t in self.unnorm_task:
            setattr(self,t,getattr(args,t))                

        for t in args.tasks:
            if t in self.unnorm_task:
                continue
            self.decoders[t] = get_liner(args, self.num_class,t, use_norm = (t in ['ldam_drw','kps'] ) ).cuda()

        if self.bcl:
            self.bcl_decoder = get_liner(args,self.num_class,'bcl',use_norm).cuda()
        
        if self.shike:
            self.shike_decoder = get_liner(args,self.num_class,'shike',True).cuda()
            self.shike_decoder.call_last_layer(self.encoder.get_last_layer())
        self.W_acc = Weight_acc(self.num_class,args.tasks)
    
    def forward(self, x ,output_type = 'feat',crt = False):
        out_put = {}
        if isinstance(x,torch.Tensor):
            x = x.cuda(non_blocking=True)
            x,shallow_outs = self.encoder(x)

            for t in self.decoders.keys():
                out_put[t] = self.decoders[t](x,output_type)

            if self.bcl:
                feat_mlp, logits, centers = self.bcl_decoder(x)
                out_put['bcl'] = logits

        else:
            shallow_outs = [0,0,0]
            for i in range(3):
                x[i] = x[i].cuda(non_blocking=True)
                x[i],shallow_outs[i] = self.encoder(x[i])
            
            for t in self.decoders.keys():
                out_put[t] = self.decoders[t](x[0],output_type)
 
            batch_size = x[0].shape[0]
            feat_mlp, logits, centers = self.bcl_decoder(torch.cat(x))
            centers = centers[:self.num_class]
            f1, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
            features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
            logits = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)[0]
            out_put['bcl'] = centers, logits, features
            
            shallow_outs = shallow_outs[0]
        
        if self.shike:
            out_put['shike'] = self.shike_decoder(shallow_outs,crt)#[30* self.shike_decoder[i](x,output_type) for i in range(3)]
            
        return out_put

       
    








