import torch
import shutil
from models.resnet import resnet32, resnet32_rep, resnet32_liner
from models.resnet_ride import *
from models.resnet_bcl import *
from models.resnet_ncl import *

import torch.nn as nn
import torchvision.models as models

def get_model(args, num_class_list):
    if args.loss_fn in ['ride']:
        model = resnet32_ride(args.num_class, num_experts=args.num_experts).cuda()
        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    elif args.loss_fn in ['ncl']:
        model = ncl_model(args, num_class_list)
        print('    Total params: %.2fM' % (sum(p.numel() for p in model['model'].parameters())/1000000.0))

    elif args.loss_fn in ['bcl']:
        model = bcl_model(args.num_class, use_norm=args.use_norm).cuda()
        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))


    else:
        model = resnet32(args.num_class, use_norm= args.loss_fn == 'ldam_drw').cuda()
        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        
    torch.backends.cudnn.benchmark = True
    return model

def load_model(args):
    if args.loss_fn == 'ride' and args.num_experts == 3 and args.ride_distill:
        print("---- ride teacher load ----")
        filepath = os.path.join(args.out, 'checkpoint_teacher.pth.tar')
        if os.path.isfile(filepath):
            pass    
        else:
            shutil.copy2(os.path.join(args.out, 'checkpoint.pth.tar'), os.path.join(args.out, 'checkpoint_teacher.pth.tar'))
        checkpoint = torch.load(filepath)
        teacher = resnet32_ride(args.num_class, num_experts = 6).cuda()
        teacher.load_state_dict(checkpoint['state_dict'])
    else:
        teacher = None
    return teacher
    

def get_model_by_name(args, name, num_class_list):
    if name in ['ride']:
        model = resnet32_ride(args.num_class, num_experts=args.num_experts).cuda()

    elif name in ['ncl']:
        model = ncl_model(args, num_class_list)

    elif name in ['bcl']:
        model = bcl_model(args.num_class, use_norm=args.use_norm).cuda()

    else:
        model = resnet32(args.num_class, use_norm= args.loss_fn == 'ldam_drw').cuda()
        
    torch.backends.cudnn.benchmark = True
    return model

def get_rep_layer(args):
    model = resnet32_rep(args.num_class, use_norm= False).cuda()
    return model

def get_liner(num_class, loss_name,use_norm = False):
    if loss_name == 'bcl':
        return bcl_classfir(num_class,use_norm).cuda()
    model = resnet32_liner(num_class, use_norm= loss_name == 'ldam_drw').cuda()
    return model

class Net(nn.Module):
    def __init__(self,args, use_norm=False):
        super(Net, self).__init__()
        self.num_class = args.num_class
        self.encoder = get_rep_layer(args)
        self.decoders =  torch.nn.ModuleDict()
        self.bcl = args.bcl
        self.ncl = args.ncl

        if self.bcl:
            self.bcl_decoder = bcl_classfir(self.num_class,use_norm).cuda()
        
        for t in args.tasks:
            if t == 'bcl':
                continue
            self.decoders[t] = resnet32_liner(self.num_class, use_norm = (t == 'ldam_drw') ).cuda()
    
    def forward(self, x,output_type = 'feat'):
        out_put = {}
        
        if self.bcl:
            #TODO  some other loss fun can get an erro in this
            if self.training:
                batch_size = x[0].shape[0]
                inputs_b = torch.cat([x[0], x[1], x[2]], dim=0).cuda(non_blocking=True)
            else:
                inputs_b = x.cuda(non_blocking=True)
            
            x = self.encoder(inputs_b)
            feat_mlp, logits, centers = self.bcl_decoder(x)
            
            if self.training:
                centers = centers[:self.num_class]
                f1, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
                features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
                logits = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)[0]
                out_put['bcl'] = centers, logits, features
                x = torch.split(x, [batch_size, batch_size, batch_size], dim=0)[0]
            else:
                out_put['bcl'] = logits

        else:
            x = self.encoder(x.cuda(non_blocking=True))
        
        for t in self.decoders.keys():
            out_put[t] = self.decoders[t](x,output_type)
        return out_put

        
    








