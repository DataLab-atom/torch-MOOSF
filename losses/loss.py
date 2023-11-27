import torch
import torch.optim as optim
from bisect import bisect_right

from losses.bs import BS
from losses.ce_drw import CE_DRW
from losses.ce import CE
from losses.ldam_drw import LDAM_DRW
from losses.bcl import BCLLoss
from losses.kps import KPSLoss
from losses.GML import GMLLoss
from losses.SHIKE import SHIKELoss
from torch.optim import lr_scheduler


def get_optimizer(args, model):
    _model = model['model'] if args.loss_fn == 'ncl' else model
    return optim.SGD(_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd,
                     nesterov=args.nesterov)

def get_scheduler(args, optimizer):
    if args.scheduler == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min = 0)
    elif args.scheduler == 'warmup':
        return None


def get_loss_by_name(loss_name, N_SAMPLES_PER_CLASS,args,confidence = None):
    if loss_name == 'ce':
        train_criterion = CE().cuda()
    elif loss_name == 'bs':
        train_criterion = BS(N_SAMPLES_PER_CLASS).cuda()
    elif loss_name == 'ce_drw':
        train_criterion = CE_DRW(cls_num_list=N_SAMPLES_PER_CLASS, reweight_epoch=160).cuda()
    elif loss_name == 'ldam_drw':
        train_criterion = LDAM_DRW(cls_num_list=N_SAMPLES_PER_CLASS, reweight_epoch=160, max_m=0.5, s=30).cuda()
    elif loss_name == 'bcl':
        train_criterion = BCLLoss(N_SAMPLES_PER_CLASS).cuda()
    elif loss_name == 'kps':
        train_criterion = KPSLoss(cls_num_list=N_SAMPLES_PER_CLASS, max_m = 0.1, s = 3).cuda()
    elif loss_name == 'gml':
        train_criterion = GMLLoss(cls_num_list=N_SAMPLES_PER_CLASS,num_classes=args.num_class).cuda()
    elif loss_name == 'shike':
        train_criterion = SHIKELoss(args).cuda()
    
    return train_criterion

