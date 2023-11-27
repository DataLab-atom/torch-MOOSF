from __future__ import print_function

import argparse, os, shutil, time, random, math
import numpy as np
import torch.optim as optim
from imbalance_data.lt_data import LT_Dataset
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torch.nn.functional as F

import losses

from datasets.cifar100 import *

from train.train import *
from train.validate import *

from models.net import *
from losses.loss import *

from utils.config import *
from utils.plot import *
from utils.common import make_imb_data, save_checkpoint, hms_string

from utils.logger import logger

args = parse_args()
reproducibility(args.seed)
args = dataset_argument(args)
args.logger = logger(args)

best_acc = 0 # best test accuracy
criterion = nn.CrossEntropyLoss()

def main():
    global best_acc
    args.num_class = 1000 if args.dataset == 'imgnet' else 8142
    try:
        assert args.num_max <= 50000. / args.num_class
    except AssertionError:
        args.num_max = int(50000 / args.num_class)

    tasks = [args.loss_1]
    if args.loss_2 != '':
        tasks.append(args.loss_2)
    
    if args.loss_3 != '':
        tasks.append(args.loss_3)

    print(tasks)
    if 'bcl' in tasks:
        args.loss_fn = 'bcl'
        #把bcl放到第一个，方便下面释放显存                
        tasks.remove('bcl')
        tasks = ['bcl'] + tasks
    
    if args.dataset == 'imgnet':
        train_config = 'config/ImageNet/ImageNet_LT_train.txt'
        valid_config = 'config/ImageNet/ImageNet_LT_test.txt'
        root =  '/home/data/ImageNet/Data/CLS-LOC/'
    elif args.dataset == 'inat':
        train_config = 'config/iNaturalist/iNaturalist18_train.txt'
        valid_config = 'config/iNaturalist/iNaturalist18_val.txt'
        root ='/home/data/iNaturalist' #需要改成本地路径

    train_dataset = LT_Dataset(root, train_config, args, args.dataset, args.loss_fn, 
            use_randaug=args.use_randaug, split='train', aug_prob=args.aug_prob,
            upgrade=args.cuda_upgrade, downgrade=args.cuda_downgrade)
    val_dataset = LT_Dataset(root, valid_config, args, args.dataset, args.loss_fn, split = 'valid')

    num_classes = len(np.unique(train_dataset.targets))
    assert num_classes == 1000 if args.dataset == 'imgnet' else 8142
    args.num_class = num_classes

    cls_num_list = [0] * num_classes
    for label in train_dataset.targets:
        cls_num_list[label] += 1
        
    args.cls_num_list = cls_num_list
    train_cls_num_list = np.array(cls_num_list)
    
    train_sampler = None

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, sampler=train_sampler)

    testloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)    
    
    cls_num_list_cuda = torch.from_numpy(np.array(cls_num_list)).float().cuda()

    if 'CMO' in args.data_aug:
        cls_weight = 1.0 / (np.array(cls_num_list))
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        labels = trainloader.dataset.targets
        samples_weight = np.array([cls_weight[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(labels), replacement=True)
        weighted_trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=sampler)
    else:
        weighted_trainloader = None

    print ("==> creating {}".format(args.network))

    model = {}
    model['rep'] = get_rep_layer(args)

    for t in tasks:
       model[t] =  get_liner(args.num_class, t)

    train_criterion = {}

    for t in tasks:
       train_criterion[t] = get_loss_by_name(t, cls_num_list,args)

    model_params = []
    for m in model:
        model_params += model[m].parameters()
    
    optimizer = optim.SGD(model_params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd,
                     nesterov=args.nesterov)
    # optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args,optimizer)

    teacher = load_model(args)
    
    start_time = time.time()
    
    test_accs = []
    for epoch in range(args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, scheduler, args)
        if args.cuda:
            if epoch % args.update_epoch == 0:
                curr_state, label = update_score_base(trainloader,testloader, model, cls_num_list_cuda, posthoc_la = args.posthoc_la, num_test = args.num_test, accept_rate = args.accept_rate, tasks = tasks,args = args)

            if args.verbose:
                if epoch == 0:
                    maps = np.zeros((args.epochs,args.num_class))
                maps = plot_score_epoch(curr_state,label, epoch, maps, args.out)
        train_loss = train_base(args, trainloader, model, optimizer,train_criterion, epoch, weighted_trainloader, teacher, tasks) 


        test_loss, test_acc, test_cls = valid_normal(args, testloader, model, criterion, epoch, cls_num_list,  num_class=args.num_class, mode='test Valid', tasks=tasks)

        if best_acc <= test_acc:
            best_acc = test_acc
            many_best = test_cls[0]
            med_best = test_cls[1]
            few_best = test_cls[2]
            # Save models
            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'state_dict': model['model'].state_dict() if args.loss_fn == 'ncl' else model.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            # }, epoch + 1, args.out)
        test_accs.append(test_acc)

        args.logger(f'Epoch: [{epoch+1} | {args.epochs}]', level=1)
        if args.cuda:
            args.logger(f'Max_state: {int(torch.max(curr_state))}, min_state: {int(torch.min(curr_state))}', level=2)
        args.logger(f'[Train]\tLoss:\t{train_loss:.4f}', level=2)
        args.logger(f'[Test ]\tLoss:\t{test_loss:.4f}\tAcc:\t{test_acc:.4f}', level=2)
        args.logger(f'[Stats]\tMany:\t{test_cls[0]:.4f}\tMedium:\t{test_cls[1]:.4f}\tFew:\t{test_cls[2]:.4f}', level=2)
        args.logger(f'[Best ]\tAcc:\t{np.max(test_accs):.4f}\tMany:\t{100*many_best:.4f}\tMedium:\t{100*med_best:.4f}\tFew:\t{100*few_best:.4f}', level=2)
        args.logger(f'[Param]\tLR:\t{lr:.8f}', level=2)
    
    end_time = time.time()

    # Print the final results
    args.logger(f'Final performance...', level=1)
    args.logger(f'best bAcc (test):\t{np.max(test_accs)}', level=2)
    args.logger(f'best statistics:\tMany:\t{many_best}\tMed:\t{med_best}\tFew:\t{few_best}', level=2)
    args.logger(f'Training Time: {hms_string(end_time - start_time)}', level=1)
    
    if args.verbose:
        args.logger.map_save(maps)

if __name__ == '__main__':
    main()