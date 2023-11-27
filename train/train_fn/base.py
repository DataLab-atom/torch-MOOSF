from __future__ import print_function

import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from aug.cutmix import *

import copy, time

from utils.common import Bar
from utils.accuracy import AverageMeter

from datasets.cifar100 import test_CIFAR100
import random

from train.validate import get_section_acc

def update_score_base(loader, tmp_loader, model, n_samples_per_class, posthoc_la, num_test, accept_rate, tasks,args):
    model.eval()
    
    if posthoc_la:
        dist = torch.tensor(n_samples_per_class)
        prob = dist / dist.sum()
    
    # curr_state = loader.dataset.curr_state
    # max_state = torch.max(curr_state).int() + 1
    
    with torch.no_grad():
        # pos, state = [], []
            
        # for s in range(max_state):
        #     _pos = torch.where(curr_state >= s)[0]
        #     pos_list = _pos.tolist() * (s+1) 
        #     pos +=  pos_list
        #     state += [s] * len(pos_list)
        # tmp_dataest = test_CIFAR100(pos,  state, loader.dataset)
        # tmp_loader = torch.utils.data.DataLoader(tmp_dataest, batch_size = 128,             
        #                                         shuffle=False, num_workers = 8)
        
        n = num_test
        pos, state = [], []
        for cidx in range(len(n_samples_per_class)):
            class_pos = torch.where(torch.tensor(loader.dataset.targets) == cidx)[0]
            max_state = loader.dataset.curr_state[class_pos[0]].int() 
            for s in range(max_state+1):
                _pos = random.choices(class_pos.tolist(), k = n * (s+1))
                pos += _pos 
                state += [s] * len(_pos)

        if 'cifar' in args.dataset:
            tmp_dataset = test_CIFAR100(pos, state, loader.dataset)
            tmp_loader = torch.utils.data.DataLoader(tmp_dataset, batch_size = 128, shuffle=False, num_workers=8)
        else:
            raise 'train.trainfun.base line 56 only support cifar'
        
        for batch_idx, data_tuple in enumerate(tmp_loader):
            
            data = data_tuple[0].cuda()
            label = data_tuple[1].cuda()
            idx = data_tuple[2]
            state = data_tuple[3]

            logit = model(data,None)
            logits = sum([logit[tasks[i]] for i in range(len(tasks))]) # TODO  交叉检验
            correct = (logits.max(dim=1)[1] == label).int().detach().cpu()
            loader.dataset.update_scores(correct,idx, state)
    
    # loader.dataset.update()
    for cidx in range(len(n_samples_per_class)):
        class_pos = torch.where(torch.tensor(loader.dataset.targets) == cidx)[0]
        
        correct_sum_row = torch.sum(loader.dataset.score_tmp[class_pos], dim=0)
        trial_sum_row = torch.sum(loader.dataset.num_test[class_pos], dim=0)

        ratio = correct_sum_row / trial_sum_row 
        idx = loader.dataset.curr_state[class_pos][0].int() + 1
        condition = torch.sum((ratio[:idx] > accept_rate)) == idx 
        
        # if correct_sum == trial_sum:
        # if float(correct_sum) >= float(trial_sum * 0.6):
        if condition:
            loader.dataset.curr_state[class_pos] += 1
        else:
            loader.dataset.curr_state[class_pos] -= 1

    loader.dataset.curr_state = loader.dataset.curr_state.clamp(loader.dataset.min_state, loader.dataset.max_state-1)
    loader.dataset.score_tmp *= 0
    loader.dataset.num_test *= 0


    # print(f'Max correct: {int(torch.max(correct_sum_per_class))} Max trial: {int(torch.max(trial_sum_per_class))}')
    
    # loader.dataset.update()
    
    model.train()
    
    # Debug
    curr_state = loader.dataset.curr_state
    label = loader.dataset.targets
    print(f'Max state: {int(torch.max(curr_state))} // Min state: {int(torch.min(curr_state))}')

    return curr_state, label


# def update_score_base(loader, model, n_samples_per_class, posthoc_la):
#     model.eval()
    
#     if posthoc_la:
#         dist = torch.tensor(n_samples_per_class)
#         prob = dist / dist.sum()
    
#     # curr_state = loader.dataset.curr_state
#     # max_state = torch.max(curr_state).int() + 1
    
#     with torch.no_grad():
#         # pos, state = [], []
            
#         # for s in range(max_state):
#         #     _pos = torch.where(curr_state >= s)[0]
#         #     pos_list = _pos.tolist() * (s+1) 
#         #     pos +=  pos_list
#         #     state += [s] * len(pos_list)
#         # tmp_dataest = test_CIFAR100(pos,  state, loader.dataset)
#         # tmp_loader = torch.utils.data.DataLoader(tmp_dataest, batch_size = 128,             
#         #                                         shuffle=False, num_workers = 8)
        
#         n = 10
#         pos, state = [], []
#         for cidx in range(len(n_samples_per_class)):
#             class_pos = torch.where(torch.tensor(loader.dataset.targets) == cidx)[0]
#             max_state = loader.dataset.curr_state[class_pos[0]].int() 
#             for s in range(max_state+1):
#                 _pos = random.choices(class_pos.tolist(), k = n * (s+1))
#                 pos += _pos 
#                 state += [s] * len(_pos)
 
#         tmp_dataset = test_CIFAR100(pos, state, loader.dataset)
#         tmp_loader = torch.utils.data.DataLoader(tmp_dataset, batch_size = 128, shuffle=False, num_workers=8)
        

#         for batch_idx, data_tuple in enumerate(tmp_loader):
#             data = data_tuple[0].cuda()
#             label = data_tuple[1]
#             idx = data_tuple[2]

#             logit = model(data, output_type = None).cpu()

#             if posthoc_la:
#                 logit = logit.cpu() - torch.log(prob.view(1, -1).expand(logit.shape[0],-1))

#             correct = (logit.max(dim=1)[1] == label).int().detach().cpu()
#             loader.dataset.update_scores(correct,idx)
#     print(f'Max correct: {int(torch.max(loader.dataset.score_tmp))} Max trial: {int(torch.max(loader.dataset.num_test))}')
    
#     # loader.dataset.update()
#     for cidx in range(len(n_samples_per_class)):
#         class_pos = torch.where(torch.tensor(loader.dataset.targets) == cidx)[0]
        
#         correct_sum = torch.sum(loader.dataset.score_tmp[class_pos])
#         trial_sum = torch.sum(loader.dataset.num_test[class_pos])

#         # if correct_sum == trial_sum:
#         if float(correct_sum) >= float(trial_sum * 0.8):
#             loader.dataset.curr_state[class_pos] += 1
#         else:
#             loader.dataset.curr_state[class_pos] -= 1

#     loader.dataset.curr_state = loader.dataset.curr_state.clamp(loader.dataset.min_state, loader.dataset.max_state-1)
#     loader.dataset.score_tmp *= 0
#     loader.dataset.num_test *= 0




#     model.train()
    
#     # Debug
#     curr_state = loader.dataset.curr_state
#     label = loader.dataset.targets
#     print(f'Max state: {int(torch.max(curr_state))} // Min state: {int(torch.min(curr_state))}')

#     return curr_state, label


def train_base(args, trainloader, model, optimizer, criterion, epoch, weighted_trainloader,mback, teacher = None, tasks = None ):
    
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = {t:AverageMeter() for t in tasks}
    end = time.time()
    

    bar = Bar('Training', max=len(trainloader))
    if args.cmo and 3 < epoch < (args.epochs - 3):
        inverse_iter = iter(weighted_trainloader)
    
    output = {t:[] for t in tasks}
    targets = []

    batch_idx = 0
    for data_tuple in trainloader:
        inputs_b = data_tuple[0]
        targets_b = data_tuple[1]
        indexs = data_tuple[2]
        targets += targets_b.tolist()

        saved_loss = {}

        # Measure data loading
        data_time.update(time.time() - end)
        
        if args.cmo and 3 < epoch < (args.epochs - 3):
            try:
                data_tuple_f = next(inverse_iter)
            except:
                inverse_iter = iter(weighted_trainloader)
                data_tuple_f = next(inverse_iter)

            inputs_f = data_tuple_f[0]
            targets_f = data_tuple_f[1]
            inputs_f = inputs_f[:len(inputs_b)]
            targets_f = targets_f[:len(targets_b)]
            inputs_f = inputs_f.cuda(non_blocking=True)
            targets_f = targets_f.cuda(non_blocking=True)
        
        targets_b = targets_b.cuda(non_blocking=True)
        
        r = np.random.rand(1)
        optimizer.zero_grad()

        if args.cmo and 3 < epoch < (args.epochs - 3) and r < 0.5:
            inputs_b, lam = cutmix(inputs_f, inputs_b)
        
        outputs = model(inputs_b, None,(epoch >= criterion['shike'].cornerstone) if 'shike' in tasks else False)
        
        for index,t in enumerate(tasks):
            if t == 'bcl':
                centers, logits, features = outputs['bcl']
                loss = criterion['bcl'](centers, logits, features, targets_b)
            elif t== 'gml':
                query, key, k_labels, k_idx, logits = outputs[t]
                loss = criterion[t](query, targets_b, indexs.cuda(non_blocking=True), key, k_labels, k_idx, logits)
            else:
                logits = outputs[t]
                loss = criterion[t](logits, targets_b, epoch)
            if t == 'shike':
                logits = sum(logits)/3

            saved_loss[t] = loss
            output[t] += logits.max(dim=1)[1].tolist()
            losses[t].update(loss.item(), targets_b.size(0))
        mback.backward(saved_loss)
        batch_time.update(time.time() - end)
        end = time.time()
        
        # plot
        suffixet = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | '.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    ) + 'Loss:['
        
        for t in tasks:
            suffixet += '[{} : {:.4f}] ,'.format(t,losses[t].avg)
        
        bar.suffix  =  suffixet + ']'
        bar.next()
        batch_idx +=1
    
    bar.finish()

    acc_s = [get_section_acc(args.num_class,args.cls_num_list,np.array(targets),np.array(output[t])) for t in tasks]
    mback.pla_update(acc_s)
    #model.W_acc.update(output,targets)
    return sum([losses[t].avg for t in tasks])/len(tasks)