from utils.accuracy import AverageMeter, accuracy
from scipy import optimize
from utils.common import Bar
import torch
import numpy as np
import time

def get_valid_fn(args):
    return valid_normal

def valid_normal(args, valloader, model, criterion, epoch, per_class_num, num_class=10, mode='Test Stats', trainloader = None, tasks = None):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    if args.out_cut:
        temp_tasks = tasks + ['cat_out']
    else:
        temp_tasks = tasks
    losses = {}
    top1 = {}
    top5 = {}
    all_preds = {}
    
    for t in temp_tasks:
        losses[t] = AverageMeter()
        top1[t] = AverageMeter()
        top5[t] = AverageMeter()
        all_preds[t] = np.zeros(len(valloader.dataset))

    # switch to evaluate mode
    
    model.eval()
    model.W_acc.cuda()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    all_targets = np.array(valloader.dataset.targets)
    
    with torch.no_grad():
        batch_idx = 0
        for data_tuple in valloader:
            batch_idx += 1 
            inputs = data_tuple[0]
            targets = data_tuple[1].cuda(non_blocking=True)
            indexs = data_tuple[2]
            all_targets
            # measure data loading time
            data_time.update(time.time() - end)
            
            prec1 = {}
            prec5 = {}
            pred_label = {}
            loss = {}
            # compute output
            all_outputs = model(inputs, None)
            all_outputs['cat_out'] = model.W_acc.cat_targets(all_outputs,targets,epoch)

            for index,t in enumerate(temp_tasks):     
                outputs = all_outputs[t] 
                if t in [ 'bcl' , 'cat_out' , 'gml' , 'shike']:
                    loss[t] = torch.nn.functional.cross_entropy(outputs, targets)
                else:
                    loss[t] = criterion[t](outputs, targets, epoch)
                # measure accuracy and record loss
                prec1[t], prec5[t] = accuracy(all_outputs[t], targets, topk=(1,5))
                
                # classwise prediction
                pred_label[t] = outputs.max(1)[1]
            
                losses[t].update(loss[t].item(), inputs.size(0))
                top1[t].update(prec1[t].item(), inputs.size(0))
                top5[t].update(prec5[t].item(), inputs.size(0))
                all_preds[t][indexs] = pred_label[t].cpu().numpy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # plot progress
            info_header = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | '.format(
                batch=batch_idx,
                size=len(valloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
            )

            info_body = ['Loss: ','| top1:','| top5: '] # Loss ,top1 , top5
            
            for t in tasks:
                for index,info_name in enumerate([losses,top1,top5]):
                    info_body[index] = info_body[index] + '[{} : {:.4f}]'.format(t,info_name[t].avg)

            bar.suffix = info_header + ''.join(info_body)
            bar.next()
            
        bar.finish()
        # Major, Neutral, Minor
    
        section_acc = {}
        for t in temp_tasks:
            section_acc[t] = get_section_acc(num_class,per_class_num,all_targets,all_preds[t])

        model.W_acc.update({t:all_preds[t] for t in tasks } ,all_targets)
    return (losses, top1, section_acc)

def get_section_acc(num_class,per_class_num,all_targets,all_preds):
    classwise_correct = torch.zeros(num_class)
    classwise_num = torch.zeros(num_class)
    section_acc = torch.zeros(3)
   
    pred_mask = (all_targets == all_preds).astype(np.float)
    for i in range(num_class):
        class_mask = np.where(all_targets == i)[0].reshape(-1)
        classwise_correct[i] += pred_mask[class_mask].sum()
        classwise_num[i] += len(class_mask)
    classwise_acc = (classwise_correct / classwise_num)
        
    per_class_num = torch.tensor(per_class_num)
    many_pos = torch.where(per_class_num > 100)
    med_pos = torch.where((per_class_num <= 100) & (per_class_num >=20))
    few_pos = torch.where(per_class_num < 20)

    section_acc[0] = classwise_acc[many_pos].mean()
    section_acc[1] = classwise_acc[med_pos].mean()
    if few_pos[0].numel():
        section_acc[2] = classwise_acc[few_pos].mean() 

    return section_acc

def valid_bcl(args, valloader, model, criterion, per_class_num, num_class=10, mode='Test Stats', trainloader = None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    
    classwise_correct = torch.zeros(num_class)
    classwise_num = torch.zeros(num_class)
    section_acc = torch.zeros(3)
    
    all_preds = np.zeros(len(valloader.dataset))
    with torch.no_grad():
        for batch_idx, data_tuple in enumerate(valloader):
            inputs = data_tuple[0].cuda(non_blocking=True)
            targets = data_tuple[1].cuda(non_blocking=True)
            indexs = data_tuple[2]
            
            # measure data loading time
            data_time.update(time.time() - end)
            
            # compute output
            _, outputs, _ = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1,5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            
            # classwise prediction
            pred_label = outputs.max(1)[1]
            all_preds[indexs] = pred_label.cpu().numpy()
                
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
                        
            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                          'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
        # Major, Neutral, Minor
        
        all_targets = np.array(valloader.dataset.targets)
        pred_mask = (all_targets == all_preds).astype(np.float)
        for i in range(num_class):
            class_mask = np.where(all_targets == i)[0].reshape(-1)
            classwise_correct[i] += pred_mask[class_mask].sum()
            classwise_num[i] += len(class_mask)
            
        classwise_acc = (classwise_correct / classwise_num)
        
        per_class_num = torch.tensor(per_class_num)
        many_pos = torch.where(per_class_num > 100)
        med_pos = torch.where((per_class_num <= 100) & (per_class_num >=20))
        few_pos = torch.where(per_class_num < 20)
        section_acc[0] = classwise_acc[many_pos].mean()
        section_acc[1] = classwise_acc[med_pos].mean()
        section_acc[2] = classwise_acc[few_pos].mean()

    return (losses.avg, top1.avg,  section_acc)
