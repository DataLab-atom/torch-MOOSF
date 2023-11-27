import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def soft_entropy(input, target, reduction='mean'):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = soft_entropy(input, target)
        loss.backward()
    """
    logsoftmax = nn.LogSoftmax(dim=1)
    res = -target * logsoftmax(input)
    if reduction == 'mean':
        return torch.mean(torch.sum(res, dim=1))
    elif reduction == 'sum':
        return torch.sum(torch.sum(res, dim=1))
    else:
        return
    

def mix_outputs(outputs, labels, balance=False, label_dis=None):
    
    logits_rank = outputs[0].unsqueeze(1)
    for i in range(len(outputs) - 1):
        logits_rank = torch.cat(
            (logits_rank, outputs[i+1].unsqueeze(1)), dim=1)

    max_tea, max_idx = torch.max(logits_rank, dim=1)
    # min_tea, min_idx = torch.min(logits_rank, dim=1)

    non_target_labels = torch.ones_like(labels) - labels

    avg_logits = torch.sum(logits_rank, dim=1) / len(outputs)
    non_target_logits = (-30 * labels) + avg_logits * non_target_labels

    _hardest_nt, hn_idx = torch.max(non_target_logits, dim=1)

    hardest_idx = torch.zeros_like(labels)
    hardest_idx.scatter_(1, hn_idx.data.view(-1, 1), 1)
    hardest_logit = non_target_logits * hardest_idx

    rest_nt_logits = max_tea * (1 - hardest_idx) * (1 - labels)
    reformed_nt = rest_nt_logits + hardest_logit

    preds = [F.softmax(logits, dim=1) for logits in outputs]

    reformed_non_targets = []
    for i in range(len(preds)):
        target_preds = preds[i] * labels

        target_preds = torch.sum(target_preds, dim=-1, keepdim=True)
        target_min = -30 * labels
        target_excluded_preds = F.log_softmax(
            outputs[i] * (1 - labels) + target_min, dim=1)
        reformed_non_targets.append(target_excluded_preds)

    label_dis = torch.tensor(
        label_dis, dtype=torch.float, requires_grad=False).cuda()
    label_dis = label_dis.unsqueeze(0).expand(labels.shape[0], -1)
    loss = 0.0

    if balance == True:
        for i in range(len(outputs)):
            loss += soft_entropy(outputs[i] + label_dis.log(), labels)
    else:
        for i in range(len(outputs)):
            # base ce
            loss += soft_entropy(outputs[i], labels)
            # hardest negative suppression
            loss += 10. *F.kl_div(
                    reformed_non_targets[i], F.softmax(reformed_nt, dim=1),
                    reduction='batchmean')
            # mutual distillation loss
            for j in range(len(outputs)):
                if i != j:
                    loss += F.kl_div(F.log_softmax(outputs[i], dim=1),
                                     F.softmax(outputs[j], dim=1),reduction='batchmean')
    return loss


class SHIKELoss(nn.Module):
    # https://arxiv.org/abs/2304.01279
    def __init__(self,args):
        super(SHIKELoss, self).__init__()
        self.cornerstone = 180
        self.num_class =  args.num_class
        # img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        imb_factor = 1./args.imb_ratio
        self.label_dis = np.array([int(np.floor(args.num_max * (imb_factor ** (i / (args.cls_num_list[i] + 1.0)))))
                   for i in range(args.num_class)])

    def forward(self, input, label, epoch):
        return mix_outputs(outputs=input, labels=F.one_hot(label,num_classes = self.num_class) , balance=(
                epoch >= self.cornerstone), label_dis=self.label_dis)
        