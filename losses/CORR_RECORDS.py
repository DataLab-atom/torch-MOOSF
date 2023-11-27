#https://github.com/MediaBrain-SJTU/RECORDS-LTPLL/tree/maste

import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle

class CORR_loss(torch.nn.Module):
    def __init__(self, target):
        super().__init__()
        self.confidence = target


    def forward(self,output_w,output_s,index,update_target=True):
        pred_s = F.softmax(output_s, dim=1)
        pred_w = F.softmax(output_w, dim=1)
        target = self.confidence[index, :]
        neg = (target==0).float()
        sup_loss = neg * (-torch.log(abs(1-pred_w)+1e-9)-torch.log(abs(1-pred_s)+1e-9))
        sup_loss1 = torch.sum(sup_loss) / sup_loss.size(0)
        con_loss = F.kl_div(torch.log_softmax(output_w,dim=1),target,reduction='batchmean')+F.kl_div(torch.log_softmax(output_s,dim=1),target,reduction='batchmean')
        loss = sup_loss1 + con_loss
        if update_target:
            revisedY = target.clone()
            revisedY[revisedY > 0]  = 1
            revisedY_s = revisedY * pred_s
            resisedY_w = revisedY * pred_w
            revisedY = revisedY_s * resisedY_w
            # sqr
            revisedY = torch.sqrt(revisedY)
            revisedY = (revisedY) / (revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)+1e-9)

            new_target = revisedY

            self.confidence[index,:]=new_target.detach()

        return loss
    

class CORR_loss_RECORDS(torch.nn.Module):
    def __init__(self,confidence, s = 30 , m = 0.9):
        super().__init__()
        
        self.confidence = confidence
        self.init_confidence = confidence.clone()
        self.feat_mean = None
        self.m = m

    def forward(self,output_w,output_s,feat_w,feat_s,index,model_cls = None,update_target=False):
        pred_s = F.softmax(output_s, dim=1)
        pred_w = F.softmax(output_w, dim=1)
        target = self.confidence[index, :]
        neg = (target==0).float()
        sup_loss = neg * (-torch.log(abs(1-pred_w)+1e-9)-torch.log(abs(1-pred_s)+1e-9))
        if torch.any(torch.isnan(sup_loss)):
            print("sup_loss:nan")
        sup_loss1 = torch.sum(sup_loss) / sup_loss.size(0)
        con_loss = F.kl_div(torch.log_softmax(output_w,dim=1),target,reduction='batchmean')+F.kl_div(torch.log_softmax(output_s,dim=1),target,reduction='batchmean')
        if torch.any(torch.isnan(con_loss)):
            print("con_loss:nan")
        loss = sup_loss1 + con_loss

        if self.feat_mean is None:
            self.feat_mean = (1-self.m)*((feat_w+feat_s)/2).detach().mean(0)
        else:
            self.feat_mean = self.m*self.feat_mean + (1-self.m)*((feat_w+feat_s)/2).detach().mean(0)
        
        if update_target:
            bias = model_cls(self.feat_mean.unsqueeze(0)).detach()  # just need a classfire
            bias = F.softmax(bias, dim=1)
            logits_s = output_s - torch.log(bias + 1e-9) 
            logits_w = output_w - torch.log(bias + 1e-9) 
            pred_s = F.softmax(logits_s, dim=1)
            pred_w = F.softmax(logits_w, dim=1)


            # revisedY = target.clone()
            revisedY = self.init_confidence[index,:].clone()
            revisedY[revisedY > 0]  = 1
            revisedY_s = revisedY * pred_s
            resisedY_w = revisedY * pred_w
            revisedY = revisedY_s * resisedY_w            
            revisedY = (revisedY) / (revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)+1e-9)

            # sqr
            revisedY = torch.sqrt(revisedY)
            revisedY = (revisedY) / (revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)+1e-9)

            new_target = revisedY
            # new_target = torch.where(torch.isnan(new_target), self.init_confidence[index,:].clone(), new_target)

            self.confidence[index,:]=new_target.detach()

        return loss


def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

def generate_hierarchical_and_uniform_cv_candidate_labels(dataname, train_labels, partial_rate=0.1,root = "data",ratio_hi=8):
    assert dataname == 'cifar100'

    meta_root = os.path.join(root,'cifar-100-python/meta')
    meta = unpickle(meta_root)

    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]
    label2idx = {fine_label_names[i]:i for i in range(100)}

    x = '''aquatic mammals#beaver, dolphin, otter, seal, whale
fish#aquarium fish, flatfish, ray, shark, trout
flowers#orchid, poppy, rose, sunflower, tulip
food containers#bottle, bowl, can, cup, plate
fruit and vegetables#apple, mushroom, orange, pear, sweet pepper
household electrical devices#clock, keyboard, lamp, telephone, television
household furniture#bed, chair, couch, table, wardrobe
insects#bee, beetle, butterfly, caterpillar, cockroach
large carnivores#bear, leopard, lion, tiger, wolf
large man-made outdoor things#bridge, castle, house, road, skyscraper
large natural outdoor scenes#cloud, forest, mountain, plain, sea
large omnivores and herbivores#camel, cattle, chimpanzee, elephant, kangaroo
medium-sized mammals#fox, porcupine, possum, raccoon, skunk
non-insect invertebrates#crab, lobster, snail, spider, worm
people#baby, boy, girl, man, woman
reptiles#crocodile, dinosaur, lizard, snake, turtle
small mammals#hamster, mouse, rabbit, shrew, squirrel
trees#maple_tree, oak_tree, palm_tree, pine_tree, willow_tree
vehicles 1#bicycle, bus, motorcycle, pickup truck, train
vehicles 2#lawn_mower, rocket, streetcar, tank, tractor'''

    x_split = x.split('\n')
    hierarchical = {}
    reverse_hierarchical = {}
    hierarchical_idx = [None] * 20
    # superclass to find other sub classes
    reverse_hierarchical_idx = [None] * 100
    # class to superclass
    super_classes = []
    labels_by_h = []
    for i in range(len(x_split)):
        s_split = x_split[i].split('#')
        super_classes.append(s_split[0])
        hierarchical[s_split[0]] = s_split[1].split(', ')
        for lb in s_split[1].split(', '):
            reverse_hierarchical[lb.replace(' ', '_')] = s_split[0]
            
        labels_by_h += s_split[1].split(', ')
        hierarchical_idx[i] = [label2idx[lb.replace(' ', '_')] for lb in s_split[1].split(', ')]
        for idx in hierarchical_idx[i]:
            reverse_hierarchical_idx[idx] = i

    # end generate hierarchical
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    p_1 = partial_rate
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))]=p_1*ratio_hi
    mask = np.zeros_like(transition_matrix)
    for i in range(len(transition_matrix)):
        superclass = reverse_hierarchical_idx[i]
        subclasses = hierarchical_idx[superclass]
        mask[i, subclasses] = 1

    transition_matrix *= mask
    # transition_matrix *= 5
    transition_matrix[transition_matrix==0] = p_1

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K): # for each class 
            if jj == train_labels[j]: # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0
    print("Finish Generating Candidate Label Sets!\n")
    return partialY