import torch
import numpy as np

class Weight_acc:
    # cat_mode = 'targets' or 'out'
    def __init__(self,num_class,tasks):
        self.tasks = tasks
        self.num_class = num_class
        self.weigh = {t:torch.ones(num_class) for t in tasks}  
        self.max_weight_task = [tasks[0] for i in range(num_class)] # 直接记录每个类别中acc最大的那一个task  
        self.weigh_save_list = {t:[] for t in tasks}  
        self.weigh_save_list['max'] = []
        self.weigh_save_list['conflict'] = np.zeros([400,2])
    
    def update(self,logits,targets):

        if not isinstance(targets,torch.Tensor):
            targets = torch.tensor(targets)
        cls_list = torch.tensor([(targets == i).sum() for i in range(self.num_class)]) 
        sum_exp = torch.zeros(self.num_class)

        for t in self.tasks:
            if not isinstance(logits[t],torch.Tensor):
                logits[t] = torch.tensor(logits[t])

            temp = []
            for target in range(self.num_class):
                target_acc = (logits[t][targets == target] == target).sum()/cls_list[target]
                temp.append(target_acc)
            
            losses_target_acc = torch.stack(temp)

            # softmax  step1
            self.weigh_save_list[t].append(losses_target_acc.cpu().numpy())
            self.weigh[t] = torch.exp(losses_target_acc) 
            sum_exp += self.weigh[t]
            
        # softmax  step2
        for t in self.tasks:
            self.weigh[t] /= sum_exp

        for i in range(self.num_class):
            
            last_max_task = self.max_weight_task[i]
            for t in self.tasks:
                if t == last_max_task:
                    continue
                
                if self.weigh[t][i] > self.weigh[last_max_task][i]:
                    last_max_task = t
                    self.max_weight_task[i] = t
        
        self.weigh_save_list['max'].append(self.max_weight_task)
    
    def cat_out(self,logits):
        return sum(self.weigh[t]*torch.softmax(logits[t],dim=1) for t in self.tasks)
    
    def cat_targets(self,logits,targets,epoch):
        # to managing conflict in  out_puts
        
        out_puts = {t:torch.argmax(logits[t],dim = 1) for t in self.tasks}
        catout  = torch.zeros_like(out_puts[self.tasks[0]])
            
        # in eques
        
        same_index = out_puts[self.tasks[0]] == out_puts[self.tasks[-1]] 
        for i in range(1,len(self.tasks) - 1 ):
            temp = out_puts[self.tasks[0]] ==  out_puts[self.tasks[i]]
            same_index =  same_index == temp

        catout[same_index] = out_puts[self.tasks[0]][same_index]
        self.weigh_save_list['conflict'][epoch][0] += same_index.sum().item()
        self.weigh_save_list['conflict'][epoch][1] += (same_index.shape[0] - self.weigh_save_list['conflict'][epoch][0])
        # in conflict      
        for i in torch.nonzero(torch.logical_not(same_index)):
            target_max_task = self.max_weight_task[targets[i]]
            catout[i] = out_puts[target_max_task][i]  

        return torch.nn.functional.one_hot(catout,num_classes=self.num_class).float()
        
    def cuda(self):
        for t in self.tasks:
            self.weigh[t] = self.weigh[t].cuda()