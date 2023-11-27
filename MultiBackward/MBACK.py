from .LossFun.mgda.min_norm_solvers import MinNormSolver
from .LossFun.ChebShev import Chebyshev
from .GradFun.pcgrad import PCGrad
import time
from torch.autograd import Variable
from functools import partial

import torch
import numpy as np

class Pla:
    def __init__(self,num_tasks = 100):
        '''
        @ losses : list of loss : [loss1 , loss2]
        @ acc_s : list of losses acc on [loss1:[Many,Medium,Few] , loss2:[Many,Medium,Few]] 

        math :
        (1) $\text{similarity} = \cos(\theta) = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}$

        (2) $\dfrac{\Vert x_1 \Vert _2 \cos(\theta)}{\Vert x_2 \Vert _2 } $
        
        (1)and(2) $\beta = \dfrac{x_1 \cdot x_2 } {\Vert x_2 \Vert _2 ^ 2}$

        return $[\beta * loss1,loss2 ]$ 
        '''
        self.beta = np.ones(num_tasks)
    
    def update(self,acc_s):
        for i in range(len(acc_s)):
            if not isinstance(acc_s[i],torch.Tensor):
                acc_s[i] = torch.tensor(acc_s[i])    
            acc_s[i] += 1e-5

        for i in range(len(acc_s) - 1):
            self.beta[i] =  (acc_s[i]*acc_s[-1]).sum(0)/(acc_s[-1].norm(2)**2)

    def pla(self,losses):
        for i in range(len(self.beta)):
            assert 0 < np.abs(self.beta[i])
            assert 1e8 > np.abs(self.beta[i])
            losses[i] = self.beta[i]*losses[i]
        return losses



class MBACK():
    def __init__(self,
                 optimizer,
                 args,
                 mgda_encoder = None):
        
        self.optimizer = optimizer
        self.args = args
        
        if args.pla:
            self.Pla = Pla(len(args.tasks))
        if args.mgda:
            assert mgda_encoder != None
            self.mgda_encoder = mgda_encoder

        if args.pcg:
            self.pcg_opt = PCGrad(optimizer)
        elif args.chs:
            self.chebshev = Chebyshev()
    
    def backward(self,losses):
        if isinstance(losses,dict):
            losses = list(losses.values())
        
        if self.args.pla:
            losses = self.Pla.pla(losses) 
        
        if self.args.mgda:
            losses = self.mgda(losses,task = self.args.tasks, gn_mode = self.args.mgda_mode)
        
        if self.args.pcg:
            return self.pcg(losses)
        elif self.args.chs:
            return self.chs(losses)
        
        return self.base(losses)

    def pla_update(self,acc_s):
        if self.args.pla:
            self.Pla.update(acc_s)

    def mgda(self,losses,task=None,gn_mode = 'none'):
        def get_parameters_grad(model):
            grads = []
            for param in model.parameters():
                if param.grad is not None:
                    grads.append(Variable(param.grad.data.clone(), requires_grad=False))

            return grads
        
        loss_data = {}
        grads = {}
        
        if task==None:
            task = [i for i in range(len(losses))]
        
        for t in task:
            loss = losses[t]
            self.optimizer.zero_grad()
            loss_data[t] = loss.data
            loss.backward(retain_graph=True)
            grads[t] = get_parameters_grad(self.mgda_encoder)    

        gn = MinNormSolver.gradient_normalizers(grads, loss_data, gn_mode)
        for t in loss_data:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t].to(grads[t][gr_i].device)
        sol, _ = MinNormSolver.find_min_norm_element([grads[t] for t in task])
        for i,t in enumerate(task):
            losses[t] = losses[t]*float(sol[i])
        return losses

    def pcg(self,losses):
        self.pcg_opt.zero_grad()
        self.pcg_opt.pc_backward(losses)
        self.pcg_opt.step()
        return sum(losses)

    def chs(self,losses):
        self.chebshev.append(losses)
        self.optimizer.zero_grad()
        self.chebshev.backward()
        self.optimizer.step()
        return sum(losses)
    
    def base(self,losses):        
        loss = sum(losses)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    


    