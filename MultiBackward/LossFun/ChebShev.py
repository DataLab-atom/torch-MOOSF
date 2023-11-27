import torch
class Chebyshev:
    '''
    cbs =  Chebyshev()
    for e in range(epoches):
        Y_p = model(x)
        looses = []
        for t in tasks:
            looses.append(criterion[t](Y_p,Y_true)

        cbs.append(saved_loss,args.Chebyshev)      
        optimizer.zero_grad()
        cbs.backward()
        optimizer.step()
    '''
    def __init__(self,mu = 0.1):
        self.losses = [] # 存储历史信息
        self.item = None # 指向用于反向传播的 那一个损失
        self.mu = mu
    def append(self,loss):
        lossdata = torch.tensor([i.item() for i in loss]) 

        # if not self.losses:
        #     temp = lossdata
        #     self.losses.append(lossdata)
        #     self.item = sum(loss)
        #     return 
        
        # temp = torch.abs(lossdata - self.losses[-1])/self.losses[-1] # 求取一个在这个损失函数上的变化比
        # w = torch.exp(temp/self.mu) / (temp/self.mu).sum()
        # temp *= w
        # temp = temp*2/temp.sum()
        # self.losses.append(lossdata) # 保存一下历史信息
        max_index = torch.argmax(lossdata)
        self.item = loss[max_index]   # 构造反向传播时使用的损失
    
    def backward(self):
        self.item.backward()
        self.item = None