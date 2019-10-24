#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 18:16:09 2019

@author: chrisr
"""
import torch
from torch.autograd import grad

class Wiberg(torch.autograd.Function):
    @staticmethod
    def forward(self, f,estimate,*parameters,robust=0.01,FinalNewton=False):
        #Optionally, perform a final step of Newton optimisation on the estimate
        # This and robust=0 are typically needed for the analytic and numeric derivatives to 
        # coincide
        #Technically, all forward should do is return the estimate,
        # but we precompute useful things here
        #Retaining the Hessian and dnabla/dparameters for backwards
        #parameters=[parameters]
        with torch.enable_grad():    
            estimate=estimate.clone().detach().requires_grad_()
        if (FinalNewton):
            with torch.enable_grad():    
                x_1grad, = grad(f(estimate,*parameters),
                            estimate, create_graph=True)
                
                H=list()
                for i in range(x_1grad.numel()):
                    temp,= grad(x_1grad[i], estimate)
                    H.append(temp)
                    
            Hessian = torch.stack(H, dim=0) 
            Hessian+=robust*torch.eye(Hessian.shape[0],dtype=Hessian.dtype)
            try:
                update,_=torch.lstsq(x_1grad.reshape(Hessian.shape[0],1),Hessian)
            except:
                print ("Error: inv H failed.")
                print("x is:",estimate)
                print ("H is:",Hessian)
                
                print ("Cost is:",f(estimate,*parameters))
                update=torch.zeros(len(H))
                assert(False)
            estimate-=update.reshape(estimate.shape)
            #estimate=estimate.clone().detach().requires_grad_(True)
        ##Compute hessian and dNablda/dparameter
        with torch.enable_grad():
            nabla, = grad(f(estimate,*parameters),
                            [estimate], create_graph=True)
            H=list()
            dndp=list() 
            for i in range(nabla.numel()):
                eg,*pg= grad(nabla[i], [estimate,*parameters])
                H.append(eg)
                dndp.append(pg)

        Hessian=torch.stack(H,dim=0)
        if robust!=0:
            Hessian+=robust* torch.eye(estimate.shape[0],dtype=estimate.dtype)
               
        #dndp is now a list of lists in the wrong order
        #we need to transpose and stack e.g. the nth element of every 
        #sublist into a single tensor for each parameter
        nabla_P=list(map(lambda x: torch.stack(x,dim=0), zip(*dndp)))
        
        
        self.save_for_backward(Hessian, *nabla_P)
        
        return estimate
    
    @staticmethod
    def backward(self,grad_output):
        Hessian, *nabla_P=self.saved_tensors
        
        compress,_=torch.lstsq(-grad_output,Hessian)
        compress=compress.reshape(1,Hessian.shape[0])
        det_grad=[]
        for p in nabla_P:
            tmp=compress.mm(p.reshape(Hessian.shape[0],-1)).reshape(p.shape[1:])
            det_grad.append(tmp)
        if not(torch.all(torch.isfinite(tmp))):
            print (Hessian)
            assert((torch.all(torch.isfinite(tmp))))
        
        return (None,None,*det_grad)


class LFBGS(Wiberg):
    @staticmethod
    def forward(self, f,estimate,*parameters,robust=0.01,
                lr=1,iterations=100,FinalNewton=False):
        #     FinalNewton=True Controls if their is a final iteration of newton step
                         #in the forward pass
        #If we use A and det directly in backwards
        #an endless recusive call of backwards is triggered
        #We create copied variables, these must have requires_grad=False
        #for reasons... which are set to true in backwards
        #Fortunately (ish) similar issues arise in forwards with shared 
        # gradients  poluting the behaviour of LBFGS
        # so we upfront copy the input vectors and disable
        # gradients
        #e=estimate.clone()
        with torch.enable_grad():
            opt=torch.optim.LBFGS([estimate,],lr=lr,max_iter=iterations)
            pclone=list(map(lambda x: x.clone().detach().requires_grad_(),parameters))
            def close():
                opt.zero_grad()
                loss=f(estimate,*pclone)
                loss.backward()
                return loss  
    
            opt.step(close)
            
        estimate=estimate.clone().detach().requires_grad_(True)
        estimate=Wiberg.forward(self,f, estimate, *parameters, 
                                robust=robust, FinalNewton=FinalNewton)
        
        assert((torch.all(torch.isfinite(estimate))))
       #
        return estimate
    
    @staticmethod
    def backward(self,grad_output):
        return Wiberg.backward(self,grad_output)
    
class seeded_LFBGS(LFBGS):
    @staticmethod
    def forward(self,broadcast_f,f, seeds, *parameters,robust=0.1,
                lr=1,iterations=10,FinalNewton=False):
        
        costs=broadcast_f(seeds,*parameters)
        estimate=seeds[costs.argmin()]
        if estimate.dim()==0:
            estimate=estimate.reshape(-1)
        estimate=LFBGS.forward(self,f,estimate,*parameters,robust=robust,
                               iterations=iterations,FinalNewton=True)
        return estimate
    @staticmethod
    def backward(self,grad_output):
        return (None,*LFBGS.backward(self,grad_output))


class seeded_LFBGS_H(LFBGS):
    @staticmethod
    def forward(self,broadcast_f,f, seeds, *parameters,robust=0.0,
                lr=1,iterations=10,FinalNewton=True):
        
        costs=broadcast_f(seeds,*parameters)
        estimate=seeds[costs.argmin()]
        if estimate.dim()==0:
            estimate=estimate.reshape(-1)
        estimate=LFBGS.forward(self,f,estimate,*parameters,robust=False,
                               iterations=iterations,FinalNewton=True)
        return estimate
    @staticmethod
    def backward(self,grad_output):
        return (None,*LFBGS.backward(self,grad_output))
