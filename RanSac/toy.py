#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 18:56:57 2019

@author: chrisr
"""

from wiberg import seeded_LFBGS, seeded_LFBGS_H


def broadcast_full(x,weights,sigma,centers):
    dist=(centers[:,None]-x[None,:])**2
    norm_dist=dist/(sigma**2)
    exp_dist=-torch.exp(-norm_dist)
    return exp_dist.mv(weights)
        
def full(x,weights,sigma,centers):
    dist=((centers-x)**2)
    norm_dist=dist/sigma**2
    exp_dist=-torch.exp(-norm_dist)
    return exp_dist.dot(weights)



x=torch.arange(-100,100,0.1,dtype=torch.double)
points= torch.randn(100,dtype=torch.double)*20+50
outliers=(torch.rand(300,dtype=torch.double)-0.75)*200
#outliers[0]=10
#points[0]=9.5
centers=torch.cat((points,outliers))
#centers.requires_grad=True

weights=torch.ones_like(centers)

sigma=torch.ones(1,dtype=torch.double).reshape(-1)*1
sigma.requires_grad=True

def bf(x,sigma):
    dist=(centers[:,None]-x[None,:])**2
    norm_dist=dist/(sigma**2)
    exp_dist=-torch.exp(-norm_dist)
    #exp_dist=-torch.max(1.0-norm_dist,torch.zeros_like(norm_dist))#torch.exp(-norm_dist)
    return exp_dist.sum(0)

def f(x,sigma):
    dist=(centers-x)**2
    norm_dist=dist/(sigma**2)
    exp_dist=-torch.exp(-norm_dist)
    #exp_dist=-torch.max(1.0-norm_dist,torch.zeros_like(norm_dist))#torch.exp(-norm_dist)
    #exp_dist=-torch.exp(-norm_dist)
    return exp_dist.sum(0)
        
RBF=seeded_LFBGS_H.apply
RBF2=seeded_LFBGS.apply

def cost(estimate,points):
    #l=torch.nn.SmoothL1Loss()
    return (estimate-points.mean())**2
def test_sigma():
    from torch.autograd import gradcheck
    #a[:,2]=0
    #a[:,0]=1
    input = (sigma,)
    def my_func(*args):
        return RBF(bf,f,centers,*args) #**2).sum()
        
    test = gradcheck(my_func, input, eps=1e-8, atol=1e-4)
    print(test)

def test_full():
    from torch.autograd import gradcheck
    #a[:,2]=0
    #a[:,0]=1
    inputs = (weights,sigma,centers)
    def my_func(*args):
        return RBF(broadcast_full,full,centers,*args) #**2).sum()
        
    test = gradcheck(my_func, inputs, eps=1e-8, atol=1e-3)
    print(test)

    
def vis(sigma):
    import matplotlib.pyplot as plt
    import numpy as np

    

#_, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [10, 1]})
    p=points.detach().numpy()
    o=outliers.detach().numpy()
    x=torch.arange(centers.min()*1.05,centers.max()*1.05,0.5,dtype=torch.double)
    fx=bf(x,sigma).detach().numpy()
    fmin=float(fx.min())*1.05
    plt.plot(x,fx)
    plt.scatter(o,np.ones_like(o)*fmin,color='r',alpha=0.2)
    plt.scatter(p,np.ones_like(p)*fmin,color='b',alpha=0.6)
    estimatemu=RBF(bf,f,centers,sigma)
    
    plt.scatter(estimatemu.detach(),f(estimatemu,sigma).detach(),color='g',marker='X',s=200)
    plt.ylabel('Learnt cost function')
    plt.xlabel('Estimate value')
    mu=p.mean()
    plt.scatter(mu,fmin,color='g',marker='X',s=200)
    plt.title('Misclassification error = %2.1f'%cost(estimatemu,points))
vis(sigma)
#test_sigma()
sigma2=sigma.clone().detach().requires_grad_()

averaging=True
if averaging:
    opt=torch.optim.ASGD([sigma],lr=0.5,t0=0,lambd=0.0005)
    opt2=torch.optim.ASGD([sigma2],lr=0.5,t0=0,lambd=0.0005)
else:
    opt=torch.optim.SGD([sigma],lr=5e-3,momentum=0.99)
    opt2=torch.optim.SGD([sigma2],lr=5e-3,momentum=0.99)
import numpy as np
summand=0
summand2=0
gmax=0
gmax2=0
def eval(iterations=10000):
    summand=0
    summand2=0
    for i in range(iterations):
        points= torch.randn(10,dtype=torch.double)*4+2
        outliers=(torch.rand(100,dtype=torch.double)-0.75)*80
    #outliers[0]=10
    #points[0]=9.5
        centers=torch.cat((points,outliers))
        #estimatemu=RBF(bf,f,centers,[sigma])
        tmp=RBF(bf,f,centers,sigma)
        loss=cost(tmp,points)
        summand+=float(loss)
        tmp2=RBF2(bf,f,centers,sigma2)
        loss2=cost(tmp2,points)
        summand2+=float(loss2)
        val[i,1]=sigma2
    return summand/iterations, summand2/iterations

upto=10000
ggrad=np.empty([upto,2])
error=np.empty([upto,2])
val=np.empty([upto,2])
    
for i in range(upto):
    points= torch.randn(10,dtype=torch.double)*4+2
    outliers=(torch.rand(100,dtype=torch.double)-0.75)*80
#outliers[0]=10
#points[0]=9.5
    centers=torch.cat((points,outliers))
    #estimatemu=RBF(bf,f,centers,[sigma])
    tmp=RBF(bf,f,centers,sigma)
    loss=cost(tmp,points)
    summand+=float(loss)
    error[i,0]=loss
    val[i,0]=sigma
    
    loss.backward()
    gmax=max(sigma.grad.abs(),gmax)
    ggrad[i,0]=sigma.grad.abs()
    opt.step()
    opt.zero_grad()
    
    tmp2=RBF2(bf,f,centers,sigma2)
    loss2=cost(tmp2,points)
    summand2+=float(loss2)
    error[i,1]=loss2
    val[i,1]=sigma2
    loss2.backward()
    gmax2=max(sigma2.grad.abs(),gmax2)
    ggrad[i,1]=sigma2.grad.abs()
    opt2.step()
    opt2.zero_grad()
    if i%500==0:
        print ((summand/500,float(sigma),float(gmax),'or',
                summand2/500,float(sigma2),float(gmax2)))
        summand=0
        summand2=0
        gmax=0
        gmax2=0

import matplotlib.pyplot as plt
plt.figure()
plt.plot(val**2)
plt.xlabel('Iteration')
plt.ylabel('sigma squared')
#plt.figure()
#plt.plot(error**2)
#plt.xlabel('Iteration')
#plt.ylabel('sigma squared')

#plt.figure()
#plt.plot(ggrad)
plt.figure()
vis(sigma)
plt.figure()
vis(sigma2)
print(eval)
