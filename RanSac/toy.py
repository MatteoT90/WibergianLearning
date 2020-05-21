#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 18:56:57 2019

@author: chrisr
"""
import torch
from wiberg import seeded_LFBGS, seeded_LFBGS_H
import numpy as np


# Broadcast version of the cost function, to evaluate it at multiple points.
def broadcast_full(c_x, c_weights, c_sigma, c_centers):
    dist = (c_centers[:, None]-c_x[None, :])**2
    norm_dist = dist/(c_sigma**2)
    exp_dist = -torch.exp(-norm_dist)
    return exp_dist.mv(c_weights)


# Cost function as linear composition of various Gaussians of spread sigma centered in centers
def full(c_x, c_weights, c_sigma, c_centers):
    dist = ((c_centers-c_x)**2)
    norm_dist = dist/c_sigma**2
    exp_dist = -torch.exp(-norm_dist)
    return exp_dist.dot(c_weights)


# Cost function to evaluate the final output
def cost(estimate, points):
    # l = torch.nn.SmoothL1Loss()
    return (estimate-points.mean())**2


def main():
    """ We define the problem, by generating input data (some true values - points - and some noisy data - outliers)
    and fixing the parameters of the cost function (the weights and spreads of the Gaussians in the RBFs). We also
    provide versions of the cost function, but taking the input data from scope, not as argument """

    x = torch.arange(-100, 100, 0.1, dtype=torch.double)
    points = torch.randn(100, dtype=torch.double)*20+50
    outliers = (torch.rand(300, dtype=torch.double)-0.75)*200
    centers = torch.cat((points, outliers))
    weights = torch.ones_like(centers)
    sigma = torch.ones(1, dtype=torch.double).reshape(-1)*1
    sigma.requires_grad = True

    def bf(c_x, c_sigma):
        dist = (centers[:, None]-c_x[None, :])**2
        norm_dist = dist/(c_sigma**2)
        exp_dist = -torch.exp(-norm_dist)
        # exp_dist = -torch.max(1.0-norm_dist,torch.zeros_like(norm_dist))#torch.exp(-norm_dist)
        return exp_dist.sum(0)

    def f(c_x, c_sigma):
        dist = (centers-c_x)**2
        norm_dist = dist/(c_sigma**2)
        exp_dist = -torch.exp(-norm_dist)
        # exp_dist=-torch.max(1.0-norm_dist,torch.zeros_like(norm_dist))#torch.exp(-norm_dist)
        return exp_dist.sum(0)

    """ We define the two approaches for optimisation, implicit differentiation (RBF2) and robust 
    wibergian approach (RBF). The two functions `test' to check if numerical and analytical derivatives coincide """
    RBF = seeded_LFBGS_H.apply
    RBF2 = seeded_LFBGS.apply

    def test_sigma():
        from torch.autograd import gradcheck
        # a[:,2] = 0
        # a[:,0] = 1
        input_f = (sigma,)

        def my_func(*args):
            return RBF(bf, f, centers, *args)  # **2).sum()

        test = gradcheck(my_func, input_f, eps=1e-8, atol=1e-4)
        print(test)

    def test_full():
        from torch.autograd import gradcheck
        # a[:,2] = 0
        # a[:,0] = 1
        inputs = (weights, sigma, centers)

        def my_func(*args):
            return RBF(broadcast_full, full, centers, *args)  # **2).sum()

        test = gradcheck(my_func, inputs, eps=1e-8, atol=1e-3)
        print(test)

    """ Function to exemplify the models's performances: given sigma, we call the optimizer once to get an estimate
      of the argmin and we plot it, together with the cost function and the input data """
    def vis(sigma):
        import matplotlib.pyplot as plt
        import numpy as np

        p = points.detach().numpy()
        o = outliers.detach().numpy()
        x = torch.arange(centers.min()*1.05, centers.max()*1.05, 0.5, dtype=torch.double)
        fx = bf(x, sigma).detach().numpy()
        fmin = float(fx.min())*1.05
        plt.plot(x, fx)
        plt.scatter(o, np.ones_like(o)*fmin, color='r', alpha=0.2)
        plt.scatter(p, np.ones_like(p)*fmin, color='b', alpha=0.6)
        estimatemu = RBF(bf, f, centers, sigma)

        plt.scatter(estimatemu.detach(), f(estimatemu, sigma).detach(), color='g', marker='X', s=200)
        plt.ylabel('Learnt cost function')
        plt.xlabel('Estimate value')
        mu = p.mean()
        plt.scatter(mu, fmin, color='g', marker='X', s=200)
        plt.title('Misclassification error = %2.1f' % cost(estimatemu, points))

    vis(sigma)
    # test_sigma()
    sigma2 = sigma.clone().detach().requires_grad_()

    """ RBF and RBF2 allow us to find the minimum of the cost function, and provide gradients of the outputs w.r.t.
    the hyper-parameters (sigma). We now use standard optimization methods (stochastic gradient descent) """

    averaging = True
    if averaging:
        opt = torch.optim.ASGD([sigma], lr=0.5, t0=0, lambd=0.0005)
        opt2 = torch.optim.ASGD([sigma2], lr=0.5, t0=0, lambd=0.0005)
    else:
        opt = torch.optim.SGD([sigma], lr=5e-3, momentum=0.99)
        opt2 = torch.optim.SGD([sigma2], lr=5e-3, momentum=0.99)

    summand = 0
    summand2 = 0
    gmax = 0
    gmax2 = 0

    """ This function runs several steps of optimisation with both methods, returning for each run the average loss """
    def eval(iterations=10000):
        summand = 0
        summand2 = 0
        for i in range(iterations):
            points = torch.randn(10, dtype=torch.double) * 4 + 2
            outliers = (torch.rand(100, dtype=torch.double) - 0.75) * 80
            centers = torch.cat((points, outliers))
            #estimatemu=RBF(bf,f,centers,[sigma])
            tmp = RBF(bf, f, centers, sigma)
            loss = cost(tmp, points)
            summand += float(loss)
            tmp2 = RBF2(bf, f, centers, sigma2)
            loss2 = cost(tmp2, points)
            summand2 += float(loss2)
            val[i, 1] = sigma2
        return summand/iterations, summand2/iterations

    """ This block actually runs the hyper-parameters training, using both methods and storing parameter and 
    residual at each step """
    upto = 10000
    ggrad = np.empty([upto, 2])
    error = np.empty([upto, 2])
    val = np.empty([upto, 2])

    for i in range(upto):
        points = torch.randn(10, dtype=torch.double) * 4 + 2
        outliers = (torch.rand(100, dtype=torch.double)-0.75)*80
        centers = torch.cat((points, outliers))
        # estimatemu = RBF(bf,f,centers,[sigma])
        tmp = RBF(bf, f, centers, sigma)
        loss = cost(tmp, points)
        summand += float(loss)
        error[i, 0] = loss
        val[i, 0] = sigma

        loss.backward()
        gmax = max(sigma.grad.abs(), gmax)
        ggrad[i, 0] = sigma.grad.abs()
        opt.step()
        opt.zero_grad()

        tmp2 = RBF2(bf, f, centers, sigma2)
        loss2 = cost(tmp2, points)
        summand2 += float(loss2)
        error[i, 1] = loss2
        val[i, 1] = sigma2
        loss2.backward()
        gmax2 = max(sigma2.grad.abs(), gmax2)
        ggrad[i, 1] = sigma2.grad.abs()
        opt2.step()
        opt2.zero_grad()
        if i % 500 == 0:
            print((summand/500, float(sigma), float(gmax), 'or', summand2/500, float(sigma2), float(gmax2)))
            summand = 0
            summand2 = 0
            gmax = 0
            gmax2 = 0

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(val**2)
    plt.xlabel('Iteration')
    plt.ylabel('sigma squared')

    plt.figure()
    vis(sigma)
    plt.figure()
    vis(sigma2)
    print(eval)


if __name__ == '__main__':
    main()
