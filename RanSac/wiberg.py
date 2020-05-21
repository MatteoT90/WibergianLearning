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
    def forward(self, f, estimate, *parameters, robust=0.01, FinalNewton=False):
        """
        Forward pass of the Wieberg optimiser: it updates the initial guess of the argmin of the cost function f by
        taking a newton step (if  needed) of the cost function f and it pre-computes elements for the backward pass (
        for exsample, the Hessian). FinalNewton = True and robust = 0 makes analytic and numeric derivatives coincide.
        :param self: as usual, it gives the function access to the other class methods
        :param f: cost function whose hyper-parameters are to be optimised
        :param estimate: current guess of the argmin of f
        :param parameters: list of parameters of f to be optimised
        :param robust: small value added to the diagonal of the Hessian to make it invertible
        :param FinalNewton: if True, we perform a final step of Newton optimisation on the estimate
        :return: estimate (updated, if FinalNewton = True)
        """

        with torch.enable_grad():    
            estimate = estimate.clone().detach().requires_grad_()

        # If FinalNewton, we compute gradient g and hessian H of f w.r.t. x, and update x-> x-D, D argmin || g - HD ||^2

        if FinalNewton:
            with torch.enable_grad():    
                x_1grad, = grad(f(estimate, *parameters), estimate, create_graph=True)
                H = list()
                for i in range(x_1grad.numel()):
                    temp, = grad(x_1grad[i], estimate)
                    H.append(temp)
                    
            Hessian = torch.stack(H, dim=0)
            Hessian += robust*torch.eye(Hessian.shape[0], dtype=Hessian.dtype)
            try:
                update, _ = torch.lstsq(x_1grad.reshape(Hessian.shape[0], 1), Hessian)
            except:
                print("Error: inv H failed.")
                print("x is:", estimate)
                print("H is:", Hessian)
                
                print("Cost is:", f(estimate, *parameters))
                update = torch.zeros(len(H))
                assert False
            estimate -= update.reshape(estimate.shape)
            # estimate = estimate.clone().detach().requires_grad_(True)

        # Compute as above the hessian, and d nabla/ d parameter
        with torch.enable_grad():
            nabla, = grad(f(estimate, *parameters), [estimate], create_graph=True)
            H = list()
            dndp = list()
            for i in range(nabla.numel()):
                eg, *pg = grad(nabla[i], [estimate, *parameters])
                H.append(eg)
                dndp.append(pg)

        # To make the Hessian invertible, H -> H + robust * I
        Hessian = torch.stack(H, dim=0)
        if robust != 0:
            Hessian += robust * torch.eye(estimate.shape[0], dtype=estimate.dtype)
               
        # dndp is now a list of lists in the wrong order,
        # we need to transpose and stack e.g. the nth element of every
        # sublist into a single tensor for each parameter
        nabla_P = list(map(lambda x: torch.stack(x, dim=0), zip(*dndp)))

        # The derivatives and Hessian are stored for later use
        self.save_for_backward(Hessian, *nabla_P)
        
        return estimate
    
    @staticmethod
    def backward(self, grad_output):
        """
        :param self: as usual, it gives the function access to the other class methods
        :param grad_output: the gradient of the output of this function w.r.t. the hyper-parameters of f
        :return: gradients w.r.t. the hyper-parameters of f to back-propagate trough the model
        """
        # Load Hessian and derivative of the gradient w.r.t. the hyper-parameters of f
        Hessian, *nabla_P = self.saved_tensors
        
        compress, _ = torch.lstsq(-grad_output, Hessian)
        compress = compress.reshape(1, Hessian.shape[0])
        det_grad = []
        for p in nabla_P:
            tmp = compress.mm(p.reshape(Hessian.shape[0], -1)).reshape(p.shape[1:])
            det_grad.append(tmp)
        if not(torch.all(torch.isfinite(tmp))):
            print(Hessian)
            assert(torch.all(torch.isfinite(tmp)))
        
        return (None, None, *det_grad)


class LFBGS(Wiberg):
    @staticmethod
    def forward(self, f, estimate, *parameters, robust=0.01, lr=1, iterations=100, FinalNewton=False):
        """
        We provide a better estimate of the argmin of f by using the quasi-newton optimisation algorithm L-BFGS.
        :param self: as usual, it gives the function access to the other class methods
        :param f: cost function whose hyper-parameters are to be optimised
        :param estimate: current guess of the argmin of f
        :param parameters: list of parameters of f to be optimised
        :param robust: small value added to the diagonal of the Hessian to make it invertible
        :param lr: learning rate of the optimizer
        :param iterations: maximum number of iterations of the optimization process
        :param FinalNewton: if True, we perform a final step of Newton optimisation on the estimate
        :return: Estimated argmin of the cost function f
        """

        """ Using directly the trained parameters, the backwards pass triggers an endless, recursive call of backwards;
        to solve this, the variables are cloned and set to not require gradients. The require_grad option is then 
        switched on in the backwards call. As similar issues also arise in the forward pass, where shared gradients
        pollute the behaviour of LBFGS optimisation, we can directly copy the input vectors and disable gradients """

        # e = estimate.clone()
        with torch.enable_grad():
            opt = torch.optim.LBFGS([estimate, ], lr=lr, max_iter=iterations)
            pclone = list(map(lambda x: x.clone().detach().requires_grad_(), parameters))

            def close():
                opt.zero_grad()
                loss = f(estimate, *pclone)
                loss.backward()
                return loss  
    
            opt.step(close)
            
        estimate = estimate.clone().detach().requires_grad_(True)
        estimate = Wiberg.forward(self, f, estimate, *parameters, robust=robust, FinalNewton=FinalNewton)
        
        assert(torch.all(torch.isfinite(estimate)))
        return estimate
    
    @staticmethod
    def backward(self, grad_output):
        return Wiberg.backward(self, grad_output)


class seeded_LFBGS(LFBGS):
    """ Class to start the optimization using the robuts, trust-region approach with no
    final newton step"""
    @staticmethod
    def forward(self, broadcast_f, f, seeds, *parameters, robust=0.1, lr=1, iterations=10, FinalNewton=False):
        """
        :param self: as usual, it gives the function access to the other class methods
        :param broadcast_f:  broadcast version of the cost function, that evaluates f at multiple points
        :param f: cost function whose hyper-parameters are to be optimised
        :param seeds: sample points in the domain of f, used to find the initial guess of argmin(f)
        :param parameters: list of parameters of f to be optimised
        :param robust: small value added to the diagonal of the Hessian to make it invertible
        :param lr: learning rate of the optimizer
        :param iterations: maximum number of iterations of the optimization process
        :param FinalNewton: if True, we perform a final step of Newton optimisation on the estimate
        :return: Estimated argmin of the cost function f
        """

        costs = broadcast_f(seeds, *parameters)
        estimate = seeds[costs.argmin()]
        if estimate.dim() == 0:
            estimate = estimate.reshape(-1)
        estimate = LFBGS.forward(self, f, estimate, *parameters, robust=robust, iterations=iterations, FinalNewton=True)
        return estimate

    @staticmethod
    def backward(self, grad_output):
        return (None, *LFBGS.backward(self, grad_output))



class seeded_LFBGS_H(LFBGS):

    """ Class to start the optimization of the parameters of f using the unstable implicit differentiation
     approach, with a final newton step"""
    @staticmethod
    def forward(self, broadcast_f, f, seeds, *parameters, robust=0.0, lr=1, iterations=10, FinalNewton=True):
        """
        :param self: as usual, it gives the function access to the other class methods
        :param broadcast_f: broadcast version of the cost function, that evaluates f at multiple points
        :param f: cost function whose hyper-parameters are to be optimised
        :param seeds: sample points in the domain of f, used to find the initial guess of argmin(f)
        :param parameters: list of parameters of f to be optimised
        :param robust: small value added to the diagonal of the Hessian to make it invertible
        :param lr: learning rate of the optimizer
        :param iterations: maximum number of iterations of the optimization process
        :param FinalNewton: if True, we perform a final step of Newton optimisation on the estimate
        :return: Estimated argmin of the cost function f
        """

        costs = broadcast_f(seeds, *parameters)
        estimate = seeds[costs.argmin()]
        if estimate.dim() == 0:
            estimate = estimate.reshape(-1)
        estimate = LFBGS.forward(self, f, estimate, *parameters, robust=False, iterations=iterations, FinalNewton=True)
        return estimate

    @staticmethod
    def backward(self, grad_output):
        return (None, *LFBGS.backward(self, grad_output))
