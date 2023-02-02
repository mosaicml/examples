from __future__ import annotations

from typing import Iterable, List, Tuple, Union
from composer.optim import DecoupledAdamW
import torch
import random
import math
class OnlinePercentileEstimate:
    def __init__(self, percentile=0.90):
        self.step = None
        self.step_up = 1.0 - percentile
        self.target_percentile = percentile
        self.percentile_estimate = None
        self.outlier_counter = 0

    def push(self, observation):
        if self.percentile_estimate is None:
            self.percentile_estimate = observation
            self.step = abs(self.percentile_estimate/2)

        if self.percentile_estimate > observation:
            self.percentile_estimate -= self.step * self.step_up
        elif self.percentile_estimate < observation:
            self.percentile_estimate += self.step * self.target_percentile
        self.step = abs(observation - self.percentile_estimate) / 2

    def query_threshold(self):
        return self.percentile_estimate



class AdaLR(DecoupledAdamW):
    def __init__(self,
                params: Union[Iterable[torch.Tensor], Iterable[dict]],
                lr: float = 1e-3,
                betas: Tuple[float, float] = (0.9, 0.95),
                eps: float = 1e-8,
                weight_decay: float = 1e-5,
                percentile_cutoff: float = 0.85,
                warmup: int = 5000,
                amsgrad: bool = False,
                lr_decay: float = 0.02,
                min_scaling: float = 0,
                max_scaling: float = float('inf')):
        super().__init__(params=params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        self.target_percentile_cutoff = percentile_cutoff
        self.warmup = max(warmup, 1)
        self.lr_decay = lr_decay
        self.min_scaling = min_scaling
        self.max_scaling = max_scaling
        
    @staticmethod
    def adam_step(params: List[torch.Tensor], grads: List[torch.Tensor], exp_avgs: List[torch.Tensor],
              exp_avg_sqs: List[torch.Tensor], max_exp_avg_sqs: List[torch.Tensor], layerwise_lr_scaling: List[torch.Tensor],
              state_steps: List[int], online_percentile_estimates: List[OnlinePercentileEstimate], *,
              beta1: float, beta2: float, lr: float, initial_lr: float, weight_decay: float,
              eps: float, warmup: int, amsgrad:bool, lr_decay: bool, min_scaling: float, max_scaling: float) -> None:
        r"""Functional API that performs AdamW algorithm computation with decoupled weight decay.

        Args:
            params (list): List of parameters to update.
            grads (list): List of parameter gradients.
            exp_avgs (list): List of average gradients.
            exp_avg_sqs (list): List of average squared gradients.
            max_exp_avg_sqs (list): List of max average squared gradients for amsgrad updates.
            state_steps (list): List of steps taken for all parameters.
            amsgrad (bool): Enables amsgrad variant of Adam.
            beta1 (float): Coefficient for computing the moving average of gradient values.
            beta2 (float): Coefficient for computing the moving average of squared gradient values.
            initial_lr (float): Initial learning rate.
            weight_decay (float): Factor for decoupled weight decay
            eps (float): Term added to the denominator to improve numerical stability.
        """

        for i, param in enumerate(params):
            ope = online_percentile_estimates[i]
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]
            layerwise_lr_scale = layerwise_lr_scaling[i]
            
            if step == 1:
                beta1 = 0
                beta2 = 0

            exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))
                
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                update = (max_exp_avg_sqs[i].sqrt()).add_(eps)
            else:
                update = (exp_avg_sq.sqrt()).add_(eps)

            effective_lr = lr * layerwise_lr_scale.item()
            # calculate the gradient-based update
            update.pow_(-1).mul_(exp_avg)

            # Perform stepweight decay
            if weight_decay != 0:
                decay_factor = (effective_lr / initial_lr) if initial_lr else 1.0
                update.add_(param, alpha=-decay_factor * weight_decay)
            update_norm = torch.linalg.vector_norm(update * effective_lr).item()
            ope.push(update_norm)
            percentile_cutoff = ope.query_threshold()
            if update_norm > percentile_cutoff and step > warmup:
                # if we are over the percentile always decrease LR
                ope.outlier_counter += 1
                layerwise_lr_scale.mul_(1-lr_decay)
            elif update_norm < percentile_cutoff and step > warmup and random.random() > ope.target_percentile:
                # if we are under the percentile, decrease it only 1-percentile pctg of the time
                # i.e. if percentile is 95%, decrease the percentile only 5% of the time
                layerwise_lr_scale.mul_(1+lr_decay)
            
            layerwise_lr_scale.clamp_(min_scaling, max_scaling)
            
            param.add_(update*-effective_lr)
    
    
 
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            online_percentile_estimates = []
            grads = []
            exp_avgs = []
            layerwise_lr_scaling = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            amsgrad = group['amsgrad']
            eps = group['eps']
            initial_lr = group['initial_lr']
            lr = group['lr']

            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state['ope'] = OnlinePercentileEstimate(self.target_percentile_cutoff)
                    state['layerwise_lr_scaling'] = torch.tensor(1.0)
                
                online_percentile_estimates.append(state['ope'])
                layerwise_lr_scaling.append(state['layerwise_lr_scaling'])
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

            self.adam_step(params_with_grad,
                       grads,
                       exp_avgs,
                       exp_avg_sqs,
                       max_exp_avg_sqs,
                       layerwise_lr_scaling,
                       state_steps,
                       online_percentile_estimates,
                       beta1=beta1,
                       beta2=beta2,
                       lr=lr,
                       initial_lr=initial_lr,
                       weight_decay=weight_decay,
                       eps=eps,
                       warmup=self.warmup,
                       amsgrad=amsgrad,
                       lr_decay=self.lr_decay,
                       min_scaling=self.min_scaling,
                       max_scaling=self.max_scaling)

        return loss


    def report_per_parameter_metrics(self, param: torch.Tensor, name: str, optimizer_metrics: dict):
        optimizer_metrics = super().report_per_parameter_metrics(param, name, optimizer_metrics)

        lr = self.param_groups[0]['lr']

        

        if param in self.state:
            param_optim_state = self.state[param]
            optimizer_metrics[f"layerwise_lr_scaling/{name}"] = param_optim_state['layerwise_lr_scaling'].item()
            optimizer_metrics[f"ope/threshold_estimate/{name}"] = param_optim_state['ope'].query_threshold()
            optimizer_metrics[f"ope/outlier_counter/{name}"] = param_optim_state['ope'].outlier_counter
            optimizer_metrics[f"ope/ope_step_size/{name}"] = param_optim_state['ope'].step
            optimizer_metrics[f"layerwise_lr_scaling/{name}"] = param_optim_state['layerwise_lr_scaling'].item()
            param_optim_state = self.state[param]
            convergence_tensor = lr * torch.ones(param_optim_state['exp_avg'].size())
            optimizer_metrics[f"expected_converged_update_norm/{name}"] = torch.linalg.vector_norm(convergence_tensor).item()

        return optimizer_metrics