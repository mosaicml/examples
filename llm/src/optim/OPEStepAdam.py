from __future__ import annotations

from typing import Iterable, List, Tuple, Union
from composer.optim import DecoupledAdamW
import torch

class OnlinePercentileEstimate:
    def __init__(self, percentile=0.85, step=0.5, initial_estimate=0.2):
        self.step = step
        self.step_up = 1.0 - percentile
        self.step_down = percentile
        self.percentile_estimate = initial_estimate
        self.skip_counter = 0

    def push(self, observation):
        if self.percentile_estimate is None:
            self.percentile_estimate = observation
            self.step = max(abs(self.percentile_estimate/2), self.step)
            return

        if self.percentile_estimate > observation:
            self.percentile_estimate -= self.step * self.step_up
        elif self.percentile_estimate < observation:
            self.percentile_estimate += self.step * self.step_down
        if abs(observation - self.percentile_estimate) < self.step:
            self.step /= 2.0

    def query_threshold(self):
        return self.percentile_estimate



class OPEStepAdam(DecoupledAdamW):
    def __init__(self,
                params: Union[Iterable[torch.Tensor], Iterable[dict]],
                lr: float = 1e-3,
                betas: Tuple[float, float] = (0.9, 0.95),
                eps: float = 1e-8,
                weight_decay: float = 1e-5,
                percentile_cutoff: float = 0.85,
                warmup: int = 5000,
                amsgrad: bool = False):
        super().__init__(params=params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        for group in self.param_groups:
            group['initial_lr'] = group['lr']
        self.target_percentile_cutoff = percentile_cutoff
        self.warmup = max(warmup, 1)
        

    
    @staticmethod
    def fame_adam(params: List[torch.Tensor], grads: List[torch.Tensor], exp_avgs: List[torch.Tensor],
              exp_avg_sqs: List[torch.Tensor], max_exp_avg_sqs: List[torch.Tensor], state_steps: List[int], 
              online_percentile_estimates: List[OnlinePercentileEstimate], *,
              beta1: float, beta2: float, lr: float, initial_lr: float, weight_decay: float,
              eps: float, warmup: int, amsgrad:bool) -> None:
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
            lr (float): Learning rate.
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
          
            # Perform stepweight decay
            if weight_decay != 0:
                decay_factor = (lr / initial_lr) if initial_lr else 1.0
                param.mul_(1 - decay_factor * weight_decay)

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

            update.power_(-1).mul_(-lr*exp_avg)
            update_norm = torch.linalg.vector_norm(update)
            percentile_cutoff = ope.query_threshold()
            ope.push(update_norm)
            if update_norm > percentile_cutoff and step > warmup:
                ope.skip_counter += 1
                continue
            else:
                param.add_(update)
          

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
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            amsgrad = group['amsgrad']
            eps = group['eps']
            lr = group['lr']
            initial_lr = group['initial_lr']
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
                
                online_percentile_estimates.append(state['ope'])
                    
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])
            self.fame_adam(params_with_grad,
                       grads,
                       exp_avgs,
                       exp_avg_sqs,
                       max_exp_avg_sqs,
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
                       lr_slowdown=self.lr_slowdown,
                       skip_outliers=self.skip_outliers)

        return loss


    def report_per_parameter_metrics(self, param: torch.Tensor, name: str, optimizer_metrics: dict):
        optimizer_metrics = super().report_per_parameter_metrics(param, name, optimizer_metrics)
        if param in self.state:
            param_optim_state = self.state[param]
            optimizer_metrics[f"outlier/threshold_estimate/{name}"] = param_optim_state['ope'].query_threshold()
            optimizer_metrics[f"outlier/skip_counter/{name}"] = param_optim_state['ope'].skip_counter

        return optimizer_metrics