from __future__ import annotations

from typing import Iterable, List, Tuple, Union
from composer.optim import DecoupledAdamW
import torch


class EMASmoothStepAdam(DecoupledAdamW):
    def __init__(self,
                params: Union[Iterable[torch.Tensor], Iterable[dict]],
                lr: float = 1e-3,
                betas: Tuple[float, float] = (0.9, 0.95, 0.95),
                eps: float = 1e-8,
                weight_decay: float = 1e-5):
        super().__init__(params=params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            group['initial_lr'] = group['lr']
      
   
    @staticmethod
    def adam_step(params: List[torch.Tensor], grads: List[torch.Tensor], exp_avgs: List[torch.Tensor],
              exp_avg_sqs: List[torch.Tensor], state_steps: List[int], update_norm_emas: List[torch.Tensor], *,
              beta1: float, beta2: float, beta3: float, lr: float, initial_lr: float,
              weight_decay: float, eps: float,) -> None:
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
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            update_norm_ema = update_norm_emas[i]
            step = state_steps[i]

            # Perform stepweight decay
            if weight_decay != 0:
                decay_factor = (lr / initial_lr) if initial_lr else 1.0
                param.mul_(1-decay_factor * weight_decay)

            if step == 1:
                beta1 = 0
                beta2 = 0
                beta3 = 0

            exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))
            
            # calculate the gradient-based update
            update = (exp_avg_sq.sqrt()).add_(eps)
            update.pow_(-1).mul_(lr*exp_avg)

            # calculate update norm EMA
            update_norm = torch.linalg.vector_norm(update)
            update_norm_ema.mul_(beta3).add_(update_norm, alpha=(1-beta3))

            # renormalize the update
            update.div_(update_norm).mul_(update_norm_ema)

            # adjust update norm
            param.add_(update, alpha=-1)


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
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            update_norm_emas = []
            beta1, beta2, beta3 = group['betas']
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
                    state['update_norm_ema'] = torch.tensor(0.0)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                update_norm_emas.append(state['update_norm_ema'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

            self.adam_step(params_with_grad,
                       grads,
                       exp_avgs,
                       exp_avg_sqs,
                       state_steps,
                       update_norm_emas,
                       beta1=beta1,
                       beta2=beta2,
                       beta3=beta3,
                       lr=lr,
                       initial_lr=initial_lr,
                       weight_decay=weight_decay,
                       eps=eps)

        return loss


    def report_per_parameter_metrics(self, param: torch.Tensor, name: str, optimizer_metrics: dict):
        optimizer_metrics = super().report_per_parameter_metrics(param, name, optimizer_metrics)
        if param in self.state:
            param_optim_state = self.state[param]
            optimizer_metrics[f"update_norm_ema/{name}"] = param_optim_state['update_norm_ema']
            
        return optimizer_metrics