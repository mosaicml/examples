
from __future__ import annotations

import logging
import math
from typing import Iterable, List, Tuple, Union

import torch
from torch.optim import SGD, AdamW
from torch.optim.optimizer import required  # type: ignore
import collections
from composer.utils import dist

log = logging.getLogger(__name__)


class CosineTracker:
    def __init__(self, window_sz=25):
        self.window_sz = window_sz
        self.window = collections.deque(maxlen=window_sz)

    def insert_observation(self, obs):
        self.window.append(obs)
    
    def get_metric(self):
        if len(self.window) < self.window_sz:
            return 0
        else:
            pct_lt_zero = sum(y < 0 for y in self.window) / len(self.window)
            return pct_lt_zero


class AdaptiveCosineAdam(AdamW):
    metric_functions = {
        'l2_norm/moment':
            lambda param, optim_state, step_tensor: torch.linalg.vector_norm(optim_state['exp_avg']),
        'l2_norm/param':
            lambda param, optim_state, step_tensor: torch.linalg.vector_norm(param.data),
        'l2_norm/second_moment_sqrt':
            lambda param, optim_state, step_tensor: torch.linalg.vector_norm(optim_state['exp_avg_sq']).sqrt(),
        'l2_norm/update':
            lambda param, optim_state, step_tensor: torch.linalg.vector_norm(step_tensor),
        'l2_norm/grad':
            lambda param, optim_state, step_tensor: torch.linalg.vector_norm(param.grad),
        'cosine/update_grad':
            lambda param, optim_state, step_tensor: torch.nn.functional.cosine_similarity(
                param.grad.flatten(), step_tensor.flatten(), dim=0),
        'cosine/moment_grad':
            lambda param, optim_state, step_tensor: torch.nn.functional.cosine_similarity(
                param.grad.flatten(), optim_state['exp_avg'].flatten(), dim=0),
        # 'pct_cosine_negative/moment_grad':
        #     lambda param, optim_state, step_tensor: torch.nn.functional.cosine_similarity(
        #         param.grad.flatten(), optim_state['exp_avg'].flatten(), dim=0)
    }

    def __init__(self,
                 params: Union[Iterable[torch.Tensor], Iterable[dict]],
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.95),
                 eps: float = 1e-8,
                 weight_decay: float = 1e-5,
                 amsgrad: bool = False):
        if weight_decay >= 1e-3:
            log.warning(
                f'You are using a high value of `weight_decay={weight_decay}` for the `DecoupledAdamW` optimizer. Are you sure you want to do this? '
                f'Your model\'s weights will be multiplied by {1.0 - weight_decay} on every step!')
        super().__init__(params=params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        for group in self.param_groups:
            group['initial_lr'] = group['lr']
        self.amsgrad = amsgrad

    @staticmethod
    def adamw(params: List[torch.Tensor], grads: List[torch.Tensor], exp_avgs: List[torch.Tensor],
              exp_avg_sqs: List[torch.Tensor], max_exp_avg_sqs: List[torch.Tensor], state_steps: List[int], *,
              amsgrad: bool, beta1: float, beta2: float, lr: float, initial_lr: float, weight_decay: float,
              eps: float) -> None:
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
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            # Perform stepweight decay
            if weight_decay != 0:
                decay_factor = (lr / initial_lr) if initial_lr else 1.0
                param.mul_(1 - decay_factor * weight_decay)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step


            # update the gradient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

            # Decay the first and second moment running average coefficient
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1

            param.addcdiv_(exp_avg, denom, value=-step_size)


    def update_cosine_trackers(self, grads, exp_avgs, cosine_trackers):
        pass

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
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            cosine_trackers = []
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            if 'initial_lr' not in group:
                group['initial_lr'] = lr
            initial_lr = group['initial_lr']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['cosine_trackers'] = CosineTracker()
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                cosine_trackers.append(state['cosine_trackers'])
                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # Update the steps for each param group update
                state['step'] += 1
                # Record the step after step update
                state_steps.append(state['step'])
                
            self.update_cosine_tracker(grads, exp_avgs, cosine_trackers)
            self.adamw(params_with_grad,
                       grads,
                       exp_avgs,
                       exp_avg_sqs,
                       max_exp_avg_sqs,
                       state_steps,
                       amsgrad=amsgrad,
                       beta1=beta1,
                       beta2=beta2,
                       lr=lr,
                       initial_lr=initial_lr,
                       weight_decay=weight_decay,
                       eps=eps,
                       cosine_trackers=cosine_trackers)
            
        
        return loss

    def dist_reduce_metrics(self, optimizer_metrics):
        for metric in optimizer_metrics:
            if metric.startswith('l2_norm'):
                reduced = optimizer_metrics[metric]
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation='SUM')

                optimizer_metrics[metric] = math.sqrt(reduced)
            elif metric.startswith('cosine'):
                reduced = optimizer_metrics[metric]
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation='SUM')

                _, vectors, layer = tuple(metric.split('/'))

                A, B = tuple(vectors.split('_'))

                A_reduced_norm = optimizer_metrics[f'l2_norm/{A}/{layer}']
                B_reduced_norm = optimizer_metrics[f'l2_norm/{B}/{layer}']
                optimizer_metrics[metric] = reduced / (A_reduced_norm * B_reduced_norm)
            else:
                reduced = optimizer_metrics[metric]
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation='SUM')
                optimizer_metrics[metric] = reduced / dist.get_world_size()

        return optimizer_metrics

    def pre_reduce_metrics(self, optimizer_metrics):
        """Preprocess metrics to reduce across ranks correctly."""
        # Sort L2 norms first so they are squared before other metrics, which depend on squared values
        metrics = optimizer_metrics.keys()
        metrics = sorted(metrics, key=lambda metric: 0 if 'l2_norm' in metric else 1)
        for metric in metrics:
            if metric.startswith('l2_norm'):
                # L2 norms need to be squared, before they are reduced via summation
                optimizer_metrics[metric] = optimizer_metrics[metric]**2
            elif metric.startswith('cosine'):
                _, vectors, layer = tuple(metric.split('/'))

                A, B = tuple(vectors.split('_'))

                # L2 norm would've been squared in previous branch
                A_rank_subset_norm = math.sqrt(optimizer_metrics[f'l2_norm/{A}/{layer}'])
                B_rank_subset_norm = math.sqrt(optimizer_metrics[f'l2_norm/{B}/{layer}'])

                optimizer_metrics[metric] *= A_rank_subset_norm * B_rank_subset_norm

        return optimizer_metrics

    def report_per_parameter_metrics(self, param: torch.Tensor, name: str, optimizer_metrics: dict):
        lr = self.param_groups[0]['lr']
        eps = self.param_groups[0]['eps']
        weight_decay = self.param_groups[0]['weight_decay']
        initial_lr = self.param_groups[0]['initial_lr']

        beta1, beta2 = self.param_groups[0]['betas']
        if param in self.state:
            param_optim_state = self.state[param]
            step = param_optim_state['step']
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            denom = (param_optim_state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            step_size = lr / bias_correction1
            step_tensor = step_size * param_optim_state['exp_avg'].div(denom)
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            step_tensor.add_(param, alpha=-weight_decay * decay_factor)
            for metric in self.metric_functions:
                optimizer_metrics[f'{metric}/{name}'] = self.metric_functions[metric](param, param_optim_state,
                                                                                      step_tensor)

        return optimizer_metrics


class MaskAdam(AdamW):
    metric_functions = {
        'l2_norm/moment':
            lambda param, optim_state, step_tensor: torch.linalg.vector_norm(optim_state['exp_avg']),
        'l2_norm/param':
            lambda param, optim_state, step_tensor: torch.linalg.vector_norm(param.data),
        'l2_norm/second_moment_sqrt':
            lambda param, optim_state, step_tensor: torch.linalg.vector_norm(optim_state['exp_avg_sq']).sqrt(),
        'l2_norm/update':
            lambda param, optim_state, step_tensor: torch.linalg.vector_norm(step_tensor),
        'l2_norm/grad':
            lambda param, optim_state, step_tensor: torch.linalg.vector_norm(param.grad),
        'cosine/update_grad':
            lambda param, optim_state, step_tensor: torch.nn.functional.cosine_similarity(
                param.grad.flatten(), step_tensor.flatten(), dim=0),
        'cosine/moment_grad':
            lambda param, optim_state, step_tensor: torch.nn.functional.cosine_similarity(
                param.grad.flatten(), optim_state['exp_avg'].flatten(), dim=0)
    }

    def __init__(self,
                 params: Union[Iterable[torch.Tensor], Iterable[dict]],
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.95),
                 eps: float = 1e-8,
                 weight_decay: float = 1e-5,
                 amsgrad: bool = False):
        if weight_decay >= 1e-3:
            log.warning(
                f'You are using a high value of `weight_decay={weight_decay}` for the `DecoupledAdamW` optimizer. Are you sure you want to do this? '
                f'Your model\'s weights will be multiplied by {1.0 - weight_decay} on every step!')
        super().__init__(params=params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        for group in self.param_groups:
            group['initial_lr'] = group['lr']
        self.amsgrad = amsgrad

    @staticmethod
    def adamw(params: List[torch.Tensor], grads: List[torch.Tensor], exp_avgs: List[torch.Tensor],
              exp_avg_sqs: List[torch.Tensor], max_exp_avg_sqs: List[torch.Tensor], state_steps: List[int], *,
              amsgrad: bool, beta1: float, beta2: float, lr: float, initial_lr: float, weight_decay: float,
              eps: float) -> None:
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
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            # Perform stepweight decay
            if weight_decay != 0:
                decay_factor = (lr / initial_lr) if initial_lr else 1.0
                param.mul_(1 - decay_factor * weight_decay)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step


            # mask out any params from the moment that point in the opposite direction of
            # the grad
            exp_avg.mul_(exp_avg.sign().add_(grad.sign()).sign())

            # update the gradient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

            # Decay the first and second moment running average coefficient
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1

            param.addcdiv_(exp_avg, denom, value=-step_size)


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
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            if 'initial_lr' not in group:
                group['initial_lr'] = lr
            initial_lr = group['initial_lr']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # Update the steps for each param group update
                state['step'] += 1
                # Record the step after step update
                state_steps.append(state['step'])

            self.adamw(params_with_grad,
                       grads,
                       exp_avgs,
                       exp_avg_sqs,
                       max_exp_avg_sqs,
                       state_steps,
                       amsgrad=amsgrad,
                       beta1=beta1,
                       beta2=beta2,
                       lr=lr,
                       initial_lr=initial_lr,
                       weight_decay=weight_decay,
                       eps=eps)

        return loss

    def dist_reduce_metrics(self, optimizer_metrics):
        for metric in optimizer_metrics:
            if metric.startswith('l2_norm'):
                reduced = optimizer_metrics[metric]
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation='SUM')

                optimizer_metrics[metric] = math.sqrt(reduced)
            elif metric.startswith('cosine'):
                reduced = optimizer_metrics[metric]
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation='SUM')

                _, vectors, layer = tuple(metric.split('/'))

                A, B = tuple(vectors.split('_'))

                A_reduced_norm = optimizer_metrics[f'l2_norm/{A}/{layer}']
                B_reduced_norm = optimizer_metrics[f'l2_norm/{B}/{layer}']
                optimizer_metrics[metric] = reduced / (A_reduced_norm * B_reduced_norm)
            else:
                reduced = optimizer_metrics[metric]
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation='SUM')
                optimizer_metrics[metric] = reduced / dist.get_world_size()

        return optimizer_metrics

    def pre_reduce_metrics(self, optimizer_metrics):
        """Preprocess metrics to reduce across ranks correctly."""
        # Sort L2 norms first so they are squared before other metrics, which depend on squared values
        metrics = optimizer_metrics.keys()
        metrics = sorted(metrics, key=lambda metric: 0 if 'l2_norm' in metric else 1)
        for metric in metrics:
            if metric.startswith('l2_norm'):
                # L2 norms need to be squared, before they are reduced via summation
                optimizer_metrics[metric] = optimizer_metrics[metric]**2
            elif metric.startswith('cosine'):
                _, vectors, layer = tuple(metric.split('/'))

                A, B = tuple(vectors.split('_'))

                # L2 norm would've been squared in previous branch
                A_rank_subset_norm = math.sqrt(optimizer_metrics[f'l2_norm/{A}/{layer}'])
                B_rank_subset_norm = math.sqrt(optimizer_metrics[f'l2_norm/{B}/{layer}'])

                optimizer_metrics[metric] *= A_rank_subset_norm * B_rank_subset_norm

        return optimizer_metrics

    def report_per_parameter_metrics(self, param: torch.Tensor, name: str, optimizer_metrics: dict):
        lr = self.param_groups[0]['lr']
        eps = self.param_groups[0]['eps']
        weight_decay = self.param_groups[0]['weight_decay']
        initial_lr = self.param_groups[0]['initial_lr']

        beta1, beta2 = self.param_groups[0]['betas']
        if param in self.state:
            param_optim_state = self.state[param]
            step = param_optim_state['step']
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            denom = (param_optim_state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            step_size = lr / bias_correction1
            step_tensor = step_size * param_optim_state['exp_avg'].div(denom)
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            step_tensor.add_(param, alpha=-weight_decay * decay_factor)
            for metric in self.metric_functions:
                optimizer_metrics[f'{metric}/{name}'] = self.metric_functions[metric](param, param_optim_state,
                                                                                      step_tensor)

        return optimizer_metrics
