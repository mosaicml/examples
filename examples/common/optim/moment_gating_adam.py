
# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Optimizers with weight decay decoupled from the learning rate.

These optimizers are based off of `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`_, which
proposes this decoupling. In general, it is recommended to use these optimizers over their native PyTorch equivalents.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, List, Tuple, Union

import torch
from torch.optim import SGD, AdamW
from torch.optim.optimizer import required  # type: ignore

from composer.utils import dist
import collections

log = logging.getLogger(__name__)

__all__ = ['MomentGatingAdam']



class MomentGatingAdam(AdamW):
  
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
        'percentage_nonzero/moment':
            lambda param, optim_state, step_tensor:
            (torch.count_nonzero(optim_state['exp_avg']) / optim_state['exp_avg_sq'].nelement())
            if optim_state['exp_avg'].flatten().shape[0] > 0 else 0,
        'percentage_nonzero/grad':
            lambda param, optim_state, step_tensor: (torch.count_nonzero(param.grad) / param.grad.nelement())
            if param.grad.nelement() > 0 else 0,
        'percentage_nonzero/second_moment':
            lambda param, optim_state, step_tensor:
            (torch.count_nonzero(optim_state['exp_avg_sq']) / optim_state['exp_avg_sq'].nelement())
            if optim_state['exp_avg_sq'].flatten().shape[0] > 0 else 0,
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
              amsgrad: bool, beta1: float, beta2: float, lr: float, initial_lr: float, weight_decay: float, eps: float,
              layerwise_gating) -> None:
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

            # Decay the first and second moment running average coefficient

            update = exp_avg.mul(beta1).mul_(layerwise_gating[param]).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                update.div_(max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                update.div_(exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1

            param.add_(update, alpha=-step_size)
            
            # update 1st moment
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)


    def reset_state(self):
        for group in self.param_groups:
            amsgrad = group['amsgrad']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def reset_param_state(self, param):
        if param not in self.state:
            return
        state = self.state[param]
        state['step'] = 0
        # Exponential moving average of gradient values
        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
        # Exponential moving average of squared gradient values
        state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
        if self.amsgrad:
            # Maintains max of all exp. moving avg. of sq. grad. values
            state['max_exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

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
            layerwise_gating = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
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

                    state['mva_tracker'] = MVATracker()
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                layerwise_gating.append(state['mva_tracker'].get_gating())
                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
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
                       eps=eps,
                       layerwise_gating=layerwise_gating)
        self.update_gradient_tracker()
        return loss

    def update_gradient_tracker(self):
        layer_grads = {}
        counter = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                layer_grads[f'l2_norm/grad/{counter}'] = torch.linalg.vector_norm(p.grad)
                counter += 1
        
        layer_grads = self.pre_reduce_metrics(layer_grads)
        layer_grads = self.dist_reduce_metrics(layer_grads)

        counter = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                self.state[p]['mva_tracker'].insert_observation(
                    layer_grads[f'l2_norm/grad/{counter}']
                )
                counter += 1

    def dist_reduce_metrics(self, optimizer_metrics):
        for metric in optimizer_metrics:
            if metric.startswith('layerwise_lr_scaling'):
                continue
            elif metric.startswith('l2_norm'):
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

                # it would've already been squared, so let's undo that
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
        # some of the metrics need to be modified before being reduced in order for the
        # reduction to work properly

        for metric in optimizer_metrics:
            if metric.startswith('layerwise_lr_scaling'):
                continue
            elif metric.startswith('l2_norm'):
                # l2 norms need to be squared, before they are reduced via summation
                optimizer_metrics[metric] = optimizer_metrics[metric]**2
            elif metric.startswith('cosine'):
                _, vectors, layer = tuple(metric.split('/'))

                A, B = tuple(vectors.split('_'))

                # it would've already been squared, so let's undo that
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
            local_lr = lr * self.get_scaling(param)
            step = param_optim_state['step']
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            denom = (param_optim_state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            step_size = local_lr / bias_correction1
            step_tensor = step_size * param_optim_state['exp_avg'].div(denom)
            decay_factor = (local_lr / initial_lr) if initial_lr else 1.0
            step_tensor.add_(param, alpha=-weight_decay * decay_factor)
            for metric in self.metric_functions:
                optimizer_metrics[f'{metric}/{name}'] = self.metric_functions[metric](param, param_optim_state,
                                                                                      step_tensor)

            optimizer_metrics[f'layerwise_lr_scaling/{name}'] = self.get_scaling(param)

        return optimizer_metrics



class MVATracker:

    def __init__(self,
                 window_moving_average=25,
                 increase_lookback=500):

        self.window_moving_average = window_moving_average
        self.increase_lookback = increase_lookback
        self.fast_moving_average = collections.deque(maxlen=window_moving_average)
        self.intermediate_data_queue = collections.deque(maxlen=increase_lookback - window_moving_average)
        self.slow_moving_average = collections.deque(maxlen=increase_lookback)

    def insert_observation(self, obs):
        if len(self.fast_moving_average) >= self.fast_moving_average.maxlen:
            # move the oldest obs out of the fast moving average into the
            # intermediate data queue
            fast_obs = self.fast_moving_average.popleft()

            if len(self.intermediate_data_queue) >= self.intermediate_data_queue.maxlen:
                # move data from intermediate quque to slow MCVA queue
                intermediate_obs = self.intermediate_data_queue.popleft()
                self.slow_moving_average.append(intermediate_obs)

            self.intermediate_data_queue.append(fast_obs)

        self.fast_moving_average.append(obs)

    def get_gating(self):
        fast_mva = sum(self.fast_moving_average) / len(self.fast_moving_average)
        if len(self.slow_moving_average) > self.window_moving_average:
            slow_mva = sum(self.slow_moving_average) / len(self.slow_moving_average)
        else:
            slow_mva = fast_mva

        frac = fast_mva / slow_mva
        return max(min(1, -frac + 2), -0.5)
        
