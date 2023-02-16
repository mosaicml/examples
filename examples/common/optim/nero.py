import torch
from torch.optim.optimizer import Optimizer
import math
from composer.utils import dist

def neuron_norm(x):
    if x.dim() > 1:
        view_shape = [x.shape[0]] + [1]*(x.dim()-1)
        x = x.view(x.shape[0],-1)
        return x.norm(dim=1).view(*view_shape)
    else:
        return x.abs()

def neuron_mean(x):
    if x.dim() > 1:
        view_shape = [x.shape[0]] + [1]*(x.dim()-1)
        x = x.view(x.shape[0],-1)
        return x.mean(dim=1).view(*view_shape)
    else:
        raise Exception("neuron_mean not defined on 1D tensors.")

class Nero(Optimizer):
    metric_functions = {
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
        'percentage_nonzero/grad':
            lambda param, optim_state, step_tensor: (torch.count_nonzero(param.grad) / param.grad.nelement())
            if param.grad.nelement() > 0 else 0,
        'percentage_nonzero/second_moment':
            lambda param, optim_state, step_tensor:
            (torch.count_nonzero(optim_state['exp_avg_sq']) / optim_state['exp_avg_sq'].nelement())
            if optim_state['exp_avg_sq'].flatten().shape[0] > 0 else 0,
    }

    def __init__(self, params, lr=0.01, beta=0.999, constraints=True):
        self.beta = beta
        self.constraints = constraints
        defaults = dict(lr=lr)
        super(Nero, self).__init__(params, defaults)

        

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
            

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    if self.constraints and p.dim() > 1:
                            p.data -= neuron_mean(p)
                            p.data /= neuron_norm(p)
                    state = self.state[p]
                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(neuron_norm(p))
                    state['scale'] = neuron_norm(p).mean()
                    if state['scale'] == 0.0:
                        state['scale'] = 0.01

                if p.grad is None:
                    continue
                
                state['step'] += 1
                bias_correction = 1 - self.beta ** state['step']
                state['exp_avg_sq'] = self.beta * state['exp_avg_sq'] + (1-self.beta) * neuron_norm(p.grad)**2

                grad_normed = p.grad / (state['exp_avg_sq']/bias_correction).sqrt()
                grad_normed[torch.isnan(grad_normed)] = 0
                
                p.data -= group['lr'] * state['scale'] * grad_normed

                if self.constraints and p.dim() > 1:
                    p.data -= neuron_mean(p)
                    p.data /= neuron_norm(p)

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
            if metric.startswith('l2_norm'):
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
                                                                                           
        for group in self.param_groups:
            for param in group['params']:
                param_optim_state = self.state[param]

                bias_correction = 1 - self.beta ** param_optim_state['step']
                step_tensor = param.grad / (param_optim_state['exp_avg_sq']/bias_correction).sqrt()
                step_tensor[torch.isnan(step_tensor)] = 0
                step_tensor.mul_(group['lr']).mul_(param_optim_state['scale'])

                for metric in self.metric_functions:
                    optimizer_metrics[f'{metric}/{name}'] = self.metric_functions[metric](param, param_optim_state,
                                                                                        step_tensor)

        return optimizer_metrics
