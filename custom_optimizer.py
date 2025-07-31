import math
import torch
from torch.optim import Optimizer
import logging


class CustomAdamW(Optimizer):
    """
    Custom optimizer based on AdamW with McGinley Dynamic-inspired adaptive smoothing.
    
    This optimizer uses dynamic beta values that adapt based on gradient changes,
    inspired by the McGinley Dynamic indicator from technical analysis.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
            These act as base values when dynamic_smoothing is enabled
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay (default: 1e-2)
        amsgrad (bool, optional): whether to use the AMSGrad variant (default: False)
        dynamic_smoothing (bool, optional): whether to use dynamic beta values (default: True)
        min_beta1 (float, optional): minimum value for dynamic beta1 (default: 0.5)
        min_beta2 (float, optional): minimum value for dynamic beta2 (default: 0.9)
        global_scaling (bool, optional): whether to use global gradient norms (default: False)
        log_betas (bool, optional): whether to log beta value statistics (default: False)
        
    Example:
        >>> optimizer = CustomAdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, dynamic_smoothing=True,
                 min_beta1=0.5, min_beta2=0.9, global_scaling=False, log_betas=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= min_beta1 <= betas[0]:
            raise ValueError(f"Invalid min_beta1 value: {min_beta1}, should be between 0 and {betas[0]}")
        if not 0.0 <= min_beta2 <= betas[1]:
            raise ValueError(f"Invalid min_beta2 value: {min_beta2}, should be between 0 and {betas[1]}")
            
        self.dynamic_smoothing = dynamic_smoothing
        self.min_beta1 = min_beta1
        self.min_beta2 = min_beta2
        self.global_scaling = global_scaling
        self.log_betas = log_betas
        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                      weight_decay=weight_decay, amsgrad=amsgrad)
        super(CustomAdamW, self).__init__(params, defaults)
        
        if self.log_betas:
            self.beta_stats = {'beta1_mean': [], 'beta1_min': [], 'beta1_max': [],
                             'beta2_mean': [], 'beta2_min': [], 'beta2_max': []}
    
    def __setstate__(self, state):
        super(CustomAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    
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
                
        # Your custom optimization logic goes here
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            
            # Parameters
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']
            lr = group['lr']
            amsgrad = group['amsgrad']
            
            # Collect parameters and states
            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('CustomAdamW does not support sparse gradients')
                grads.append(p.grad)
                
                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Previous gradient for calculating dynamic beta
                    state['prev_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Previous squared gradient for calculating dynamic beta
                    state['prev_grad_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                
                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                
                # Update the steps for each param group update
                state['step'] += 1
                state_steps.append(state['step'])
                
            # Apply decoupled weight decay
            for i, param in enumerate(params_with_grad):
                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)
                    
                # Update step with McGinley Dynamic-inspired adaptive smoothing
                step = state_steps[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                grad = grads[i]
                
                # Get previous gradients from state
                prev_grad = self.state[params_with_grad[i]]['prev_grad']
                prev_grad_sq = self.state[params_with_grad[i]]['prev_grad_sq']
                
                # Calculate dynamic betas based on gradient changes
                if self.dynamic_smoothing and step > 1:
                    # Calculate changes in gradient and squared gradient
                    if self.global_scaling:
                        # Global scaling across all parameters
                        grad_norm = torch.norm(grad)
                        prev_grad_norm = torch.norm(prev_grad)
                        grad_change = torch.abs(grad_norm - prev_grad_norm) / (prev_grad_norm + eps)
                        
                        grad_sq_norm = torch.norm(grad * grad)
                        prev_grad_sq_norm = torch.norm(prev_grad_sq)
                        grad_sq_change = torch.abs(grad_sq_norm - prev_grad_sq_norm) / (prev_grad_sq_norm + eps)
                    else:
                        # Per-parameter scaling
                        grad_change = torch.norm(grad - prev_grad) / (torch.norm(prev_grad) + eps)
                        grad_sq_change = torch.norm(grad * grad - prev_grad_sq) / (torch.norm(prev_grad_sq) + eps)
                    
                    # Dynamic beta1 based on McGinley Dynamic formula
                    beta1_t = beta1 / (1 + grad_change**2)
                    beta1_t = max(self.min_beta1, beta1_t.item())  # Ensure minimum value for stability
                    
                    # Dynamic beta2 based on McGinley Dynamic formula
                    beta2_t = beta2 / (1 + grad_sq_change**2)
                    beta2_t = max(self.min_beta2, beta2_t.item())  # Ensure minimum value for stability
                else:
                    # Use default betas for first step
                    beta1_t, beta2_t = beta1, beta2
                
                # Save current gradients for next iteration
                self.state[params_with_grad[i]]['prev_grad'].copy_(grad)
                self.state[params_with_grad[i]]['prev_grad_sq'].copy_(grad * grad)
                
                # Calculate bias corrections
                bias_correction1 = 1 - (beta1_t ** step)
                bias_correction2 = 1 - (beta2_t ** step)
                
                # Update moments with dynamic betas
                exp_avg.mul_(beta1_t).add_(grad, alpha=1 - beta1_t)
                exp_avg_sq.mul_(beta2_t).addcmul_(grad, grad, value=1 - beta2_t)
                
                # Log beta statistics if enabled
                if self.log_betas and i == 0:  # Only log for first parameter to avoid clutter
                    if step % 100 == 0:  # Log periodically
                        if step > 1:
                            self.beta_stats['beta1_mean'].append(beta1_t)
                            self.beta_stats['beta2_mean'].append(beta2_t)
                            self.beta_stats['beta1_min'].append(beta1_t)
                            self.beta_stats['beta2_min'].append(beta2_t)
                            self.beta_stats['beta1_max'].append(beta1_t)
                            self.beta_stats['beta2_max'].append(beta2_t)
                            logging.info(f"Step {step}: beta1={beta1_t:.4f}, beta2={beta2_t:.4f}, grad_change={grad_change:.4f}")
                
                # Standard Adam-like update
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                
                step_size = lr / bias_correction1
                
                param.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


# Add these functions to easily integrate the optimizer into train.py
def get_custom_adamw_optimizer(model_parameters, lr=1e-3, betas=(0.9, 0.999), 
                               eps=1e-8, weight_decay=1e-2, amsgrad=False,
                               dynamic_smoothing=True, min_beta1=0.5, min_beta2=0.9,
                               global_scaling=False, log_betas=False):
    """Helper function to create CustomAdamW optimizer
    
    Args:
        model_parameters: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        betas: coefficients for computing running averages (default: (0.9, 0.999))
        eps: term added for numerical stability (default: 1e-8)
        weight_decay: weight decay (default: 1e-2)
        amsgrad: whether to use AMSGrad variant (default: False)
        dynamic_smoothing: whether to use McGinley Dynamic-inspired smoothing (default: True)
        min_beta1: minimum value for dynamic beta1 (default: 0.5)
        min_beta2: minimum value for dynamic beta2 (default: 0.9)
        global_scaling: whether to use global gradient norms (default: False)
        log_betas: whether to log beta value statistics (default: False)
    """
    return CustomAdamW(model_parameters, lr=lr, betas=betas, eps=eps, 
                      weight_decay=weight_decay, amsgrad=amsgrad,
                      dynamic_smoothing=dynamic_smoothing, min_beta1=min_beta1,
                      min_beta2=min_beta2, global_scaling=global_scaling,
                      log_betas=log_betas)
