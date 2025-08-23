import torch
import torch.optim as optim
from typing import List, Optional


class SignSGD(optim.Optimizer):
    """
    SignSGD optimizer for quantization parameter optimization.
    
    This optimizer is specifically designed for quantization tasks and uses
    the sign of gradients rather than their magnitude, which often works better
    for discrete optimization problems like quantization.
    
    Based on the auto-round paper: "Optimize Weight Rounding via Signed Gradient Descent
    for the Quantization of LLMs"
    """
    
    def __init__(
        self, 
        params, 
        lr: float = 0.005, 
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0
    ):
        """
        Initialize SignSGD optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate (default: 0.005)
            momentum: Momentum factor (default: 0.0)  
            dampening: Dampening for momentum (default: 0.0)
            weight_decay: Weight decay (L2 penalty) (default: 0.0)
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            dampening=dampening,
            weight_decay=weight_decay
        )
        super(SignSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SignSGD, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step using signed gradients.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                param_state = self.state[p]
                
                # State initialization
                if len(param_state) == 0:
                    param_state['momentum_buffer'] = torch.zeros_like(p)
                
                buf = param_state['momentum_buffer']
                
                # Apply momentum with sign of gradient
                if momentum != 0:
                    if len(param_state) > 1:  # Not first step
                        buf.mul_(momentum).add_(torch.sign(grad), alpha=1 - dampening)
                    else:  # First step
                        buf.copy_(torch.sign(grad))
                    
                    # Use sign of momentum buffer
                    update = torch.sign(buf)
                else:
                    # Use sign of gradient directly
                    update = torch.sign(grad)
                
                # Apply update
                p.add_(update, alpha=-group['lr'])

        return loss


class AdaptiveLRScheduler:
    """
    Adaptive learning rate scheduler for quantization optimization.
    
    This scheduler reduces the learning rate when the loss stops improving,
    which helps fine-tune the quantization parameters in later iterations.
    """
    
    def __init__(
        self,
        optimizer: SignSGD,
        patience: int = 20,
        factor: float = 0.8,
        min_lr: float = 1e-6,
        verbose: bool = False
    ):
        """
        Initialize adaptive LR scheduler.
        
        Args:
            optimizer: The SignSGD optimizer to schedule
            patience: Number of iterations to wait before reducing LR
            factor: Factor by which to reduce LR (new_lr = lr * factor)
            min_lr: Minimum learning rate
            verbose: Whether to print LR changes
        """
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.num_bad_iterations = 0
        self.last_iteration = 0
        
    def step(self, loss: float, iteration: int):
        """
        Step the scheduler with current loss.
        
        Args:
            loss: Current loss value
            iteration: Current iteration number
        """
        self.last_iteration = iteration
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.num_bad_iterations = 0
        else:
            self.num_bad_iterations += 1
            
        if self.num_bad_iterations >= self.patience:
            self._reduce_lr()
            self.num_bad_iterations = 0
    
    def _reduce_lr(self):
        """Reduce learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            
            if new_lr < old_lr:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f"Iteration {self.last_iteration}: Reducing LR from {old_lr:.6f} to {new_lr:.6f}")
    
    def get_lr(self) -> List[float]:
        """Get current learning rates for all parameter groups."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class LinearWarmupScheduler:
    """
    Linear warmup scheduler for the initial phase of training.
    
    Gradually increases learning rate from 0 to target LR over warmup_steps,
    then maintains target LR.
    """
    
    def __init__(
        self,
        optimizer: SignSGD,
        target_lr: float,
        warmup_steps: int
    ):
        """
        Initialize linear warmup scheduler.
        
        Args:
            optimizer: The SignSGD optimizer to schedule
            target_lr: Target learning rate after warmup
            warmup_steps: Number of warmup steps
        """
        self.optimizer = optimizer
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        
        # Set initial LR to 0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.0
    
    def step(self, iteration: int):
        """
        Step the warmup scheduler.
        
        Args:
            iteration: Current iteration number (0-indexed)
        """
        if iteration < self.warmup_steps:
            # Linear warmup
            lr = self.target_lr * (iteration + 1) / self.warmup_steps
        else:
            # Maintain target LR
            lr = self.target_lr
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class CombinedScheduler:
    """
    Combined scheduler that applies warmup followed by adaptive LR reduction.
    """
    
    def __init__(
        self,
        optimizer: SignSGD,
        target_lr: float,
        warmup_steps: int = 20,
        patience: int = 30,
        factor: float = 0.8,
        min_lr: float = 1e-6,
        verbose: bool = False
    ):
        """
        Initialize combined scheduler.
        
        Args:
            optimizer: The SignSGD optimizer to schedule
            target_lr: Target learning rate after warmup
            warmup_steps: Number of warmup steps
            patience: Patience for adaptive reduction
            factor: Factor for LR reduction
            min_lr: Minimum learning rate
            verbose: Whether to print LR changes
        """
        self.warmup_scheduler = LinearWarmupScheduler(optimizer, target_lr, warmup_steps)
        self.adaptive_scheduler = AdaptiveLRScheduler(optimizer, patience, factor, min_lr, verbose)
        self.warmup_steps = warmup_steps
        
    def step(self, loss: float, iteration: int):
        """
        Step the combined scheduler.
        
        Args:
            loss: Current loss value  
            iteration: Current iteration number
        """
        if iteration < self.warmup_steps:
            self.warmup_scheduler.step(iteration)
        else:
            self.adaptive_scheduler.step(loss, iteration)
    
    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        return [param_group['lr'] for param_group in self.warmup_scheduler.optimizer.param_groups]