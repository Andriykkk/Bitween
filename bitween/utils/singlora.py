import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import threading
from bitween.modules import QuantizedLinear
from torch.cuda.amp import autocast
 
class SingLoRALayer(nn.Module):
    """
    Implements the SingLoRA layer as described in the paper.
    This layer wraps a frozen pre-trained layer (e.g., nn.Linear) and
    adds a low-rank update using a single matrix 'A'.

    The weight update is calculated as W = W_0 + alpha/r * u(t) * A @ A.T
    """

    def __init__(
        self,
        original_layer: nn.Module,
        rank: int,
        alpha: float,
        ramp_up_steps: int,
        training: bool = True,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Args:
            original_layer (nn.Module): The pre-trained layer to be adapted.
                                        Must be a nn.Linear layer.
            rank (int): The rank 'r' of the low-rank adaptation.
            alpha (float): The scaling factor for the adaptation.
            ramp_up_steps (int): The number of steps 'T' for the ramp-up function u(t).
        """
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.training = training
        self.ramp_up_steps = ramp_up_steps

        # Freeze the original layer's parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Determine the dimensions for matrix A based on the paper's extension
        # to non-square matrices.
        self.d_out, self.d_in = out_features, in_features
        if self.d_in > self.d_out:
            # If in_features > out_features, swap them for the logic
            self.d_out, self.d_in = self.d_in, self.d_out

        # Create the single low-rank matrix 'A'
        self.A = nn.Parameter(torch.empty(self.d_out, self.rank, device=device))

        # Initialize 'A' using Kaiming uniform distribution, as suggested
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        # Register a buffer for the training step counter 't'
        self.register_buffer(
            "training_step", torch.tensor(0, dtype=torch.float32)
        )

    def _get_update_weight(self) -> torch.Tensor:
        """
        Calculates the low-rank weight update matrix.
        Handles the ramp-up function u(t) and non-square matrices.
        """
        device = self.A.device  # Always use this device

        ramp_up_factor = torch.min(
            self.training_step / self.ramp_up_steps, torch.tensor(1.0, device=device)
        )

        scale = self.alpha / self.rank

        # Calculate A @ A.T
        aa_t = self.A @ self.A.T

        if self.original_layer.in_features > self.original_layer.out_features:
            update = aa_t[
                : self.original_layer.out_features,
                : self.original_layer.in_features,
            ]
        else:
            A_star = self.A[: self.d_in, :]
            update = self.A @ A_star.T

        return ramp_up_factor * scale * update

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SingLoRA layer.
        Uses autocast for speed in both training and inference (safe for training).
        """
        if self.training:
            self.training_step += 1

        # Determine best dtype for GPU
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:    
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.float16

        # Autocast is safe for training - keeps parameters in fp32, only computations in fp16/bf16
        with autocast(enabled=torch.cuda.is_available(), dtype=autocast_dtype):
            original_output = self.original_layer(x)
            update_weight = self._get_update_weight()
            lora_output = F.linear(x, update_weight)
            return original_output + lora_output

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(rank={self.rank}, alpha={self.alpha}, "
            f"ramp_up_steps={self.ramp_up_steps}, "
            f"original_layer={self.original_layer})"
        )
 
def _get_submodule(model, key):
    """Helper to get a submodule from a model by its dot-separated key."""
    parent = model
    for part in key.split('.')[:-1]:
        parent = getattr(parent, part)
    return parent, key.split('.')[-1]

def apply_singlora_to_model(
    model: nn.Module,
    rank: int,
    alpha: float,
    ramp_up_steps: int,
    target_modules: list[str],
    print_summary: bool = True,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Replaces target linear layers in a model with SingLoRALayer.

    Args:
        model (nn.Module): The model to be modified.
        rank (int): The rank for SingLoRA.
        alpha (float): The alpha scaling for SingLoRA.
        ramp_up_steps (int): The T parameter for the ramp-up function.
        target_modules (list[str]): A list of full module names to be replaced.
    """
    for name, module in model.named_modules():
        if name in target_modules:
            parent_module, child_name = _get_submodule(model, name)
            
            # Replace the identified linear layer with a SingLoRALayer
            setattr(
                parent_module,
                child_name,
                SingLoRALayer(module, rank, alpha, ramp_up_steps, device=device),
            )
            
            # Print a summary
            if print_summary:
                print(f"Replaced '{name}' with SingLoRA layer.")


def set_singlora_training_mode(model: nn.Module, training: bool = True):
    """
    Set all SingLoRA layers in the model to training or inference mode.
    
    Args:
        model (nn.Module): The model containing SingLoRA layers.
        training (bool): If True, set to training mode. If False, set to inference mode.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, SingLoRALayer):
            module.train(training)
            count += 1
    
    mode = "training" if training else "inference"
    print(f"Set {count} SingLoRA layers to {mode} mode")


def set_singlora_inference_mode(model: nn.Module):
    """
    Set all SingLoRA layers in the model to inference mode for faster execution.
    
    Args:
        model (nn.Module): The model containing SingLoRA layers.
    """
    set_singlora_training_mode(model, training=False)


def get_singlora_layers(model: nn.Module) -> list[tuple[str, SingLoRALayer]]:
    """
    Get all SingLoRA layers in the model with their names.
    
    Args:
        model (nn.Module): The model to search.
        
    Returns:
        List of (name, layer) tuples for all SingLoRA layers.
    """
    singlora_layers = []
    for name, module in model.named_modules():
        if isinstance(module, SingLoRALayer):
            singlora_layers.append((name, module))
    return singlora_layers