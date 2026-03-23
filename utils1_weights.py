# ════════════════════════════════════════════════════════════════════════════════
# Conv-Tas Process Weights
# ════════════════════════════════════════════════════════════════════════════════

import numpy as np
import torch
import re
from typing import Dict, List, Union, Literal, Tuple, Optional, Any
from pathlib import Path


def load_checkpoints_chronologically(
    checkpoint_dir: str,
    device: str = 'cpu'
) -> List[Tuple[int, Dict[str, torch.Tensor]]]:
    """
    Load all checkpoints from a directory in chronological order.
    
    Expects checkpoint filenames like 'checkpoint_epoch_1.pth', 'checkpoint_epoch_2.pth', etc.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        device: Device to load tensors to
        
    Returns:
        List of tuples: [(epoch_number, state_dict), ...] sorted by epoch
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_path.glob('checkpoint_epoch_*.pth'))
    
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Extract epoch numbers and sort
    def extract_epoch(filepath: Path) -> int:
        """Extract epoch number from filename like 'checkpoint_epoch_42.pth'"""
        match = re.search(r'checkpoint_epoch_(\d+)\.pth', filepath.name)
        if match:
            return int(match.group(1))
        return -1
    
    # Sort by epoch number
    checkpoint_files = sorted(checkpoint_files, key=extract_epoch)
    
    # Load checkpoints
    checkpoints = []
    for filepath in checkpoint_files:
        epoch = extract_epoch(filepath)
        if epoch < 0:
            continue
            
        checkpoint = torch.load(filepath, map_location=device)
        state_dict = checkpoint['model_state_dict']
        checkpoints.append((epoch, state_dict))
        print(f"Loaded checkpoint: epoch {epoch}")
    
    print(f"\nTotal checkpoints loaded: {len(checkpoints)}")
    return checkpoints

def extract_2d_weights_from_state_dict(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Extract all 2D weight matrices from a state dict, excluding normalization layers.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        Dictionary mapping parameter names to 2D tensors
    """
    weights_2d = {}
    norm_keywords = ['bn', 'batch_norm', 'norm', 'ln', 'layer_norm', 'group_norm', 'instance_norm']
    
    for name, param in state_dict.items():
        # Skip normalization layer parameters
        if any(keyword in name.lower() for keyword in norm_keywords):
            continue
        
        # Skip non-weight parameters (biases, etc. that are 1D)
        if not name.endswith('.weight'):
            continue
            
        weight = param.detach().clone()
        squeezed = torch.squeeze(weight)
        if squeezed.ndim == 2:
            weights_2d[name] = squeezed
    
    return weights_2d

def classify_convtasnet_weights(
    weights_2d: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Classify extracted Conv-TasNet weights into two categories:
    1. 1-D Convolutional Kernels (kernel_size > 1)
    2. 1×1 Convolutional Weights (equivalent to linear layers)
    
    Args:
        weights_2d: Dictionary from extract_2d_weights(), mapping names to 2D tensors
        
    Returns:
        conv1d_kernels: Dict of weights that are true 1-D conv kernels
        linear_weights: Dict of weights that are 1×1 convs (equivalent to linear)
    """
    # Pattern matching for 1-D convolutional kernels (kernel_size > 1)
    conv1d_patterns = ['encoder.weight', 'decoder.weight', 'conv_depthwise.weight']
    
    conv1d_kernels = {}
    linear_weights = {}
    
    for name, weight in weights_2d.items():
        is_conv1d = any(pattern in name for pattern in conv1d_patterns)
        
        if is_conv1d:
            conv1d_kernels[name] = weight
        else:
            linear_weights[name] = weight
    
    return conv1d_kernels, linear_weights

def extract_linear_weights_from_checkpoints(
    checkpoints: List[Tuple[int, Dict[str, torch.Tensor]]]
) -> Tuple[List[int], List[Dict[str, torch.Tensor]]]:
    """
    Extract linear weights from all checkpoints.
    
    Args:
        checkpoints: List of (epoch, state_dict) tuples from load_checkpoints_chronologically
        
    Returns:
        epochs: List of epoch numbers
        linear_weights_list: List of linear weight dictionaries (one per checkpoint)
    """
    epochs = []
    linear_weights_list = []
    
    for epoch, state_dict in checkpoints:
        # Extract 2D weights
        weights_2d = extract_2d_weights_from_state_dict(state_dict)
        
        # Classify into conv1d kernels and linear weights
        _, linear_weights = classify_convtasnet_weights(weights_2d)
        
        epochs.append(epoch)
        linear_weights_list.append(linear_weights)
        
    return epochs, linear_weights_list